"""predict-batch: a blind, resumable, strictly sequential batch runner for external validation.

Design rules (OASIS-1 protocol, "staged run"):
  * Blind by construction. The runner opens ONLY `<manifest>/inputs.csv`. Labels live in `labels.csv`, which this
    module never opens; the analysis step reads it after the run is complete. inputs.csv may not carry extra
    columns, so a label cannot be smuggled in.
  * Per-case outputs go to `<run_root>/cases/<case_id>/` and only there. run_root must be outside every git work
    tree (pathguard). Logs and state use case ids only; no scan path, subject identifier or score is ever written
    to stdout or to state.jsonl. Exceptions are recorded by category, never by message.
  * Append-only state.jsonl. The header pins the inputs hash, the git commit (and whether tracked files were
    modified), the artifact-registry hash and the mode. --resume refuses on any mismatch. Completed cases are
    skipped, failed cases are not retried unless asked, an interrupted case is rerun from scratch.
  * Counters-only stdout: integers, no scores, no mode string.
  * Sequential: SynthSeg peaks at up to 13.61 GiB resident (measured), so the mri and dbn workers are stopped before each
    SynthSeg call and MemAvailable is checked against a 15.2 GiB floor; nothing else memory-heavy may run alongside.
  * MRI_ONLY only: the clinical stream cannot be tested on OASIS. The engine is the unchanged engine; ALZ,
    brain age and the specialist are read from its reports by the analysis step.
"""
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import time

from . import labelmaps, memguard, pathguard, registry
from .paths import REPO
from .readers import analyze75
from .report_reader import plumbing_view

INPUT_COLUMNS = ("case_id", "scan_path", "scan_format", "orientation", "age", "sex")
EXTENDED_COLUMNS = INPUT_COLUMNS + ("scan_rank", "volume_source")      # the Stage A manifest; nothing else is accepted
VOLUME_SOURCES = ("synthseg",)                                         # the only volume source this runner supports
SEX = {"M": "Male", "F": "Female", "Male": "Male", "Female": "Female"}
FORMATS = ("analyze75", "nifti")
STAGES = ("plumbing", "full")
REGISTRY_JSON = REPO / "src" / "assessment" / "artifacts" / "registry.json"
SCORE_LIKE = re.compile(r"\b(alz|p_ad|prob\w*|score|gap|stage)\b\D{0,16}\d\.\d+", re.I)


class BatchError(RuntimeError):
    """Refusal or configuration error. Messages never contain subject identifiers, paths or scores."""


class BatchHalt(BatchError):
    """The run stopped cleanly before finishing (memory floor). It can be resumed with --resume."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_inputs(manifest_dir):
    """Rows of inputs.csv sorted by case_id. The only file of the manifest directory that is ever opened."""
    path = pathlib.Path(manifest_dir) / "inputs.csv"
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        cols = tuple(rd.fieldnames or ())
        if cols not in (INPUT_COLUMNS, EXTENDED_COLUMNS):
            raise BatchError("inputs.csv columns must be exactly: " + ",".join(INPUT_COLUMNS) + " or " + ",".join(EXTENDED_COLUMNS))
        rows, seen = [], set()
        for i, r in enumerate(rd, 2):
            cid = pathguard.check_report_id(r["case_id"])
            if cid in seen:
                raise BatchError(f"duplicate case id on line {i}")
            seen.add(cid)
            if r["scan_format"] not in FORMATS:
                raise BatchError(f"line {i}: scan_format must be one of {FORMATS}")
            if r["scan_format"] == "analyze75" and not r["orientation"].strip():
                raise BatchError(f"line {i}: analyze75 needs an explicit orientation")
            try:
                age = float(r["age"])
            except ValueError:
                raise BatchError(f"line {i}: age is not a number") from None
            if not 0 < age < 130 or r["sex"] not in SEX:
                raise BatchError(f"line {i}: age or sex is out of range")
            row = {"case_id": cid, "scan_path": r["scan_path"], "scan_format": r["scan_format"],
                   "orientation": r["orientation"].strip(), "age": age, "sex": SEX[r["sex"]]}
            if cols == EXTENDED_COLUMNS:
                if not re.fullmatch(r"[1-9][0-9]{0,2}", r["scan_rank"]):
                    raise BatchError(f"line {i}: scan_rank must be a positive integer")
                if r["volume_source"] not in VOLUME_SOURCES:
                    raise BatchError(f"line {i}: volume_source must be one of {VOLUME_SOURCES}")
                row.update(scan_rank=int(r["scan_rank"]), volume_source=r["volume_source"])
            rows.append(row)
    if not rows:
        raise BatchError("inputs.csv has no rows")
    return sorted(rows, key=lambda r: r["case_id"])


def _git(args):
    return subprocess.run(["git"] + args, cwd=REPO, capture_output=True, text=True).stdout.strip()


def header_pins(inputs_path, mode, stage, label_map):
    return {"inputs_sha256": sha256_file(inputs_path), "git_commit": _git(["rev-parse", "HEAD"]),
            "git_tracked_modified": bool(_git(["status", "--porcelain", "--untracked-files=no"])),
            "registry_sha256": sha256_file(REGISTRY_JSON), "mode": mode, "stage": stage, "label_map": label_map}


PIN_KEYS = ("inputs_sha256", "git_commit", "registry_sha256", "mode", "stage", "label_map")


class State:
    """Append-only JSON-lines state. Every record is flushed and fsynced; a torn last line is ignored."""

    def __init__(self, path):
        self.path = pathlib.Path(path)

    def append(self, rec):
        rec = {"t": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **rec}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def load(self):
        out = []
        if self.path.exists():
            for ln in self.path.read_text().splitlines():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
        return out


def case_status(records):
    """Last known status per case: 'ok', 'fail' or 'start' (interrupted)."""
    st = {}
    for r in records:
        if r.get("type") == "case":
            st[r["case_id"]] = r
    return st


def case_complete(cases_dir, cid):
    """A case counts as done only if its report exists and parses. A kill -9 between writing files and the state
    record, or a deleted or torn report, makes it not done."""
    try:
        json.loads((pathlib.Path(cases_dir) / cid / "report.json").read_text())
        return True
    except (OSError, ValueError):
        return False


def _error_category(exc):
    if isinstance(exc, ValueError) and "quality control" in str(exc):
        return "QCFail"                       # category only; the message itself is never stored
    return type(exc).__name__


def run(manifest_dir, run_root, *, mode="MRI_ONLY", stage="full", label_map="cdr_3class", limit=None, resume=False,
        retry_failed=False, allow_dirty=False, out=print, engine_factory=None, verify=True,
        expected_shape=analyze75.OASIS_RAW_SHAPE, keep_converted=False,
        min_available_gb=memguard.DEFAULT_FLOOR_GB, release_workers=True, mem_probe=None):
    if mode != "MRI_ONLY":
        raise BatchError("only MRI_ONLY is supported: the clinical stream cannot be tested on this cohort")
    if stage not in STAGES:
        raise BatchError(f"stage must be one of {STAGES}")
    labelmaps.get(label_map)                  # name check only; labels are never read here
    root = pathguard.assert_outside_git(run_root, "run root")
    manifest = pathlib.Path(manifest_dir)
    inputs_path = manifest / "inputs.csv"
    cases = read_inputs(manifest)
    if limit:
        cases = cases[:limit]
    pins = header_pins(inputs_path, mode, stage, label_map)
    if stage == "full" and pins["git_tracked_modified"]:
        raise BatchError("full runs require a clean tree (tracked files are modified)")
    if stage == "plumbing" and pins["git_tracked_modified"] and not allow_dirty:
        raise BatchError("tracked files are modified; pass allow_dirty for a plumbing run")
    root.mkdir(parents=True, exist_ok=True)
    state = State(root / "state.jsonl")
    recs = state.load()
    if recs:
        if not resume:
            raise BatchError("this run root already has a state file; use --resume")
        head = next((r for r in recs if r.get("type") == "header"), None)
        bad = [k for k in PIN_KEYS if head is None or head.get(k) != pins[k]]
        if bad:
            raise BatchError("resume refused, pinned values differ: " + ", ".join(bad))
    else:
        if verify:
            registry.verify()
        state.append({"type": "header", "n_cases": len(cases), **pins})
    done = case_status(state.load())
    cases_dir = root / "cases"
    cases_dir.mkdir(exist_ok=True)
    make_engine = engine_factory or _default_engine
    floor = memguard.floor_check(min_available_gb, mem_probe or memguard.mem_available_gb)   # probe looked up at call time
    n_total = len(cases)
    t0, ran = time.time(), 0

    def counters():
        st = case_status(state.load())
        d = sum(1 for c in cases if st.get(c["case_id"], {}).get("status") == "ok")
        f = sum(1 for c in cases if st.get(c["case_id"], {}).get("status") == "fail")
        eta = int(((time.time() - t0) / ran) * (n_total - d - f) / 60) if ran else 0
        out(f"batch: {d + f}/{n_total} done={d} failed={f} eta_min={eta}")

    with make_engine(run_root=cases_dir, release_around_synthseg=release_workers, before_synthseg=floor) as eng:
        for c in cases:
            cid = c["case_id"]
            prev = done.get(cid)
            if prev and ((prev["status"] == "ok" and case_complete(cases_dir, cid))
                         or (prev["status"] == "fail" and not retry_failed)):
                continue
            shutil.rmtree(cases_dir / cid, ignore_errors=True)          # an interrupted case restarts clean
            state.append({"type": "case", "case_id": cid, "status": "start"})
            t1 = time.perf_counter()
            conv = None
            try:
                scan = c["scan_path"]
                if c["scan_format"] == "analyze75":
                    conv = analyze75.convert_to_nifti(scan, root / "converted" / f"{cid}.nii.gz",
                                                      orientation=c["orientation"], expected_shape=expected_shape)
                    scan = str(conv)
                eng.predict(mode=mode, clinical_values={"AGE": c["age"], "PTGENDER": c["sex"]}, native_path=scan,
                            report_id=cid, subject_label=cid, verify_artifacts=False)
                rp = cases_dir / cid / "report.json"
                json.loads(rp.read_text())                              # must exist and parse
                state.append({"type": "case", "case_id": cid, "status": "ok", "seconds": round(time.perf_counter() - t1, 1)})
            except memguard.MemoryFloorError as e:
                shutil.rmtree(cases_dir / cid, ignore_errors=True)       # nothing half-written is left behind
                state.append({"type": "halt", "reason": "memory_floor", "case_id": cid,
                              "available_gib": round(e.available_gb, 1), "floor_gib": e.floor_gb})
                out(f"batch: halted before a SynthSeg call, MemAvailable {e.available_gb:.1f} GiB < floor {e.floor_gb:.1f} GiB; "
                    "nothing was recomputed or lost, resume with --resume once memory is free")
                raise BatchHalt(str(e)) from None
            except Exception as e:                                       # category only, never the message
                state.append({"type": "case", "case_id": cid, "status": "fail", "error_class": _error_category(e),
                              "seconds": round(time.perf_counter() - t1, 1)})
            finally:
                if conv is not None and not keep_converted:
                    pathlib.Path(conv).unlink(missing_ok=True)
            ran += 1
            counters()
    if verify:
        registry.verify()
    final = case_status(state.load())
    n_ok = sum(1 for c in cases if final.get(c["case_id"], {}).get("status") == "ok")
    n_fail = sum(1 for c in cases if final.get(c["case_id"], {}).get("status") == "fail")
    state.append({"type": "footer", "n_ok": n_ok, "n_failed": n_fail, "n_cases": n_total})
    marker = "RUN_COMPLETE" if stage == "full" else "PLUMBING_COMPLETE"
    (root / marker).write_text(json.dumps({"n_ok": n_ok, "n_failed": n_fail, "n_cases": n_total}) + "\n")
    return {"n_ok": n_ok, "n_failed": n_fail, "n_cases": n_total}


def _default_engine(run_root, **opts):
    from .engine import AssessmentEngine
    return AssessmentEngine(run_root=run_root, guard_outside_git=True, **opts)


def case_ids_under_repo_var(ids):
    """Case ids that appear as a file or folder name anywhere under REPO/var: per-case output must not be there."""
    names = set()
    for d, dirs, files in os.walk(REPO / "var"):
        names.update(dirs)
        names.update(f.split(".")[0] for f in files)
    return sorted(set(ids) & names)


def check_plumbing(run_root):
    """Pass/fail plumbing checks for a finished plumbing run. Returns [(name, bool)]. Reads reports only through
    plumbing_view (no scores) and never prints values."""
    root = pathlib.Path(run_root)
    recs = State(root / "state.jsonl").load()
    st = case_status(recs)
    ids = sorted(st)
    ok_ids = [i for i in ids if st[i]["status"] == "ok"]
    views = {}
    for i in ok_ids:
        try:
            views[i] = plumbing_view(root / "cases" / i / "report.json")
        except Exception:
            pass
    reg = "registration corr_vs_mni"
    logs = list((root / "cases" / "worker_logs").glob("*.log")) if (root / "cases" / "worker_logs").exists() else []
    log_hits = sum(1 for lg in logs for ln in lg.read_text(errors="ignore").splitlines() if SCORE_LIKE.search(ln))
    checks = [
        ("run header present", any(r.get("type") == "header" for r in recs)),
        ("every case finished (ok or failed)", bool(ids) and all(st[i]["status"] in ("ok", "fail") for i in ids)),
        ("no engine failures", all(st[i]["status"] == "ok" for i in ids) and bool(ids)),
        ("reports parse and validate against the schema", bool(ok_ids) and len(views) == len(ok_ids) and all(v["schema_valid"] for v in views.values())),
        ("mode is MRI_ONLY without a stage", bool(views) and all(v["mode"] == "MRI_ONLY" and not v["has_stage"] for v in views.values())),
        ("volume provider is synthseg", bool(views) and all(v["volume_provider"] == "synthseg" for v in views.values())),
        ("registration verdict emitted", bool(views) and all(v["qc_status"].get(reg) in ("PASS", "WARN", "FAIL") for v in views.values())),
        ("ALZ present", bool(views) and all(v["has_alz"] for v in views.values())),
        ("no score-like numbers in worker logs", log_hits == 0),
        ("no converted raw scans left behind", not (root / "converted").exists() or not any((root / "converted").iterdir())),
        ("no case-id-named output under the repository var/", not case_ids_under_repo_var(ids)),
        ("state file has no score fields", not any(k in json.dumps(recs) for k in ("alz", "p_ad", "bag"))),
    ]
    return checks


def purge(run_root):
    """Delete a finished plumbing run (Stage B outputs). Only a directory that holds our state file."""
    root = pathguard.assert_outside_git(run_root, "run root")
    if not (root / "state.jsonl").exists():
        raise BatchError("not a batch run root")
    shutil.rmtree(root)
