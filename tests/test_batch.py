"""predict-batch with a fake engine (no workers, no GPU, no scans): blindness, state, resume, refusals, counters-only
output, converted-scan hygiene. Case ids and files are synthetic."""
import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

from src.assessment import batch as B
from src.assessment.paths import REPO
from src.assessment.readers import analyze75 as A

OPENED = []


def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if os.path.basename(p) == "labels.csv":
            OPENED.append(p)


sys.addaudithook(_hook)


class FakeEngine:
    calls = []
    fail = {}

    hook_at_predict = False                 # when True, predict() calls the runner's pre-SynthSeg check like the engine does
    options = []

    def __init__(self, run_root, **kw):
        self.run_root = pathlib.Path(run_root)
        self.before_synthseg = kw.get("before_synthseg")
        FakeEngine.options.append({k: v for k, v in kw.items() if k != "before_synthseg"})

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def predict(self, **kw):
        if FakeEngine.hook_at_predict and self.before_synthseg:
            d0 = self.run_root / kw["report_id"] / "m1"                     # the real engine has written m1 output by now
            d0.mkdir(parents=True, exist_ok=True)
            (d0 / "mni.nii.gz").write_bytes(b"partial")
            self.before_synthseg()
        FakeEngine.calls.append(kw)
        exc = FakeEngine.fail.get(kw["report_id"])
        if exc:
            raise exc
        d = self.run_root / kw["report_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.json").write_text(json.dumps({"meta": {"mode": "MRI_ONLY"}}))
        return None


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    FakeEngine.calls, FakeEngine.fail, FakeEngine.options, FakeEngine.hook_at_predict = [], {}, [], False
    OPENED.clear()
    monkeypatch.setattr(B, "_git", lambda args: "" if args[0] == "status" else "abc123")   # clean tree, fixed commit
    monkeypatch.setattr(B, "sha256_file", B.sha256_file)


def manifest(tmp_path, n=4, fmt="nifti", extra_cols=(), reverse=True, orientation=""):
    d = tmp_path / "manifest"
    d.mkdir(exist_ok=True)
    cols = list(B.INPUT_COLUMNS) + list(extra_cols)
    rows = []
    for i in range(1, n + 1):
        scan = tmp_path / "scans" / f"X{i:04d}"
        rows.append([f"X{i:04d}", str(scan), fmt, orientation, str(70 + i), "F" if i % 2 else "M"] + ["0"] * len(extra_cols))
    if reverse:
        rows = rows[::-1]
    (d / "inputs.csv").write_text("\n".join([",".join(cols)] + [",".join(r) for r in rows]) + "\n")
    return d


def go(tmp_path, man, **kw):
    lines = []
    kw.setdefault("stage", "plumbing")
    kw.setdefault("allow_dirty", True)
    res = B.run(man, tmp_path / "run", engine_factory=lambda run_root, **kw: FakeEngine(run_root, **kw), verify=False,
                out=lines.append, **kw)
    return res, lines


def test_inputs_are_sorted_and_validated(tmp_path):
    rows = B.read_inputs(manifest(tmp_path))
    assert [r["case_id"] for r in rows] == ["X0001", "X0002", "X0003", "X0004"]
    assert rows[0]["sex"] == "Female" and rows[1]["sex"] == "Male" and rows[0]["age"] == 71.0


@pytest.mark.parametrize("bad", ["extra", "dup", "sex", "age", "fmt", "orient", "id"])
def test_inputs_validation_errors(tmp_path, bad):
    d = manifest(tmp_path, extra_cols=("cdr",) if bad == "extra" else (), fmt="analyze75" if bad == "orient" else "nifti")
    p = d / "inputs.csv"
    t = p.read_text()
    if bad == "dup":
        t += t.splitlines()[1] + "\n"
    elif bad == "sex":
        t = t.replace(",F\n", ",X\n", 1)
    elif bad == "age":
        t = re.sub(r",7(\d),", r",abc,", t, count=1)
    elif bad == "fmt":
        t = t.replace(",nifti,", ",dicom,", 1)
    elif bad == "id":
        t = t.replace("X0002", "../x", 1)
    p.write_text(t)
    with pytest.raises((B.BatchError, B.pathguard.GuardError)):
        B.read_inputs(d)


def test_runner_never_opens_labels_file_and_passes_no_label_fields(tmp_path):
    man = manifest(tmp_path)
    lab = man / "labels.csv"
    lab.write_text("case_id,cdr\nX0001,0\n")
    lab.chmod(0)
    OPENED.clear()                                   # creating the file above is the test's own open, not the runner's
    try:
        go(tmp_path, man)
    finally:
        lab.chmod(0o600)
    assert OPENED == []
    assert len(FakeEngine.calls) == 4
    for kw in FakeEngine.calls:
        assert not any(("label" in k and k != "subject_label") or "cdr" in k for k in kw)
        assert kw["mode"] == "MRI_ONLY" and kw["verify_artifacts"] is False
        assert kw["report_id"] == kw["subject_label"] and re.fullmatch(r"X\d{4}", kw["report_id"])
        assert set(kw["clinical_values"]) == {"AGE", "PTGENDER"}


def test_source_never_references_score_view_or_labels_open():
    src = pathlib.Path(B.__file__).read_text()
    assert "score_view" not in src
    assert not re.search(r"open\([^)]*labels", src) and "labels.csv\"" not in src


def test_order_state_marker_and_counters_only_output(tmp_path):
    res, lines = go(tmp_path, manifest(tmp_path))
    assert [c["report_id"] for c in FakeEngine.calls] == ["X0001", "X0002", "X0003", "X0004"]
    assert res == {"n_ok": 4, "n_failed": 0, "n_cases": 4}
    assert all(re.fullmatch(r"batch: \d+/4 done=\d+ failed=\d+ eta_min=\d+", ln) for ln in lines) and len(lines) == 4
    recs = B.State(tmp_path / "run" / "state.jsonl").load()
    assert recs[0]["type"] == "header" and recs[-1]["type"] == "footer"
    txt = (tmp_path / "run" / "state.jsonl").read_text()
    assert str(tmp_path) not in txt and "scan_path" not in txt
    assert json.loads((tmp_path / "run" / "PLUMBING_COMPLETE").read_text()) == res
    assert (tmp_path / "run" / "cases" / "X0003" / "report.json").exists()


def test_failures_are_recorded_by_category_and_not_retried_silently(tmp_path):
    FakeEngine.fail = {"X0002": ValueError("MRI failed quality control and no clinical inputs are available"),
                       "X0003": RuntimeError("boom for " + "OAS" + "1_0001 at /home/someone/data")}
    res, _ = go(tmp_path, manifest(tmp_path))
    assert res == {"n_ok": 2, "n_failed": 2, "n_cases": 4}
    st = B.case_status(B.State(tmp_path / "run" / "state.jsonl").load())
    assert st["X0002"]["error_class"] == "QCFail" and st["X0003"]["error_class"] == "RuntimeError"
    text = (tmp_path / "run" / "state.jsonl").read_text()
    assert "boom" not in text and "home" not in text and ("OAS" + "1_") not in text
    FakeEngine.calls.clear()
    FakeEngine.fail = {}
    res2, _ = go(tmp_path, manifest(tmp_path), resume=True)
    assert FakeEngine.calls == [] and res2["n_failed"] == 2                       # failed cases are not retried
    res3, _ = go(tmp_path, manifest(tmp_path), resume=True, retry_failed=True)
    assert [c["report_id"] for c in FakeEngine.calls] == ["X0002", "X0003"] and res3 == {"n_ok": 4, "n_failed": 0, "n_cases": 4}


def test_resume_skips_done_and_reruns_interrupted_cases(tmp_path):
    man = manifest(tmp_path)
    go(tmp_path, man, limit=2)
    assert [c["report_id"] for c in FakeEngine.calls] == ["X0001", "X0002"]
    # simulate an interruption: X0003 started, never finished, partial output on disk
    s = B.State(tmp_path / "run" / "state.jsonl")
    s.append({"type": "case", "case_id": "X0003", "status": "start"})
    (tmp_path / "run" / "cases" / "X0003").mkdir()
    (tmp_path / "run" / "cases" / "X0003" / "partial.tmp").write_text("x")
    FakeEngine.calls.clear()
    res, _ = go(tmp_path, man, resume=True)
    assert [c["report_id"] for c in FakeEngine.calls] == ["X0003", "X0004"] and res["n_ok"] == 4
    assert not (tmp_path / "run" / "cases" / "X0003" / "partial.tmp").exists()


def test_existing_state_requires_resume_and_pins_must_match(tmp_path, monkeypatch):
    man = manifest(tmp_path)
    go(tmp_path, man, limit=1)
    with pytest.raises(B.BatchError, match="--resume"):
        go(tmp_path, man)
    (man / "inputs.csv").write_text((man / "inputs.csv").read_text().replace("71", "72"))
    with pytest.raises(B.BatchError, match="inputs_sha256"):
        go(tmp_path, man, resume=True)
    (man / "inputs.csv").write_text((man / "inputs.csv").read_text().replace("72", "71"))
    monkeypatch.setattr(B, "_git", lambda args: "" if args[0] == "status" else "def456")
    with pytest.raises(B.BatchError, match="git_commit"):
        go(tmp_path, man, resume=True)
    monkeypatch.setattr(B, "_git", lambda args: "" if args[0] == "status" else "abc123")
    with pytest.raises(B.BatchError, match="stage"):
        go(tmp_path, man, resume=True, stage="full")
    with pytest.raises(B.BatchError, match="label_map"):
        B.labelmaps.register("other_map", classes=("a", "b"), reference="a", positive="b")(lambda r: "a")
        try:
            go(tmp_path, man, resume=True, label_map="other_map")
        finally:
            B.labelmaps.unregister("other_map")
    monkeypatch.setattr(B, "REGISTRY_JSON", pathlib.Path(__file__))
    with pytest.raises(B.BatchError, match="registry_sha256"):
        go(tmp_path, man, resume=True)


def test_refusals(tmp_path, monkeypatch):
    man = manifest(tmp_path)
    with pytest.raises(B.BatchError, match="MRI_ONLY"):
        go(tmp_path, man, mode="FULL")
    with pytest.raises(B.BatchError):
        go(tmp_path, man, stage="nope")
    with pytest.raises(KeyError):
        go(tmp_path, man, label_map="no_such_map")
    with pytest.raises(B.pathguard.GuardError):                                  # run root inside the repo
        B.run(man, REPO / "var" / "batch_test_root", engine_factory=FakeEngine, verify=False, out=print, allow_dirty=True)
    assert not (REPO / "var" / "batch_test_root").exists()
    monkeypatch.setattr(B, "_git", lambda args: "M src/x.py" if args[0] == "status" else "abc123")
    with pytest.raises(B.BatchError, match="clean tree"):
        go(tmp_path, man, stage="full")
    with pytest.raises(B.BatchError, match="modified"):
        go(tmp_path, man, stage="plumbing", allow_dirty=False)


def write_analyze(stem, shape):
    import struct
    h = bytearray(348)
    struct.pack_into("<i", h, 0, 348)
    struct.pack_into("<8h", h, 40, 3, *shape, 1, 1, 1, 1)
    struct.pack_into("<2h", h, 70, 4, 16)
    struct.pack_into("<8f", h, 76, 0.0, 1.0, 1.0, 1.25, 0, 0, 0, 0)
    h[252] = 2
    stem.with_suffix(".hdr").write_bytes(bytes(h))
    stem.with_suffix(".img").write_bytes(np.zeros(shape, np.int16).tobytes(order="F"))


def test_analyze_inputs_are_converted_outside_the_repo_and_removed_afterwards(tmp_path):
    shape = (6, 6, 4)
    man = manifest(tmp_path, n=1, fmt="analyze75", orientation="neurological")
    (tmp_path / "scans").mkdir()
    write_analyze(tmp_path / "scans" / "X0001", shape)
    seen = {}

    class Spy(FakeEngine):
        def predict(self, **kw):
            seen["path"] = kw["native_path"]
            seen["exists"] = os.path.exists(kw["native_path"])
            return super().predict(**kw)

    B.run(man, tmp_path / "run", stage="plumbing", allow_dirty=True, engine_factory=lambda run_root, **kw: Spy(run_root, **kw),
          verify=False, out=lambda s: None, expected_shape=shape)
    assert seen["exists"] and seen["path"].endswith("X0001.nii.gz") and str(REPO) not in seen["path"]
    assert not any((tmp_path / "run" / "converted").iterdir())                    # raw head scan removed after use


def test_analyze_needs_the_exact_expected_shape_by_default(tmp_path):
    man = manifest(tmp_path, n=1, fmt="analyze75", orientation="neurological")
    (tmp_path / "scans").mkdir()
    write_analyze(tmp_path / "scans" / "X0001", (6, 6, 4))
    res, _ = go(tmp_path, man)                                                    # default shape: refused, recorded
    assert res == {"n_ok": 0, "n_failed": 1, "n_cases": 1}
    st = B.case_status(B.State(tmp_path / "run" / "state.jsonl").load())
    assert st["X0001"]["error_class"] == "RefusedInput"


def test_purge_only_removes_a_batch_run_root(tmp_path):
    go(tmp_path, manifest(tmp_path, n=1))
    B.purge(tmp_path / "run")
    assert not (tmp_path / "run").exists()
    (tmp_path / "other").mkdir()
    with pytest.raises(B.BatchError):
        B.purge(tmp_path / "other")


# ---------------- plumbing checks and CLI ----------------

def good_view(**over):
    v = {"schema_valid": True, "mode": "MRI_ONLY", "has_stage": False, "volume_provider": "synthseg",
         "qc_status": {"registration corr_vs_mni": "PASS"}, "has_alz": True}
    v.update(over)
    return v


def test_check_plumbing_pass_and_each_failure(tmp_path, monkeypatch):
    go(tmp_path, manifest(tmp_path, n=2))
    monkeypatch.setattr(B, "plumbing_view", lambda path: good_view())
    res = dict(B.check_plumbing(tmp_path / "run"))
    assert all(res.values()), [k for k, v in res.items() if not v]
    for key, over, name in [("schema_valid", False, "reports parse and validate against the schema"),
                            ("mode", "FULL", "mode is MRI_ONLY without a stage"),
                            ("has_stage", True, "mode is MRI_ONLY without a stage"),
                            ("volume_provider", "adni_fs", "volume provider is synthseg"),
                            ("qc_status", {}, "registration verdict emitted"),
                            ("has_alz", False, "ALZ present")]:
        monkeypatch.setattr(B, "plumbing_view", lambda path, k=key, o=over: good_view(**{k: o}))
        assert dict(B.check_plumbing(tmp_path / "run"))[name] is False, key


def test_check_plumbing_flags_failed_cases_leftover_scans_and_score_like_log_lines(tmp_path, monkeypatch):
    FakeEngine.fail = {"X0002": RuntimeError("x")}
    go(tmp_path, manifest(tmp_path, n=2))
    monkeypatch.setattr(B, "plumbing_view", lambda path: good_view())
    res = dict(B.check_plumbing(tmp_path / "run"))
    assert res["no engine failures"] is False
    (tmp_path / "run" / "converted").mkdir()
    (tmp_path / "run" / "converted" / "X0001.nii.gz").write_text("raw head")
    (tmp_path / "run" / "cases" / "worker_logs").mkdir()
    (tmp_path / "run" / "cases" / "worker_logs" / "mri.log").write_text("timing 3.2 s\nalz = 0.4411 done\n")
    res = dict(B.check_plumbing(tmp_path / "run"))
    assert res["no converted raw scans left behind"] is False and res["no score-like numbers in worker logs"] is False


def test_case_ids_under_repo_var_detected():
    probe = REPO / "var" / "X9999"
    assert B.case_ids_under_repo_var(["X9999"]) == []
    probe.mkdir()
    try:
        assert B.case_ids_under_repo_var(["X9999", "X0001"]) == ["X9999"]
    finally:
        probe.rmdir()


def test_cli_predict_batch_counters_only_and_refusals(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from src.assessment import cli
    man = manifest(tmp_path)
    monkeypatch.setattr(B, "_default_engine", lambda run_root, **kw: FakeEngine(run_root, **kw))
    monkeypatch.setattr(B.registry, "verify", lambda: [])
    r = CliRunner().invoke(cli.app, ["predict-batch", "--manifest", str(man), "--run-root", str(tmp_path / "run"),
                                     "--stage", "plumbing", "--allow-dirty"])
    assert r.exit_code == 0, r.output
    lines = r.output.strip().splitlines()
    assert all(re.fullmatch(r"batch: \d+/4 done=\d+ failed=\d+ eta_min=\d+", ln) for ln in lines[:-1])
    assert lines[-1] == "batch finished: ok=4 failed=0 cases=4"
    assert not re.search(r"\d\.\d", r.output)                                    # no floats anywhere on stdout
    bad = CliRunner().invoke(cli.app, ["predict-batch", "--manifest", str(man), "--run-root",
                                       str(REPO / "var" / "zz_batch"), "--stage", "plumbing", "--allow-dirty"])
    assert bad.exit_code == 2 and "inside the repository" in bad.output
    again = CliRunner().invoke(cli.app, ["predict-batch", "--manifest", str(man), "--run-root", str(tmp_path / "run"),
                                         "--stage", "plumbing", "--allow-dirty"])
    assert again.exit_code == 2 and "--resume" in again.output
    mode = CliRunner().invoke(cli.app, ["predict-batch", "--manifest", str(man), "--run-root", str(tmp_path / "r2"),
                                        "--mode", "FULL"])
    assert mode.exit_code == 2 and "MRI_ONLY" in mode.output
    purge = CliRunner().invoke(cli.app, ["batch-purge", "--run-root", str(tmp_path / "run")])
    assert purge.exit_code == 0 and not (tmp_path / "run").exists()


def _extended(tmp_path, mutate=lambda t: t):
    d = manifest(tmp_path)
    p = d / "inputs.csv"
    lines = p.read_text().splitlines()
    out = [lines[0] + ",scan_rank,volume_source"] + [ln + ",1,synthseg" for ln in lines[1:]]
    p.write_text(mutate("\n".join(out) + "\n"))
    return d


def test_extended_eight_column_manifest_is_accepted(tmp_path):
    rows = B.read_inputs(_extended(tmp_path))
    assert [r["case_id"] for r in rows] == ["X0001", "X0002", "X0003", "X0004"]
    assert rows[0]["scan_rank"] == 1 and rows[0]["volume_source"] == "synthseg" and rows[0]["sex"] == "Female"
    (tmp_path / "legacy").mkdir()
    assert B.read_inputs(manifest(tmp_path / "legacy"))[0].get("scan_rank") is None      # six columns still work


@pytest.mark.parametrize("mutate", [
    lambda t: t.replace(",1,synthseg", ",0,synthseg", 1),            # rank must be positive
    lambda t: t.replace(",1,synthseg", ",1.5,synthseg", 1),
    lambda t: t.replace(",1,synthseg", ",x,synthseg", 1),
    lambda t: t.replace(",synthseg", ",adni_fs", 1),                 # unsupported volume source
    lambda t: t.replace(",synthseg", ",", 1),
    lambda t: t.replace("scan_rank,volume_source", "volume_source,scan_rank"),   # order matters
    lambda t: t.replace("scan_rank,volume_source", "scan_rank,volume_source,CDR").replace(",1,synthseg\n", ",1,synthseg,0\n"),
    lambda t: t.replace(",volume_source", ""),                       # only one of the two extra columns
])
def test_extended_manifest_rejections(tmp_path, mutate):
    with pytest.raises(B.BatchError):
        B.read_inputs(_extended(tmp_path, mutate))


def test_extended_manifest_runs_and_pins_the_file_hash(tmp_path):
    res, _ = go(tmp_path, _extended(tmp_path))
    assert res["n_ok"] == 4 and res["n_failed"] == 0


# ---------------------------------------------------------------- memory floor, release option, resume robustness
def test_runner_passes_release_and_the_floor_check_to_the_engine(tmp_path):
    go(tmp_path, manifest(tmp_path), limit=1)
    assert FakeEngine.options == [{"release_around_synthseg": True}]            # release on by default
    FakeEngine.options.clear()
    (tmp_path / "b").mkdir()
    go(tmp_path / "b", manifest(tmp_path / "b"), limit=1, release_workers=False)
    assert FakeEngine.options == [{"release_around_synthseg": False}]


def test_memory_floor_halts_cleanly_records_state_and_resume_completes(tmp_path):
    man = manifest(tmp_path)
    FakeEngine.hook_at_predict = True
    lines = []
    reads = iter([20.0, 20.0, 3.2])                                             # X0001, X0002 fine; X0003 below the floor
    with pytest.raises(B.BatchHalt):
        B.run(man, tmp_path / "run", stage="plumbing", allow_dirty=True, verify=False, out=lines.append,
              engine_factory=lambda run_root, **kw: FakeEngine(run_root, **kw), min_available_gb=14.5,
              mem_probe=lambda: next(reads))
    recs = B.State(tmp_path / "run" / "state.jsonl").load()
    halt = [r for r in recs if r["type"] == "halt"]
    assert len(halt) == 1 and halt[0]["reason"] == "memory_floor" and halt[0]["available_gib"] == 3.2 and halt[0]["floor_gib"] == 14.5
    assert not any(r["type"] == "footer" for r in recs)
    assert not (tmp_path / "run" / "RUN_COMPLETE").exists() and not (tmp_path / "run" / "PLUMBING_COMPLETE").exists()
    assert not (tmp_path / "run" / "cases" / "X0003").exists()                  # nothing half-written left behind
    st = B.case_status(recs)
    assert st["X0001"]["status"] == st["X0002"]["status"] == "ok" and st["X0003"]["status"] == "start"
    assert [c["report_id"] for c in FakeEngine.calls] == ["X0001", "X0002"]     # the halted case never reached the engine
    assert any("halted" in ln and "GiB" in ln for ln in lines) and not any(re.search(r"X\d{4}", ln) for ln in lines)
    FakeEngine.calls.clear()
    res, _ = go(tmp_path, man, resume=True)                                      # memory is free again (default probe unused by fake)
    assert [c["report_id"] for c in FakeEngine.calls] == ["X0003", "X0004"] and res["n_ok"] == 4


def test_memory_floor_zero_disables_the_check(tmp_path):
    FakeEngine.hook_at_predict = True
    res, _ = go(tmp_path, manifest(tmp_path), min_available_gb=0, mem_probe=lambda: 0.1)
    assert res["n_ok"] == 4


def test_cli_exit_code_3_on_halt_and_sizes_only_in_messages(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from src.assessment import cli, memguard
    man = manifest(tmp_path)
    FakeEngine.hook_at_predict = True
    monkeypatch.setattr(B, "_default_engine", lambda run_root, **kw: FakeEngine(run_root, **kw))
    monkeypatch.setattr(B.registry, "verify", lambda: [])
    monkeypatch.setattr(B.memguard, "mem_available_gb", lambda: 2.0)
    r = CliRunner().invoke(cli.app, ["predict-batch", "--manifest", str(man), "--run-root", str(tmp_path / "run"),
                                     "--stage", "plumbing", "--allow-dirty", "--min-available-gb", "14.5"])
    assert r.exit_code == 3, r.output
    assert "halted" in r.output and "2.0 GiB" in r.output and "14.5 GiB" in r.output
    assert not re.search(r"(alz|prob|score|gap|stage)\D{0,16}\d\.\d", r.output, re.I)
    assert memguard.DEFAULT_FLOOR_GB >= memguard.SYNTHSEG_PEAK_GB + 1.5          # the measured rule behind the default


def test_resume_after_kill_9_redoes_the_half_written_case_and_never_recomputes_finished_ones(tmp_path):
    man = manifest(tmp_path)
    code = """
import json, os, pathlib, signal, sys
sys.path.insert(0, %r)
from src.assessment import batch as B
B._git = lambda args: "" if args[0] == "status" else "abc123"
class Dying:
    def __init__(self, run_root, **kw): self.run_root = pathlib.Path(run_root)
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def predict(self, **kw):
        d = self.run_root / kw["report_id"]; d.mkdir(parents=True, exist_ok=True)
        if kw["report_id"] == "X0003":
            (d / "report.json").write_text('{"meta": {"mod')                       # torn write
            (d / "job").mkdir(); (d / "job" / "streams.npz").write_bytes(b"half")
            os.kill(os.getpid(), signal.SIGKILL)                                     # kill -9 in the middle of the case
        (d / "report.json").write_text(json.dumps({"meta": {"mode": "MRI_ONLY"}, "v": kw["report_id"]}))
B.run(%r, %r, stage="plumbing", allow_dirty=True, verify=False, out=lambda s: None, engine_factory=lambda run_root, **kw: Dying(run_root, **kw))
""" % (str(REPO), str(man), str(tmp_path / "run"))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == -9, p.stderr[-300:]
    cases = tmp_path / "run" / "cases"
    st = B.case_status(B.State(tmp_path / "run" / "state.jsonl").load())
    assert st["X0001"]["status"] == st["X0002"]["status"] == "ok" and st["X0003"]["status"] == "start" and "X0004" not in st
    assert (cases / "X0003" / "job" / "streams.npz").exists()                    # the half-written case is on disk
    before = {c: (cases / c / "report.json").read_bytes() for c in ("X0001", "X0002")}
    FakeEngine.calls.clear()
    res, _ = go(tmp_path, man, resume=True)
    assert [c["report_id"] for c in FakeEngine.calls] == ["X0003", "X0004"]       # finished cases are not recomputed
    assert res == {"n_ok": 4, "n_failed": 0, "n_cases": 4}
    assert not (cases / "X0003" / "job").exists()                                 # fully redone from a clean directory
    assert json.loads((cases / "X0003" / "report.json").read_text())["meta"]["mode"] == "MRI_ONLY"
    assert {c: (cases / c / "report.json").read_bytes() for c in ("X0001", "X0002")} == before


def test_a_case_recorded_ok_with_a_missing_or_torn_report_is_redone(tmp_path):
    man = manifest(tmp_path)
    go(tmp_path, man)
    cases = tmp_path / "run" / "cases"
    (cases / "X0002" / "report.json").unlink()
    (cases / "X0003" / "report.json").write_text('{"meta":')
    FakeEngine.calls.clear()
    res, _ = go(tmp_path, man, resume=True)
    assert sorted(c["report_id"] for c in FakeEngine.calls) == ["X0002", "X0003"] and res["n_ok"] == 4
    assert B.case_complete(cases, "X0002") and B.case_complete(cases, "X0003") and not B.case_complete(cases, "X0099")
