#!/usr/bin/env python3
"""Stage D2 domain-shift diagnostics for the OASIS-1 main run: applies src/validation/domain_shift.py to the m1 outputs
(vol96, mni) of every cohort case and to the ADNI train-split m1 outputs, and writes ONE aggregate JSON to
docs/validation/ through the aggregate writer (which refuses case ids, OASIS ids and per-subject vectors).

Per-case features stay in memory. Acquisition parameters are read for scan `mpr-1` only (the seven fields below);
no other field of the per-session text is parsed. stdout is the output file name only. Labels are reached only
through `analyze_oasis.assemble` (completion-checked), for the CDR group of each case.

    python tools/oasis_domain_shift.py [--run-root R] [--manifest M] [--meta DIR] [--out docs/validation/oasis_domain_shift.json]
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validation import domain_shift as DS  # noqa: E402

DATA = pathlib.Path(__import__("os").environ.get("OASIS_WORK", "oasis_work"))
FIRST_SCAN = "mpr-1"
ACQ_FIELDS = ("Vox res (mm)", "Rect. Fov", "Orientation", "TR (ms)", "TE (ms)", "TI (ms)", "Flip")
SESSION_RE = re.compile("OAS" + r"1_[0-9]{4}_MR[0-9]+")          # built at run time: this file carries no identifier


def parse_acquisition(text):
    """The acquisition fields of scan `mpr-1` from a per-session text. Other scans and all other fields are ignored."""
    out, cur = {}, None
    for line in text.splitlines():
        if line.startswith("SCAN NUMBER:"):
            cur = line.split(":", 1)[1].strip()
        elif cur == FIRST_SCAN and ":" in line:
            key, val = line.split(":", 1)
            if key.strip() in ACQ_FIELDS:
                out[key.strip()] = val.strip()
    return out


def session_of(scan_path):
    m = SESSION_RE.search(str(scan_path))
    if not m:
        raise ValueError("no session id in a scan path")
    return m.group(0)


def new_columns(rows):
    """rows: dicts with vol96, mni (paths), corr, group (str), acquisition (dict), in case-id order.
    Returns (collected feature columns, groups, acquisition rows)."""
    feats, groups, acq = [], [], []
    for r in rows:
        feats.append(DS.case_features(r["vol96"], r["mni"], corr=r["corr"]))
        groups.append(r["group"])
        acq.append(r["acquisition"])
    return DS.collect(feats), groups, acq


def build_report(rows, ref_cols):
    new, groups, acq = new_columns(rows)
    rep = DS.domain_shift_report(new, ref_cols, acquisition=acq, groups=groups)
    rep["scope"] = {"new": "OASIS-1 main run, all cohort cases, m1 outputs", "reference": "ADNI train split, same code",
                    "n_new": int(len(rows))}
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-root", default=str(DATA / "stageC" / "run"))
    ap.add_argument("--manifest", default=str(DATA / "manifest"))
    ap.add_argument("--meta", default=str(DATA / "meta"))
    ap.add_argument("--out", default="docs/validation/oasis_domain_shift.json")
    a = ap.parse_args(argv)
    from src.assessment import batch
    from src.validation import analyze_oasis as A
    from src.validation.aggregate_io import write_json
    run, meta = pathlib.Path(a.run_root), pathlib.Path(a.meta)
    df = A.assemble(run, a.manifest)
    inputs = {r["case_id"]: r for r in batch.read_inputs(a.manifest)}
    rows = []
    for _, row in df.sort_values("case_id").iterrows():
        cid = row["case_id"]
        text = (meta / f"{session_of(inputs[cid]['scan_path'])}.txt").read_text()
        rows.append({"vol96": run / "cases" / cid / "m1" / "vol96.nii.gz", "mni": run / "cases" / cid / "m1" / "mni.nii.gz",
                     "corr": row["corr_vs_mni"], "group": f"CDR {row['cdr']:g}", "acquisition": parse_acquisition(text)})
    base = ROOT / "dataset" / "mri_prep"
    ref = DS.adni_reference(str(base), base / "mri_index.csv", base / "m1_report_full.csv", split="train")
    dest = write_json(a.out, build_report(rows, ref), case_ids=list(df["case_id"]))
    print("written:", dest.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
