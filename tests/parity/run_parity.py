"""L1 parity run (tests/parity/PARITY_SPEC.md). Prediction reproduction only: no labels, no metrics.

Run with LYP/.venv-app from the repo root:  python -m tests.parity.run_parity
"""
import csv
import json
import os
import pathlib
import sys

import numpy as np

from src.assessment.clinical import stats
from src.assessment.core import run_core, severity
from src.assessment.workers.pool import WorkerPool

ROOT = pathlib.Path(__file__).resolve().parents[2]
FV3 = ROOT / "analysis" / "fusion_v3"
PRIVATE = ROOT / "tests" / "parity" / "private"
STREAM_KEYS = ("rids", "has", "ba_ok", "extras", "cdrsb")
STREAM_PREFIXES = ("clin_", "mri_", "cbF_", "cbR_")
KAN_KEYS = ("p", "add")
FORBIDDEN = {"y", "label", "labels", "dx", "DX", "true"}
TOL = {"streams": 1e-6, "alz": 1e-5, "pred_age": 1e-3, "kan": 1e-5, "severity": 1e-4}


class Allowlisted:
    """npz access restricted to allowlisted keys; any other key (e.g. y) raises."""
    def __init__(self, path, keys=(), prefixes=()):
        self._z = np.load(path)
        self._keys, self._prefixes = set(keys), tuple(prefixes)

    def allowed(self, k):
        return k not in FORBIDDEN and (k in self._keys or k.startswith(self._prefixes))

    def __getitem__(self, k):
        if not self.allowed(k):
            raise KeyError(f"not allowlisted: {k}")
        return self._z[k]

    @property
    def files(self):
        return [k for k in self._z.files if self.allowed(k)]


def read_csv(path, cols):
    """Read only the named columns; asserts none is a label."""
    assert not set(cols) & FORBIDDEN
    with open(path, newline="") as f:
        return [{c: r[c] for c in cols} for r in csv.DictReader(f)]


def num(v):
    return float(v) if v not in ("", None) else float("nan")


def select(ref_test, ref_val):
    """Spec amendment 2 (2026-10-28): all test scans in scan_inputs_test.csv row order (the frozen chunks of 16)
    + the single val subject without a scan."""
    order = [int(r["RID"]) for r in read_csv(FV3 / "scan_inputs_test.csv", ["RID"])]
    assert sorted(order) == sorted(int(r) for r in ref_test["rids"]) and ref_test["has"].all()
    test = order
    vno = [int(r) for r in ref_val["rids"][~ref_val["has"].astype(bool)]]
    assert len(vno) == 1
    return test, vno


def cases(rids, split):
    feats = stats()["num"] + stats()["cat"]
    raw = {int(float(r["RID"])): r for r in read_csv(ROOT / "dataset" / "processed" / "step2_features.csv",
                                                      ["RID"] + feats)}
    si = {int(r["RID"]): r for r in read_csv(FV3 / f"scan_inputs_{split}.csv", ["RID", "PTID", "has_mri"])}
    ba = {int(r["RID"]): r for r in read_csv(FV3 / f"brainage_{split}.csv", ["RID", "PTID", "age_at_scan", "excluded"])}
    prep = ROOT / "dataset" / "mri_prep"
    out = []
    for rid in rids:
        has = si.get(rid, {}).get("has_mri") == "True"
        ptid = si[rid]["PTID"] if has else None
        r = {f: (raw[rid][f] if f == "PTGENDER" else num(raw[rid][f])) for f in feats if raw[rid][f] != ""}
        out.append({"key": str(rid), "raw": r,
                    "vol96": str(prep / "vol96" / f"{ptid}.nii.gz") if has else None,
                    "mni": str(prep / "mni" / f"{ptid}.nii.gz") if has else None,
                    "brain_age": has and ba[rid]["excluded"] != "True",
                    "age_at_scan": num(ba[rid]["age_at_scan"]) if rid in ba else None})
    return out


def maxdiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    if (np.isnan(a) ^ np.isnan(b)).any():
        return float("inf")
    d = np.abs(a - b)[~both_nan]
    return float(d.max()) if d.size else 0.0


def fmt(r):
    return "" if r["max_abs_diff"] is None else f"{r['max_abs_diff']:.2e}"


def compare(split, rids, res, rows):
    ref_s = Allowlisted(FV3 / "frozen_run" / f"{split}_streams.npz", STREAM_KEYS, STREAM_PREFIXES)
    ref_k = Allowlisted(FV3 / "frozen_run" / f"{split}_kan.npz", KAN_KEYS)
    idx = [int(np.flatnonzero(ref_s["rids"] == r)[0]) for r in rids]
    z = np.load(res["streams_npz"])

    def row(name, d, tol=None, identical=None):
        ok = (d <= tol) if identical is None else identical
        rows.append({"split": split, "quantity": name, "max_abs_diff": d if identical is None else None,
                     "tolerance": tol if identical is None else "identical", "pass": bool(ok)})

    keys = [k for k in ref_s.files if k.startswith(STREAM_PREFIXES)] + ["extras", "cdrsb"]
    row("streams (clin/mri/cbF/cbR, extras, cdrsb)", max(maxdiff(z[k], ref_s[k][idx]) for k in keys), TOL["streams"])
    row("has / ba_ok", None, identical=bool((z["has"] == ref_s["has"][idx]).all() and
                                            (z["ba_ok"] == ref_s["ba_ok"][idx]).all()))
    si = {int(r["RID"]): r for r in read_csv(FV3 / f"scan_inputs_{split}.csv", ["RID", "ALZ", "unreliable"])}
    ba = {int(r["RID"]): r for r in read_csv(FV3 / f"brainage_{split}.csv", ["RID", "pred_age"])}
    row("ALZ", maxdiff([a if a is not None else np.nan for a in res["alz"]],
                       [num(si[r]["ALZ"]) if r in si else np.nan for r in rids]), TOL["alz"])
    row("reliability flag", None, identical=all(
        (u is None and si.get(r, {}).get("unreliable", "") == "") or
        (u is not None and str(bool(u)) == si[r]["unreliable"]) for u, r in zip(res["unreliable"], rids)))
    row("brain age", maxdiff([a if a is not None else np.nan for a in res["pred_age"]],
                             [num(ba[r]["pred_age"]) if r in ba else np.nan for r in rids]), TOL["pred_age"])
    p_ref, add_ref = ref_k["p"][idx], ref_k["add"][idx]
    row("KAN p", maxdiff(res["p"], p_ref), TOL["kan"])
    row("KAN add", maxdiff(res["add"], add_ref), TOL["kan"])
    row("severity", maxdiff(severity(res["p"]), severity(p_ref)), TOL["severity"])
    row("predicted stage", None, identical=bool((res["p"].argmax(1) == p_ref.argmax(1)).all()))
    np.savez(PRIVATE / f"per_subject_{split}.npz", rids=np.array(rids), p=res["p"], p_ref=p_ref)


def main():
    test, vno = select(Allowlisted(FV3 / "frozen_run" / "test_streams.npz", STREAM_KEYS, STREAM_PREFIXES),
                       Allowlisted(FV3 / "frozen_run" / "val_streams.npz", STREAM_KEYS, STREAM_PREFIXES))
    PRIVATE.mkdir(parents=True, exist_ok=True)
    pool = WorkerPool(str(PRIVATE / "logs"))
    rows = []
    try:
        for split, rids in (("test", test), ("val", vno)):
            res = run_core(pool, cases(rids, split), str(PRIVATE / f"job_{split}"))
            compare(split, rids, res, rows)
    finally:
        pool.close()
    passed = all(r["pass"] for r in rows)
    (PRIVATE / "result.json").write_text(json.dumps(rows, indent=1))
    lines = ["# Parity L1 result", "", f"Spec: `tests/parity/PARITY_SPEC.md` (with amendments 1 and 2, 2026-10-28). "
             f"n = {len(test)} test subjects in the frozen batch order (chunks of 16) + {len(vno)} val subject without a scan. "
             f"Overall: **{'PASS' if passed else 'FAIL'}**.", "",
             "| Split | Quantity | max abs diff | Tolerance | Pass |", "|---|---|---|---|---|"]
    notes = ["", "Why the gate scores all 123 scans in the frozen order: ALZ depends on GPU batch composition "
             "(`predict_alz.score` chunks of 16). Diagnostic: `tests/parity/diagnostics/alz_batch_diag.py`. "
             "Single-scan (batch size 1) drift, how the app runs, is a known property and not gated: see "
             "`tests/parity/DRIFT.md` (`diagnostics/single_scan_drift.py`)."]
    lines += [f"| {r['split']} | {r['quantity']} | {fmt(r)} | {r['tolerance']} | {'yes' if r['pass'] else 'NO'} |"
              for r in rows]
    print("\n".join(lines))
    (ROOT / "tests" / "parity" / "RESULT.md").write_text("\n".join(lines + notes) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
