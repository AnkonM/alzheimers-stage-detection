"""Known property, not gated: how far single-scan scoring (batch size 1, how the app runs) drifts from the frozen run.

    python -m tests.parity.diagnostics.single_scan_drift     (LYP/.venv-app, repo root)

20 test subjects (default_rng(0) draw from the test scans), each run alone through the engine core, compared with
the committed frozen outputs (allowlisted fields only; no labels, no metrics). Writes tests/parity/DRIFT.md.
"""
import pathlib

import numpy as np

from src.assessment.core import run_core, severity
from src.assessment.workers.pool import WorkerPool
from tests.parity.run_parity import (FV3, KAN_KEYS, PRIVATE, ROOT, STREAM_KEYS, STREAM_PREFIXES, Allowlisted, cases,
                                     num, read_csv)

ref_s = Allowlisted(FV3 / "frozen_run" / "test_streams.npz", STREAM_KEYS, STREAM_PREFIXES)
ref_k = Allowlisted(FV3 / "frozen_run" / "test_kan.npz", KAN_KEYS)
sel = [int(r) for r in np.random.default_rng(0).choice(ref_s["rids"], 20, replace=False)]
si = {int(r["RID"]): r for r in read_csv(FV3 / "scan_inputs_test.csv", ["RID", "ALZ"])}
pool = WorkerPool(str(PRIVATE / "logs_single"))
da, dp, dsev, same = [], [], [], 0
try:
    for i, c in enumerate(cases(sel, "test")):
        res = run_core(pool, [c], str(PRIVATE / "job_single" / str(i)))
        j = int(np.flatnonzero(ref_s["rids"] == sel[i])[0])
        da.append(abs(res["alz"][0] - num(si[sel[i]]["ALZ"])))
        dp.append(float(np.abs(res["p"][0] - ref_k["p"][j]).max()))
        dsev.append(abs(float(severity(res["p"])[0]) - float(severity(ref_k["p"][j:j + 1])[0])))
        same += int(res["p"][0].argmax() == ref_k["p"][j].argmax())
finally:
    pool.close()
txt = (f"# Single-scan drift (batch size 1), not gated\n\n"
       f"n = {len(sel)} test scans, each scored alone through the engine; compared with the frozen run.\n\n"
       f"| Quantity | max | median |\n|---|---|---|\n| ALZ abs diff | {max(da):.2e} | {np.median(da):.2e} |\n"
       f"| KAN stage probability abs diff | {max(dp):.2e} | {np.median(dp):.2e} |\n"
       f"| Severity abs diff | {max(dsev):.2e} | {np.median(dsev):.2e} |\n\n"
       f"Predicted stage identical: {same}/{len(sel)}.\n")
(ROOT / "tests" / "parity" / "DRIFT.md").write_text(txt)
print(txt)
