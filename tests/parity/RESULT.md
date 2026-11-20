# Parity L1 result

Spec: `tests/parity/PARITY_SPEC.md` (with amendments 1 and 2, 2026-10-28). n = 123 test subjects in the frozen batch order (chunks of 16) + 1 val subject without a scan. Overall: **PASS**.

| Split | Quantity | max abs diff | Tolerance | Pass |
|---|---|---|---|---|
| test | streams (clin/mri/cbF/cbR, extras, cdrsb) | 8.34e-07 | 1e-06 | yes |
| test | has / ba_ok |  | identical | yes |
| test | ALZ | 4.29e-06 | 1e-05 | yes |
| test | reliability flag |  | identical | yes |
| test | brain age | 0.00e+00 | 0.001 | yes |
| test | KAN p | 3.58e-07 | 1e-05 | yes |
| test | KAN add | 1.79e-07 | 1e-05 | yes |
| test | severity | 7.53e-07 | 0.0001 | yes |
| test | predicted stage |  | identical | yes |
| val | streams (clin/mri/cbF/cbR, extras, cdrsb) | 0.00e+00 | 1e-06 | yes |
| val | has / ba_ok |  | identical | yes |
| val | ALZ | 0.00e+00 | 1e-05 | yes |
| val | reliability flag |  | identical | yes |
| val | brain age | 0.00e+00 | 0.001 | yes |
| val | KAN p | 0.00e+00 | 1e-05 | yes |
| val | KAN add | 9.31e-10 | 1e-05 | yes |
| val | severity | 0.00e+00 | 0.0001 | yes |
| val | predicted stage |  | identical | yes |

Why the gate scores all 123 scans in the frozen order: ALZ depends on GPU batch composition (`predict_alz.score` chunks of 16). Diagnostic: `tests/parity/diagnostics/alz_batch_diag.py`. Single-scan (batch size 1) drift, how the app runs, is a known property and not gated: see `tests/parity/DRIFT.md` (`diagnostics/single_scan_drift.py`).
