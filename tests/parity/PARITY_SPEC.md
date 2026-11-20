# Parity spec, L1 (committed before the run)

Date: 2026-10-28. This is the merge gate for the engine (lean plan).

## What it checks
The engine (`src/assessment/core.py` and its four workers) reproduces the outputs of the single frozen test run (`analysis/fusion_v3/frozen_run/`, run `adc4ccb`) from the same stored inputs. It checks prediction reproduction only. It computes **no metrics** and reads **no labels**.

## Subjects
Test-split subjects, chosen by this rule before any run: from `frozen_run/test_streams.npz` key `rids` (with `has`), using `numpy.random.default_rng(0)`, draw 16 subjects with a scan and 4 without, without replacement. Total n = 20.

## Inputs (L1: stored artifacts)
- Clinical values and FreeSurfer volumes: `dataset/processed/step2_features.csv`, reading only `RID` plus the 35 source features (never `label`).
- Scans: the stored m1 outputs `dataset/mri_prep/{vol96,mni}/<PTID>.nii.gz`. The PTID comes from `analysis/fusion_v3/scan_inputs_test.csv`.
- Age at scan and the brain-age exclusion (`excluded`): `analysis/fusion_v3/brainage_test.csv`.

## References (allowlisted fields only)
- `frozen_run/test_streams.npz`: `rids, has, ba_ok, extras, cdrsb, clin_*, mri_*, cbF_*, cbR_*`. Never `y`.
- `frozen_run/test_kan.npz`: `p, add`.
- `scan_inputs_test.csv`: `RID, ALZ, unreliable`. `brainage_test.csv`: `RID, pred_age`.

## Tolerances (max |Δ| over the 20 subjects)
| Quantity | Tolerance |
|---|---|
| Stream probabilities (clin, mri, cbF, cbR; all seeds); extras; cdrsb | ≤ 1e-6 |
| has, ba_ok, reliability flag | identical |
| ALZ | ≤ 1e-5 |
| Brain age (pred_age) | ≤ 0.001 y |
| KAN stage probabilities p; AD head add | ≤ 1e-5 |
| Severity Σk·p | ≤ 1e-4 |
| Predicted stage (argmax p) | identical for all 20 |

PASS requires every row. Per-subject differences are written to the gitignored `tests/parity/private/`. Only the pass/fail table with max-Δ values is committed (`tests/parity/RESULT.md`).

## Amendment, 2026-10-28, made before any output was computed
The subject draw failed before any inference ran: all 123 test subjects have a scan, so "4 without a scan" cannot be drawn from test. The new rule:
- **Test:** 20 subjects with a scan, `default_rng(0).choice(rids[has], 20, replace=False)`.
- **No-scan path:** the only validation subject without a scan in `frozen_run/val_streams.npz` (1 of 122). It is compared against `frozen_run/val_streams.npz` / `val_kan.npz` with the same allowlist, and its inputs come from `scan_inputs_val.csv` / `brainage_val.csv`. Same tolerances.

Total n = 21. Nothing else changes.

## Amendment 2, 2026-10-28, made after the first run (FAIL on ALZ) and before the rerun
The first run (n = 20, one batch) failed ALZ (3.95e-5 vs 1e-5) and the MRI-specialist stream (1.04e-5 vs 1e-6). The cause is GPU batch composition: `predict_alz.score` scores in chunks of 16 in list order, so the frozen run's numbers depend on which scans share a chunk. Rerunning the frozen batching reproduces the frozen ALZ (1.3e-6); the diagnostic is `tests/parity/diagnostics/alz_batch_diag.py`. **No tolerance is changed.** The gate becomes:
- **Subjects:** all 123 test scans, in the row order of `scan_inputs_test.csv` (the order `scan_inputs.py` scored them, chunks of 16), plus the single val subject without a scan. This replaces the 20-subject draw.
- **Comparison:** all 123 rows against the committed frozen outputs, allowlisted fields only, no labels, no metrics. Tolerances as declared above.
- **Single-scan drift** (batch size 1, how the app scores) is measured and reported in `RESULT.md` as a known property. It is **not gated**.
- TF32 / cuDNN settings are not altered in the app.
