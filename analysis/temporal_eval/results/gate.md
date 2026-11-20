# Sanity gate (PLAN.md §7) — status 2026-10-11: **STOPPED at G1c**

No stage prediction has been computed on the temporal ingestion. No metric exists.

| Check | Rule | Result | Status |
|---|---|---|---|
| Frozen checksums | 109 / 109 | 109 / 109 OK | pass |
| G1a streams | wrapper on stored TEST inputs vs `frozen_run/test_streams.npz`, max \|diff\| ≤ 1e-6 | max \|diff\| = 0 on every key (123 subjects) | **pass** |
| G1a KAN | vs `frozen_run/test_kan.npz` (`p`, `add`), ≤ 1e-6, same argmax | max \|dp\| = 0, max \|d add\| = 0, argmax identical | **pass** |
| G1b ALZ | original TEST volumes vs `scan_inputs_test.csv`, ≤ 1e-5 | 4.29e-6 (123); reliability flags identical | **pass** |
| G1b brain age | original TEST mni vs `brainage_test.csv`, ≤ 0.001 y | 0.00000 y (123) | **pass** |
| ALZ provenance, TRAIN | out-of-fold model vs `alz_crossfit.csv`, ≤ 1e-5 (first 20) | 1.1e-16 | pass — TRAIN ALZ is the cross-fitted out-of-fold value |
| ALZ provenance, VAL | 5-fold mean vs `alz_crossfit.csv`, ≤ 1e-5 (first 20) | 3.5e-5 (GPU batched), 3.1e-5 (GPU alone), 1.8e-4 (CPU); nearest alternative rule (a single fold model) is 0.082 off | tolerance missed; rule confirmed (5-fold mean). Numeric only |
| G1c builder | per feature, \|diff\| ≤ 1e-6 or both missing, ≥ 98 % of the 814 | 21 / 24 features 100 %; **ADAS11 98.9 %, ADAS13 94.5 %, RAVLT_perc_forgetting 53.1 %** | **FAIL (as written)** |

## G1c: what differs

- **RAVLT_perc_forgetting (382 subjects differ):** `step2_features.csv` carries
  ADSL's value, rounded to 2 decimals (e.g. 28.57); the builder computes the
  unrounded 100·(AVTOT5 − AVDEL30MIN)/AVTOT5. Differences 0.0008–0.0046
  points. After rounding to 2 decimals: 99.5 % agree.
- **ADAS11 / ADAS13 (9 / 45 differ):** most are float-representation
  differences of ~2e-6 (e.g. 38.330002 vs 38.33). After rounding to 2
  decimals: ADAS11 99.4 %, ADAS13 99.0 %. The remaining 0.6 % / 1.0 % are true
  value differences between ADSL and `ADAS_ADNIGO2.csv`; ADNI's own ADQS
  baseline values show the same 99.0 % agreement with ADSL.
- Cause is numeric formatting, not the documented earliest-visit fallback, so
  the plan's exception does not apply and the rule says STOP.
- Scope: the builder supplies clinical records only for follow-up scans
  (baseline scans use the exact `step2_features.csv` rows), so the
  difference affects follow-up inputs only.

## G2

Not run (it needs ingestion-wide ALZ / brain age, which follow the gate).
