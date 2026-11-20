# Inventory (step 1) and ground truth (step 2) — counts only, no predictions

Frozen checksum list: 109 / 109 files OK.

The frozen final system was fitted on **TRAIN + VAL** (691 subjects; `freeze.md`), so TRAIN and VAL scans are IN-SAMPLE. Out-of-sample groups: TEST and NOT_IN_COHORT.

- Grid rows (subject x slot): 2760; usable scans: **2554** from **690** subjects
- Baseline scans: 683; follow-up: 1871
- Usable visits per subject: {2: 6, 3: 194, 4: 490}
- Usable scans with a clinical record at the same visit (+-90 d visit): 2430

## Scans and subjects per group

| Group | in-sample | scans | subjects | baseline | follow-up |
|---|---|---|---|---|---|
| TEST | out-of-sample | 399 | 106 | 105 | 294 |
| VAL | IN-SAMPLE | 379 | 104 | 103 | 276 |
| TRAIN | IN-SAMPLE | 1776 | 480 | 475 | 1301 |
| NOT_IN_COHORT | out-of-sample | 0 | 0 | 0 | 0 |

## Unusable grid rows (preprocessing / ingestion failures, reported not dropped)

| Reason | rows |
|---|---|
| NO_ROW | 187 |
| MISSING_NOT_IN_COHORT | 12 |
| DUPLICATE_SCAN | 4 |
| REG_FAILED_LAYOUT | 1 |
| REG_FAILED | 1 |
| DATE_OUT_OF_ORDER | 1 |

## Ground truth, +-90 days (usable scans)

| Group | timepoint | OK | NO_DX_MATCH | UNDEFINED_PLAIN_MCI | UNDEFINED_BASELINE_* | CN | EMCI | LMCI | AD |
|---|---|---|---|---|---|---|---|---|---|
| TEST | baseline | 100 | 5 | 0 | 0 | 25 | 39 | 22 | 14 |
| TEST | follow-up | 127 | 10 | 157 | 0 | 75 | 2 | 0 | 50 |
| VAL | baseline | 101 | 1 | 1 | 0 | 27 | 35 | 22 | 17 |
| VAL | follow-up | 125 | 12 | 139 | 0 | 76 | 0 | 0 | 49 |
| TRAIN | baseline | 463 | 12 | 0 | 0 | 120 | 177 | 102 | 64 |
| TRAIN | follow-up | 545 | 61 | 695 | 0 | 324 | 3 | 1 | 217 |

## Ground truth, +-180 days (usable scans)

| Group | timepoint | OK | NO_DX_MATCH | UNDEFINED_PLAIN_MCI | UNDEFINED_BASELINE_* | CN | EMCI | LMCI | AD |
|---|---|---|---|---|---|---|---|---|---|
| TEST | baseline | 104 | 1 | 0 | 0 | 26 | 42 | 22 | 14 |
| TEST | follow-up | 130 | 0 | 164 | 0 | 76 | 3 | 0 | 51 |
| VAL | baseline | 101 | 1 | 1 | 0 | 27 | 35 | 22 | 17 |
| VAL | follow-up | 131 | 3 | 142 | 0 | 78 | 0 | 0 | 53 |
| TRAIN | baseline | 474 | 1 | 0 | 0 | 121 | 186 | 103 | 64 |
| TRAIN | follow-up | 571 | 11 | 719 | 0 | 338 | 7 | 2 | 224 |

## Clinical record and FreeSurfer source (usable scans, +-90 d)

| Group | timepoint | clinical record | no clinical record | FS step2 | FS IMAGEUID | FS VISCODE2 | FS unavailable |
|---|---|---|---|---|---|---|---|
| TEST | baseline | 100 | 5 | 104 | 0 | 0 | 1 |
| TEST | follow-up | 280 | 14 | 0 | 261 | 32 | 1 |
| VAL | baseline | 102 | 1 | 103 | 0 | 0 | 0 |
| VAL | follow-up | 260 | 16 | 0 | 237 | 35 | 4 |
| TRAIN | baseline | 463 | 12 | 475 | 0 | 0 | 0 |
| TRAIN | follow-up | 1225 | 76 | 0 | 1094 | 192 | 15 |

## Per-feature missingness before frozen imputation (usable scans with a clinical record)

| Feature | baseline | follow-up |
|---|---|---|
| CDRSB | 0.000 | 0.000 |
| ADAS11 | 0.003 | 0.003 |
| ADAS13 | 0.006 | 0.010 |
| MMSE | 0.000 | 0.001 |
| RAVLT_immediate | 0.000 | 0.005 |
| RAVLT_learning | 0.000 | 0.005 |
| RAVLT_forgetting | 0.000 | 0.008 |
| RAVLT_perc_forgetting | 0.000 | 0.010 |
| FAQ | 0.002 | 0.002 |
| MOCA | 0.005 | 0.008 |
| EcogPtMem | 0.003 | 0.014 |
| EcogPtLang | 0.003 | 0.014 |
| EcogPtVisspat | 0.008 | 0.028 |
| EcogPtPlan | 0.003 | 0.015 |
| EcogPtOrgan | 0.026 | 0.032 |
| EcogPtDivatt | 0.011 | 0.021 |
| EcogPtTotal | 0.003 | 0.014 |
| EcogSPMem | 0.003 | 0.011 |
| EcogSPLang | 0.003 | 0.010 |
| EcogSPVisspat | 0.026 | 0.035 |
| EcogSPPlan | 0.011 | 0.022 |
| EcogSPOrgan | 0.045 | 0.054 |
| EcogSPDivatt | 0.024 | 0.029 |
| EcogSPTotal | 0.005 | 0.010 |
