# EXP2 — Reduced clinical inputs: results

## Run metadata
- Branch `exp/reduced-clinical` (from `9bac529`) / commit (code) `634f57a` (commit 2; evaluation code added in commit 3) / commit (preregistration) `de368a1` / addendum `3744036` / commit 3: given in the hand-over message
- Date 2026-10-09 / total runtime ≈ 13.5 min (Phase 3: 677 s for 6 tiers; Phase 2 MRI + Full-paper gate run: 126 s; Phase 4 evaluation: 10 s) / seeds 42–46 (outer) × KAN seeds 42–46 / bootstrap 2000 resamples, rng seed 0 / hardware AMD Ryzen 7 260 (16 threads), 15 GB RAM, WSL2 Linux 6.18; CatBoost/SHAP in LYP/.venv-fusion, pykan 0.2.8 / torch 2.14.1+cpu in LYP/.venv-kan
- Primary metric: spec §7 (metrics of seed-averaged OOF probabilities). CV n = 569 (M: 567 with a scan); val n = 122 (M: 121); val LMCI = 25.

## Phase 0–1 audit
- Logical Memory present in frozen feature set: **NO** (columns: `LDELTOTL`, `LIMMTOTL` exist only in the unused joined tables step0/step1; 0% missing in train)
- Label-defining variables in frozen 12-feature set: CDRSB, CogIndex (MMSE − CDRSB − ADAS13, z-scored), MMSE in every seed's top-12; the frozen KAN also takes CDRSB, CogIndex, MMSE as direct inputs
- Label reconstructability (CV on 569 train; inputs {MMSE, CDGLOBAL, LDELTOTL, PTEDUCAT}; all 569 have all four):

| Model | macro-F1 | acc | F1 CN / EMCI / LMCI / AD | confusion (rows true CN/EMCI/LMCI/AD) |
|---|---|---|---|---|
| Depth-3 tree | 0.838 | 0.866 | 0.99 / 0.89 / 0.60 / 0.87 | 131 2 0 0 ; 1 207 6 3 ; 0 35 52 27 ; 0 2 0 103 |
| Multinomial logistic | 0.874 | 0.891 | 0.99 / 0.93 / 0.75 / 0.82 | 131 2 0 0 ; 1 210 5 1 ; 0 19 84 11 ; 0 2 21 82 |

- Rule-based ADNI-GO/2 cut-offs on 691 train+val: uniquely matched 575, ambiguous 108 (LMCI+AD 71, EMCI+AD 37), unmatched 8. All 691 (ambiguous/unmatched = error): acc 0.823, macro-F1 0.887. Uniquely matched only: acc 0.990, macro-F1 0.990; confusion 159 2 0 0 ; 0 222 2 1 ; 0 1 95 0 ; 0 0 0 93
- Cut-off source: checked against the ADNI-2 Protocol (ADC-039), pp. 3 and 18–19 (all values match; the procedures manual itself was not consulted). The protocol's CDR memory-box criterion (CN 0, MCI ≥ 0.5) was omitted from the rule: 2 CN and 1 MCI subjects would fail it. Our scores are baseline-visit values while the cut-offs applied at screening, so some disagreement is expected.
- Earlier fusion-feasibility ablation: found at `analysis/fusion_feasibility/ablation.py` (arm A′/B′); variables it removed: CDRSB, MMSE and CogIndex only, from the 36 step-4 features; the other 33, including the 7 raw FreeSurfer volumes and ADAS11/13, stayed in (no Logical Memory was ever in them). Its result was not reused.

## Tier definitions
| Tier | n features | Columns | Max missingness (train) |
|---|---|---|---|
| T0 | 3 (top-3; KAN width 12) | AGE, PTGENDER, PTEDUCAT | 0.000 |
| T1 | 18 (top-12; KAN width 13) | T0 + FAQ + EcogPt×7 + EcogSP×7 | 0.049 |
| T2 | 19 (top-12; KAN width 13) | T1 + MOCA | 0.049 |
| T3 | 27 (top-12; KAN width 13) | T2 + ADAS11, ADAS13, Q4SCORE, RAVLT×4, TRABSCOR | 0.049 |
| Full-spec | 29 (top-12; KAN width 13) | paper's 29 (raw, in-fold step-4 rule) | 0.049 |
| S-T2+APOE4 | 20 (top-12; KAN width 13) | T2 + APOE4 (sensitivity) | 0.049 |
| Full-paper | 29 (top-12; KAN width 13) | paper's 29 (step-4 table), fixed KAN extras | 0.000 |

## Reproduction gate
| System | Paper CV macro-F1 | Reproduced | Δ | PASS/FAIL |
|---|---|---|---|---|
| C (§7, seed-averaged probabilities) | 0.744 | 0.7415 | -0.0025 | PASS |
| C (paper-style, mean of per-seed) | 0.744 | 0.7435 | -0.0005 | PASS |
| F-KAN (§7, seed-averaged probabilities) | 0.756 | 0.7627 | +0.0067 | PASS |
| F-KAN (paper-style, mean of per-seed) | 0.756 | 0.7562 | +0.0002 | PASS |

Experiment code = saved paper predictions (max |diff| 0.0, CV and val). Δ F-KAN − C at Full-paper: §7 +0.021 [-0.006, +0.050]; paper-style (mean of per-seed) +0.013 [-0.006, +0.034]. Full-spec: §7 +0.025 [-0.000, +0.050]; paper-style +0.012 [-0.007, +0.030] (C 0.747, F-KAN 0.759).

## Main results — CV (569 train, seed-averaged OOF)
| Tier | C macro-F1 | M macro-F1 | F-KAN macro-F1 | F-CB macro-F1 | Δ F-KAN−C [95% CI] | Δ F-CB−C [95% CI] |
|---|---|---|---|---|---|---|
| T0 | 0.315 | 0.420 | 0.451 | 0.419 | +0.136 [+0.087, +0.190] | +0.104 [+0.050, +0.159] |
| T1 | 0.566 | 0.420 | 0.586 | 0.602 | +0.020 [-0.020, +0.059] | +0.036 [+0.003, +0.069] |
| T2 | 0.590 | 0.420 | 0.614 | 0.610 | +0.024 [-0.014, +0.060] | +0.021 [-0.015, +0.053] |
| T3 | 0.637 | 0.420 | 0.631 | 0.641 | -0.006 [-0.039, +0.027] | +0.004 [-0.026, +0.033] |
| Full-spec | 0.743 | 0.420 | 0.768 | 0.764 | +0.025 [-0.000, +0.050] | +0.021 [-0.003, +0.044] |
| Full-paper (ref.) | 0.741 | 0.420 | 0.763 | 0.767 | +0.021 [-0.006, +0.050] | +0.026 [+0.004, +0.050] |

| Tier | C AD-AUC | M AD-AUC | F-KAN AD-AUC | Δ F-KAN−C [95% CI] |
|---|---|---|---|---|
| T0 | 0.480 | 0.881 | 0.878 | +0.398 [+0.331, +0.467] |
| T1 | 0.921 | 0.881 | 0.950 | +0.029 [+0.013, +0.047] |
| T2 | 0.950 | 0.881 | 0.963 | +0.014 [+0.005, +0.023] |
| T3 | 0.963 | 0.881 | 0.964 | +0.001 [-0.003, +0.006] |
| Full-spec | 0.985 | 0.881 | 0.985 | -0.000 [-0.004, +0.003] |
| Full-paper (ref.) | 0.984 | 0.881 | 0.985 | +0.001 [-0.002, +0.004] |

| Tier | C EMCI-vs-LMCI AUC | M | F-KAN | Δ [95% CI] |
|---|---|---|---|---|
| T0 | 0.632 | 0.699 | 0.700 | +0.068 [-0.006, +0.145] |
| T1 | 0.633 | 0.699 | 0.707 | +0.074 [+0.016, +0.135] |
| T2 | 0.657 | 0.699 | 0.712 | +0.055 [+0.006, +0.106] |
| T3 | 0.728 | 0.699 | 0.754 | +0.026 [-0.009, +0.061] |
| Full-spec | 0.756 | 0.699 | 0.770 | +0.014 [-0.020, +0.045] |
| Full-paper (ref.) | 0.756 | 0.699 | 0.772 | +0.016 [-0.017, +0.049] |

| Tier | C CN-vs-MCI AUC | M | F-KAN | Δ [95% CI] |
|---|---|---|---|---|
| T0 | 0.603 | 0.719 | 0.725 | +0.122 [+0.068, +0.175] |
| T1 | 0.866 | 0.719 | 0.877 | +0.011 [-0.001, +0.024] |
| T2 | 0.887 | 0.719 | 0.897 | +0.010 [-0.001, +0.022] |
| T3 | 0.888 | 0.719 | 0.889 | +0.001 [-0.009, +0.012] |
| Full-spec | 0.990 | 0.719 | 0.992 | +0.002 [-0.001, +0.007] |
| Full-paper (ref.) | 0.990 | 0.719 | 0.992 | +0.001 [-0.002, +0.006] |

M is tier-independent (567 subjects with a scan).

## Validation holdout (fit on 569, evaluate on 122)
| Tier | C macro-F1 | F-KAN macro-F1 | Δ [95% CI] | C AD-AUC | F-KAN AD-AUC | Δ [95% CI] |
|---|---|---|---|---|---|---|
| T0 | 0.301 | 0.410 | +0.109 [+0.004, +0.209] | 0.608 | 0.823 | +0.215 [+0.070, +0.373] |
| T1 | 0.525 | 0.599 | +0.074 [-0.005, +0.153] | 0.939 | 0.949 | +0.011 [-0.016, +0.044] |
| T2 | 0.543 | 0.599 | +0.056 [-0.031, +0.144] | 0.949 | 0.957 | +0.008 [-0.007, +0.025] |
| T3 | 0.627 | 0.600 | -0.028 [-0.097, +0.043] | 0.967 | 0.961 | -0.006 [-0.018, +0.004] |
| Full-spec | 0.727 | 0.761 | +0.034 [-0.022, +0.099] | 0.996 | 0.995 | -0.001 [-0.006, +0.003] |
| Full-paper (ref.) | 0.727 | 0.761 | +0.034 [-0.022, +0.099] | 0.996 | 0.995 | -0.001 [-0.006, +0.003] |

M on val (121): macro-F1 0.391, AD-AUC 0.811.

## Per-class F1 (CV) at T1, T2, Full
| Tier | System | CN | EMCI | LMCI | AD |
|---|---|---|---|---|---|
| T1 | C | 0.689 | 0.630 | 0.236 | 0.708 |
| all tiers | M | 0.385 | 0.601 | 0.131 | 0.561 |
| T1 | F-KAN | 0.712 | 0.540 | 0.342 | 0.748 |
| T1 | F-CB | 0.689 | 0.658 | 0.337 | 0.724 |
| T2 | C | 0.713 | 0.638 | 0.287 | 0.720 |
| T2 | F-KAN | 0.740 | 0.563 | 0.400 | 0.755 |
| T2 | F-CB | 0.694 | 0.656 | 0.344 | 0.748 |
| Full-spec | C | 0.945 | 0.779 | 0.389 | 0.860 |
| Full-spec | F-KAN | 0.942 | 0.762 | 0.500 | 0.868 |
| Full-spec | F-CB | 0.949 | 0.790 | 0.456 | 0.861 |
| Full-paper (ref.) | C | 0.945 | 0.780 | 0.385 | 0.856 |
| Full-paper (ref.) | F-KAN | 0.930 | 0.760 | 0.502 | 0.858 |
| Full-paper (ref.) | F-CB | 0.949 | 0.795 | 0.456 | 0.869 |

## MRI masking inside fusion (CV)
| Tier | Δ macro-F1 (F-KAN − F-KAN-noMRI) [95% CI] | Δ macro-F1 (F-KAN − F-KAN-mask-inf) [95% CI] | Δ AD-AUC (F-KAN − F-KAN-noMRI) [95% CI] |
|---|---|---|---|
| T0 | +0.137 [+0.094, +0.180] | +0.134 [+0.086, +0.180] | +0.291 [+0.232, +0.353] |
| T1 | +0.015 [-0.020, +0.049] | +0.034 [+0.003, +0.067] | +0.023 [+0.007, +0.040] |
| T2 | +0.022 [-0.006, +0.050] | +0.027 [-0.000, +0.056] | +0.010 [+0.002, +0.019] |
| T3 | -0.018 [-0.042, +0.006] | -0.010 [-0.034, +0.013] | +0.006 [+0.002, +0.010] |
| Full-spec | +0.007 [-0.010, +0.025] | +0.015 [-0.002, +0.032] | +0.000 [-0.002, +0.002] |
| Full-paper (ref.) | +0.006 [-0.010, +0.023] | +0.019 [+0.001, +0.037] | +0.000 [-0.002, +0.002] |

F-KAN-noMRI: retrained with MRI masked in training and evaluation (paper's control). F-KAN-mask-inf: the same F-KAN models with MRI masked at inference only.

## E1 — Rule-ambiguity subsets (exploratory; addendum `3744036`; descriptive, no verdict)
- Train composition: AMBIGUOUS n=92 (CN 0, EMCI 28, LMCI 36, AD 28; scan 92); UNAMBIGUOUS n=469 (CN 133, EMCI 186, LMCI 77, AD 73; scan 467); UNMATCHED n=8 (CN 0, EMCI 3, LMCI 1, AD 4; scan 8)
- Val composition: AMBIGUOUS n=16 (CN 0, EMCI 7, LMCI 6, AD 3; scan 16); UNAMBIGUOUS n=106 (CN 28, EMCI 39, LMCI 19, AD 20; scan 105); UNMATCHED n=0 (CN 0, EMCI 0, LMCI 0, AD 0; scan 0)

| Eval | Subset | Tier | macro-F1* C / M / F-KAN / F-CB | AD-AUC C / M / F-KAN / F-CB | Δ F-KAN−C macro-F1* | Δ F-KAN−C AD-AUC |
|---|---|---|---|---|---|---|
| CV | AMBIGUOUS (n=92) | T0 | 0.354 / 0.325 / 0.404 / 0.360 | 0.333 / 0.640 / 0.665 / 0.628 | +0.051 [-0.072, +0.182] | +0.333 [+0.189, +0.485] |
| CV | AMBIGUOUS (n=92) | T1 | 0.399 / 0.325 / 0.447 / 0.425 | 0.734 / 0.640 / 0.776 / 0.737 | +0.048 [-0.056, +0.156] | +0.041 [-0.010, +0.099] |
| CV | AMBIGUOUS (n=92) | T2 | 0.459 / 0.325 / 0.498 / 0.437 | 0.694 / 0.640 / 0.748 / 0.703 | +0.038 [-0.067, +0.142] | +0.054 [+0.005, +0.104] |
| CV | AMBIGUOUS (n=92) | T3 | 0.432 / 0.325 / 0.423 / 0.488 | 0.714 / 0.640 / 0.746 / 0.729 | -0.009 [-0.099, +0.084] | +0.032 [+0.004, +0.064] |
| CV | AMBIGUOUS (n=92) | Full-spec | 0.464 / 0.325 / 0.528 / 0.505 | 0.732 / 0.640 / 0.755 / 0.746 | +0.063 [-0.028, +0.165] | +0.023 [-0.014, +0.061] |
| CV | AMBIGUOUS (n=92) | S-T2+APOE4 | 0.462 / 0.325 / 0.488 / 0.459 | 0.701 / 0.640 / 0.741 / 0.709 | +0.026 [-0.079, +0.127] | +0.040 [-0.013, +0.095] |
| CV | AMBIGUOUS (n=92) | Full-paper (ref.) | 0.460 / 0.325 / 0.518 / 0.529 | 0.723 / 0.640 / 0.742 / 0.742 | +0.057 [-0.032, +0.153] | +0.018 [-0.020, +0.058] |
| CV | UNAMBIGUOUS (n=469) | T0 | 0.313 / 0.442 / 0.460 / 0.434 | 0.516 / 0.913 / 0.907 / 0.912 | +0.147 [+0.094, +0.202] | +0.391 [+0.310, +0.477] |
| CV | UNAMBIGUOUS (n=469) | T1 | 0.588 / 0.442 / 0.613 / 0.632 | 0.949 / 0.913 / 0.974 / 0.970 | +0.026 [-0.020, +0.072] | +0.024 [+0.005, +0.047] |
| CV | UNAMBIGUOUS (n=469) | T2 | 0.597 / 0.442 / 0.625 / 0.635 | 0.982 / 0.913 / 0.988 / 0.985 | +0.029 [-0.013, +0.071] | +0.006 [+0.000, +0.013] |
| CV | UNAMBIGUOUS (n=469) | T3 | 0.682 / 0.442 / 0.677 / 0.672 | 0.992 / 0.913 / 0.991 / 0.992 | -0.005 [-0.039, +0.031] | -0.001 [-0.004, +0.002] |
| CV | UNAMBIGUOUS (n=469) | Full-spec | 0.794 / 0.442 / 0.807 / 0.810 | 1.000 / 0.913 / 1.000 / 1.000 | +0.013 [-0.011, +0.038] | -0.000 [-0.000, +0.000] |
| CV | UNAMBIGUOUS (n=469) | S-T2+APOE4 | 0.608 / 0.442 / 0.632 / 0.626 | 0.982 / 0.913 / 0.988 / 0.985 | +0.024 [-0.012, +0.061] | +0.006 [+0.001, +0.013] |
| CV | UNAMBIGUOUS (n=469) | Full-paper (ref.) | 0.794 / 0.442 / 0.803 / 0.808 | 1.000 / 0.913 / 1.000 / 1.000 | +0.009 [-0.019, +0.038] | -0.000 [-0.000, +0.000] |
| VAL | AMBIGUOUS (n=16) | T0 | 0.188 / 0.312 / 0.236 / 0.217 | 0.308 / 0.410 / 0.179 / 0.282 | +0.048 | -0.128 |
| VAL | AMBIGUOUS (n=16) | T1 | 0.157 / 0.312 / 0.272 / 0.291 | 0.718 / 0.410 / 0.615 / 0.667 | +0.115 | -0.103 |
| VAL | AMBIGUOUS (n=16) | T2 | 0.289 / 0.312 / 0.190 / 0.257 | 0.564 / 0.410 / 0.513 / 0.564 | -0.098 | -0.051 |
| VAL | AMBIGUOUS (n=16) | T3 | 0.333 / 0.312 / 0.354 / 0.278 | 0.590 / 0.410 / 0.564 / 0.564 | +0.020 | -0.026 |
| VAL | AMBIGUOUS (n=16) | Full-spec | 0.415 / 0.312 / 0.473 / 0.658 | 0.821 / 0.410 / 0.795 / 0.821 | +0.058 | -0.026 |
| VAL | AMBIGUOUS (n=16) | S-T2+APOE4 | 0.483 / 0.312 / 0.244 / 0.311 | 0.564 / 0.410 / 0.487 / 0.538 | -0.238 | -0.077 |
| VAL | AMBIGUOUS (n=16) | Full-paper (ref.) | 0.415 / 0.312 / 0.473 / 0.658 | 0.821 / 0.410 / 0.795 / 0.821 | +0.058 | -0.026 |
| VAL | UNAMBIGUOUS (n=106) | T0 | 0.315 / 0.392 / 0.427 / 0.385 | 0.650 / 0.862 / 0.880 / 0.874 | +0.112 [+0.007, +0.219] | +0.230 [+0.067, +0.402] |
| VAL | UNAMBIGUOUS (n=106) | T1 | 0.561 / 0.392 / 0.642 / 0.537 | 0.955 / 0.862 / 0.967 / 0.967 | +0.081 [-0.002, +0.166] | +0.012 [-0.015, +0.054] |
| VAL | UNAMBIGUOUS (n=106) | T2 | 0.566 / 0.392 / 0.649 / 0.572 | 0.972 / 0.862 / 0.980 / 0.974 | +0.083 [-0.008, +0.169] | +0.008 [-0.002, +0.021] |
| VAL | UNAMBIGUOUS (n=106) | T3 | 0.657 / 0.392 / 0.633 / 0.662 | 0.991 / 0.862 / 0.991 / 0.991 | -0.023 [-0.086, +0.041] | +0.000 [-0.009, +0.009] |
| VAL | UNAMBIGUOUS (n=106) | Full-spec | 0.770 / 0.392 / 0.798 / 0.764 | 1.000 / 0.862 / 1.000 / 1.000 | +0.028 [-0.029, +0.088] | +0.000 [+0.000, +0.000] |
| VAL | UNAMBIGUOUS (n=106) | S-T2+APOE4 | 0.576 / 0.392 / 0.657 / 0.606 | 0.976 / 0.862 / 0.980 / 0.977 | +0.081 [-0.014, +0.168] | +0.003 [-0.006, +0.015] |
| VAL | UNAMBIGUOUS (n=106) | Full-paper (ref.) | 0.770 / 0.392 / 0.798 / 0.764 | 1.000 / 0.862 / 1.000 / 1.000 | +0.028 [-0.029, +0.088] | +0.000 [+0.000, +0.000] |

*macro-F1 over the true classes present. Accuracy per cell in `tables/e1_ambiguity_subsets.csv`. Val subsets with n < 30: point estimates only. CV Δ(AMBIGUOUS) − Δ(UNAMBIGUOUS), macro-F1: T0 -0.096; T1 0.022; T2 0.010; T3 -0.004; Full-spec 0.050; S-T2+APOE4 0.002; Full-paper 0.048.

## Sensitivity: T2 + APOE4
| Table | Values |
|---|---|
| CV macro-F1 | C 0.599, F-KAN 0.620, F-CB 0.606; Δ F-KAN−C +0.021 [-0.014, +0.055]; Δ F-CB−C +0.007 [-0.027, +0.039] |
| CV AD-AUC | C 0.950, F-KAN 0.962; Δ +0.012 [+0.004, +0.020] |
| CV EMCI-vs-LMCI AUC | C 0.667, F-KAN 0.719; Δ +0.052 [+0.005, +0.101] |
| CV CN-vs-MCI AUC | C 0.888, F-KAN 0.897; Δ +0.010 [-0.001, +0.022] |
| Val | macro-F1 C 0.573, F-KAN 0.618, Δ +0.045 [-0.037, +0.128]; AD-AUC C 0.953, F-KAN 0.958, Δ +0.005 [-0.009, +0.021] |
| Masking | Δ F-KAN − F-KAN-noMRI +0.022 [-0.005, +0.051]; − mask-inf +0.033 [+0.006, +0.063] |
| Rules (reported, not in R3) | R1 NOT MET; R2 MET |

## Pre-registered verdicts
| Rule | Tier | Result | Verdict (MET / NOT MET / KAN-specific) |
|---|---|---|---|
| R1 | T0 | CV Δ +0.136 [+0.087, +0.190]; val Δ +0.109 | MET |
| R1 | T1 | CV Δ +0.020 [-0.020, +0.059]; val Δ +0.074 | NOT MET |
| R1 | T2 | CV Δ +0.024 [-0.014, +0.060]; val Δ +0.056 | NOT MET |
| R1 | T3 | CV Δ -0.006 [-0.039, +0.027]; val Δ -0.028 | NOT MET |
| R1 | Full-spec | CV Δ +0.025 [-0.000, +0.050]; val Δ +0.034 | NOT MET |
| R2 | T0 | CV Δ +0.398 [+0.331, +0.467]; val Δ +0.215 | MET |
| R2 | T1 | CV Δ +0.029 [+0.013, +0.047]; val Δ +0.011 | MET |
| R2 | T2 | CV Δ +0.014 [+0.005, +0.023]; val Δ +0.008 | MET |
| R2 | T3 | CV Δ +0.001 [-0.003, +0.006]; val Δ -0.006 | NOT MET |
| R2 | Full-spec | CV Δ -0.000 [-0.004, +0.003]; val Δ -0.001 | NOT MET |
| R3 | T0→Full-spec | Spearman ρ = -0.30 (Δ: +0.136, +0.020, +0.024, -0.006, +0.025); descriptive, 5 points | NOT MET |
| R4 | T0 | M AD-AUC 0.881 − C 0.474 (567 with scan) = +0.407 [+0.339, +0.473] | MET |
| R5 | R1 T0 | F-KAN Δ +0.136, F-CB Δ +0.104 | MET |
| R5 | R2 T0 | F-KAN Δ +0.398, F-CB Δ +0.395 | MET |
| R5 | R2 T1 | F-KAN Δ +0.029, F-CB Δ +0.020 | MET |
| R5 | R2 T2 | F-KAN Δ +0.014, F-CB Δ +0.003 | MET |
| R5 | R2 S-T2+APOE4 | F-KAN Δ +0.012, F-CB Δ +0.002 | MET |

## Seed spread
| Tier | System | min–max per-seed macro-F1 |
|---|---|---|
| T0 | C / F-KAN / F-CB / F-KAN-noMRI | 0.279–0.319 ; 0.423–0.456 ; 0.404–0.430 ; 0.300–0.344 |
| T1 | C / F-KAN / F-CB / F-KAN-noMRI | 0.544–0.571 ; 0.562–0.588 ; 0.583–0.594 ; 0.540–0.557 |
| T2 | C / F-KAN / F-CB / F-KAN-noMRI | 0.574–0.596 ; 0.595–0.624 ; 0.574–0.614 ; 0.566–0.593 |
| T3 | C / F-KAN / F-CB / F-KAN-noMRI | 0.629–0.650 ; 0.611–0.633 ; 0.618–0.635 ; 0.596–0.647 |
| Full-spec | C / F-KAN / F-CB / F-KAN-noMRI | 0.742–0.753 ; 0.751–0.774 ; 0.741–0.759 ; 0.730–0.762 |
| Full-paper (ref.) | C / F-KAN / F-CB / F-KAN-noMRI | 0.733–0.757 ; 0.743–0.779 ; 0.746–0.761 ; 0.736–0.765 |
| all | M | 0.418–0.432 |

## Sanity checks
- [x] No test rows loaded (assertion count: 554 no-test assertions, 0 failures; step1 read line-by-line after D0)
- [x] No excluded column in any T0–T3 design matrix (2317 design-matrix assertions + re-check of every unit's top-K and KAN extras)
- [x] Subject-disjoint folds asserted (1677 checks)
- [x] Frozen artefacts unchanged (hash check before/after: MANIFEST 109 files OK; config_log.csv, kan_runs/, streams.npz unchanged; snapshots phase2_start, phase3_start, phase4_end)
- [x] Reproduction gate passed

## Deviations from protocol
1. D0 (accepted by the user): during the Phase 0 audit, `dataset/processed/step1_labeled.csv` (814 subjects, including the 123 test subjects) was loaded in full into a DataFrame to check missingness. Statistics were computed on train rows only and nothing was saved. Afterwards no DataFrame held test rows (C1 line-by-line extractor → `trainval_extra.parquet`).
2. pyarrow 26.0.0 and pyyaml 6.0.3 were installed into `LYP/.pylib-exp2` and loaded via PYTHONPATH; no environment was modified. The paper's environment split was kept (CatBoost in `.venv-fusion`, KAN in `.venv-kan`).
3. Added after commit 1: for the CN-vs-MCI and EMCI-vs-LMCI scores, a zero denominator gives 0.5 (a tie). Needed only by the Phase 1 depth-3 tree (p_E = p_L = 0), which crashed before any value was computed.
4. The Phase 1 univariate CogIndex row uses the paper's step-4 CogIndex (fitted on all of train, not in-fold); audit only.
5. F-CB sets all MRI inputs, including SEX, to NaN for the 3 subjects without a scan (literal reading of the preregistration).
6. The full MRI streams and every Full-paper system were computed in Phase 2 for the reproduction gate; only the gate table was inspected before Phase 4.
7. The 3-commit policy was intentionally amended by addendum commit `3744036` (E1 analysis + the spec), made after Phase 1 and before Phase 3.
8. Preprocessing (C2 + C3): Full-paper uses the paper's step-4 values (imputation and z-score fitted once on all 569 train subjects); T0–T3, Full-spec and S-T2+APOE4 use raw values with the same rule fitted inside every fit.
9. Commit 3 also contains the Phase 4 code (`evaluate.py`, `report.py`) and `stats.py` changes: a vectorised AUC bootstrap (tested equal to the per-resample `roc_auc_score` loop to 3e-16) and the classes-present macro-F1 for E1. The spec lists results only for commit 3.
10. Template additions: a Full-paper reference row in the main tables; F-KAN-mask-inf and AD-AUC columns in the masking table; the E1 section (as requested); paper-style Δ beside the gate table.

## Anomalies and warnings
- No NaNs, no Python warnings in any Phase 3 log, 0 failed KAN fits out of 2,100 (7 tiers × 300). Best early-stopping steps 17–399; 399 occurs only in Full-paper F-KAN-noMRI, as in the paper's config 34 log (33–399).
- Val: C and F-CB at Full-spec are identical to Full-paper (max |diff| 0). Both are fitted on all 569, where the in-fold rule equals the paper's step-4 fit. F-KAN val probabilities differ (max 0.14; their training scores are the differing CV streams), but no seed-averaged val class prediction changes, so val macro-F1 is identical and the AUCs differ slightly.
- T0 (demographics only): C CV AD-vs-rest AUC 0.480 (below 0.5); val 0.608.
- Feature selection: T1 KAN extras always include FAQ and EcogSPMem (30/30 units); Full-spec extras are CDRSB + CogIndex + MMSE in 14/30 units and ADAS13 replaces MMSE in 12/30. No missingness indicator was created in any fit.
- E1: val AMBIGUOUS n = 16 (3 AD) → point estimates only; val UNMATCHED n = 0; train UNMATCHED n = 8. The 8 train UNMATCHED subjects are excluded from both E1 subsets.
- Assertion counts are cumulative over Phases 1–4, including the dry runs and the gate run.

## Observations (max 5 bullets, factual, no claims beyond the tables)
- R1 is MET only at T0 (CV Δ macro-F1 F-KAN − C +0.136 [+0.087, +0.190]; val +0.109). At T1, T2, T3 and Full-spec the CV CIs include zero (Δ +0.020, +0.024, −0.006, +0.025).
- R2 (AD-vs-rest AUC) is MET at T0, T1 and T2 and NOT MET at T3 and Full-spec. The CV Δ is +0.398, +0.029, +0.014, +0.001 and −0.000 from T0 to Full-spec. In every tier where R1 or R2 is met, F-CB − C has the same sign (R5).
- R3 is NOT MET: Spearman ρ = −0.30; the Δ macro-F1 sequence is not monotone (T3 −0.006, Full-spec +0.025).
- EMCI-vs-LMCI AUC (H4, exploratory): the CV Δ F-KAN − C CI excludes zero at T1 (+0.074) and T2 (+0.055) and includes zero at T0, T3 and Full-spec.
- E1 (CV): the Δ AD-AUC F-KAN − C is larger in AMBIGUOUS (n = 92) than in UNAMBIGUOUS (n = 469) at every tier except T0 (e.g. T3 +0.032 vs −0.001; Full-spec +0.023 vs −0.000). The Δ macro-F1 is larger in AMBIGUOUS at T1, T2, Full-spec, S-T2+APOE4 and Full-paper and smaller at T0 and T3, with all AMBIGUOUS macro-F1 CIs including zero.

## Files
- `experiments/exp2_reduced_clinical/`: `PREREGISTRATION.md`, `PREREGISTRATION_ADDENDUM.md`, `manifest.yaml`, `results.json`, `code/`
- `tables/`: `metrics_all.csv`, `deltas_all.csv`, `e1_ambiguity_subsets.csv`, `reproduction_gate.csv`, `confusion_cv_<tier>_<system>.csv`, `phase1_*.csv`
- `figures/`: `delta_by_tier.png`, `confusion_T1.png`, `confusion_T2.png`, `confusion_Full-spec.png`
- Local only (gitignored): `oof/` (streams_<tier>.npz, kan_<tier>.npz, *_meta.json), `logs/`, `trainval_extra.parquet`
