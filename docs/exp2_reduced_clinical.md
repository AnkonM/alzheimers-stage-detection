# EXP2 — What does MRI add when the label-defining instruments are removed?

---

## 1. Background

ADNI-GO/2 diagnostic labels (CN / EMCI / LMCI / AD) are assigned from cognitive and functional assessments: MMSE range, CDR global, and Logical Memory II delayed recall (education-adjusted cut-offs separate EMCI from LMCI), plus NINCDS-ADRDA criteria for AD. The frozen system's clinical stream uses CDR-SB, MMSE and CogIndex (built from MMSE, CDR and ADAS) among its top features. The model therefore partly re-derives the labelling rule. This is the most plausible reason MRI adds no measurable staging accuracy in the paper (fusion − clinical: +0.013 CV macro-F1, CI includes zero).

**Prior evidence to be aware of (do not assume it holds; re-verify):** an earlier feasibility pass on branch `fusion-feasibility` found that adding the cross-fitted ALZ score *alone* to CatBoost with "diagnosis-defining scores removed" gave +0.009 macro-F1 (CI crossing zero). That test used ALZ only. This experiment uses the **full MRI specialist** (regional volumes / ICV, ALZ, brain-age gap, age, sex), which carries much more AD signal (CV AD-vs-rest AUC 0.880 vs 0.788 for ALZ + brain-age only). Locate that earlier code if it exists and record which variables it removed, but do not reuse its results.

## 2. Research question and hypotheses

**RQ:** Does structural MRI add staging and AD-detection information when the clinical stream is restricted to inputs that do **not** define the labels, and how does that contribution change as richer clinical inputs are added?

- **H1 (primary):** At the reduced tiers T1 and T2 (Section 5), fusion improves CV macro-F1 over the clinical-only model at the same tier.
- **H2:** The fusion − clinical gain shrinks as the clinical tier gets richer (T0 → T1 → T2 → T3 → Full), approaching zero at Full.
- **H3:** At T0 (demographics only), the MRI specialist outperforms the clinical-only model on AD-vs-rest AUC.
- **H4 (exploratory):** MRI improves EMCI-vs-LMCI discrimination once Logical Memory is excluded.

## 3. Pre-registered decision rules

Write these into `experiments/exp2_reduced_clinical/PREREGISTRATION.md` and **commit them before any Phase 3 run**. Record the commit hash in `SHARE_THIS.md`. Do not change them after seeing results. If a rule turns out to be unworkable, keep it, report the rule as written, and list the problem under "Deviations".

| Rule | Definition |
|---|---|
| **R1 – MRI adds value at tier T** | Paired Δ macro-F1 (Fusion-KAN − Clin-only) on seed-averaged CV OOF predictions has a 95% bootstrap CI lower bound > 0 **and** the validation-holdout Δ point estimate is > 0. |
| **R2 – MRI adds AD-detection value at tier T** | Same as R1, using AD-vs-rest AUC. |
| **R3 – Trend (H2)** | Spearman correlation between tier order (T0=0 … Full=4) and Δ macro-F1 is ≤ −0.8. With 5 points this is descriptive, so report it as such. |
| **R4 – MRI beats demographics (H3)** | At T0, MRI-only AD-vs-rest AUC minus Clin-only AD-vs-rest AUC has a CV CI lower bound > 0. |
| **R5 – Robustness** | A tier counts as positive in the summary only if Fusion-CatBoost (Section 6) shows a Δ with the same sign. Otherwise label it "KAN-specific". |
| **Reproduction gate** | The Full-tier Clin-only and Fusion-KAN CV macro-F1 must reproduce the paper's values (0.744 and 0.756) within ±0.010. If not, **stop**; do not run the other tiers. |

## 4. Non-negotiables

1. **The test split is not touched.** No predictions, metrics or loading of test rows. If any code path would load test subjects, add an assertion that aborts. The frozen system's test read already happened; this experiment lives entirely on train (569) and validation (122).
2. **Frozen artefacts are read-only.** Do not modify, re-save or overwrite anything under the frozen-system directory. Copy what you need.
3. **Same splits as the paper.** Use the clinical master split. Primary analysis is repeated stratified 5-fold CV on the 569 training subjects with seeds 42–46 (the paper's protocol). Secondary is fitting on all 569 train subjects and evaluating on the 122 validation subjects.
4. **Everything is fitted in-fold:** imputers, scalers, SHAP-based top-K selection and KAN input selection. Assert subject disjointness at every fold.
5. **No retuning per tier.** Reuse the frozen hyperparameters exactly (CatBoost depth 6, l2 10, lr 0.1, 265 trees; MRI logistic C = 0.1; KAN width [n_in, 5], grid 5, order 3, same losses, same modality dropout 15%, same 5 × 5 ensemble). The only allowed adaptation is input width (Section 6).
6. **MRI stream is unchanged.** Reuse the existing cross-fitted ALZ scores, volumes / ICV, brain-age gap, age and sex, exactly as the paper did. Do not retrain the CNN. MRI-only results are tier-independent: compute them once.
7. **Staged execution with stop gates:** Phase 0 audit → STOP → Phase 1–2 manifest + dry run → STOP → Phase 3 full run → Phase 4 report. At each STOP, print a short report and wait for approval.
8. **Never assume earlier changes exist.** Verify file paths, column names and artefacts on disk before using them.

## 5. Clinical input tiers

Exact column names must come from the Phase 0 audit. The list below is the *intended* content; map each item to real columns in the manifest.

**Always excluded from T0–T3 (label-defining or derived from label-defining instruments):**
- MMSE total and items, including any `_bl` variants
- CDR global, CDR-SB, CDR boxes
- Logical Memory immediate and delayed (e.g. `LIMMTOTAL`, `LDELTOTAL`), including `_bl` variants
- CogIndex and any composite containing MMSE, CDR or Logical Memory (e.g. mPACC variants)
- The 17 blacklisted proxies (already excluded)
- FreeSurfer volumes (they belong to the MRI stream)

| Tier | Contents (cumulative) | Rationale |
|---|---|---|
| **T0** Demographics | age, sex, education | No cognitive information at all |
| **T1** + Function / self-report | + FAQ total, ECog (participant and study-partner, the subscales/totals used in the current pipeline) | Questionnaires; no clinician-administered testing |
| **T2** + Brief screen | + MoCA total | A 10-minute screening test, common in primary care |
| **T3** + Non-defining neuropsych | + RAVLT measures, ADAS11 / ADAS13 / ADAS-Q4, Trails B / digit scores if present | Full neuropsychology minus the label-defining instruments |
| **Full** (reference) | The paper's frozen 12-feature set | Reproduction gate |

**Sensitivity tier (report separately, not part of the trend):** T2 + APOE4.

For each tier, top-K = min(12, number of tier features), chosen in-fold by mean |SHAP| exactly as in the paper. T0 uses all 3 features with no selection.

**Caveat to record in the report:** FAQ and ECog-study-partner partly overlap conceptually with CDR's functional boxes. They are not used in the ADNI label rule, so they stay in, but say this in the report.

## 6. Systems run per tier

| ID | System | Notes |
|---|---|---|
| **C** | Clin-only CatBoost | Frozen config on the tier's features |
| **M** | MRI-only specialist | Tier-independent, computed once; metrics on subjects with a scan (567 train / 121 val) |
| **F-KAN** | Residual KAN fusion | Inputs: 4 centred clin log-probs (from C at this tier, out-of-fold) + 4 centred MRI log-probs + scan flag + age + the top-3 non-age tier features selected in-fold by SHAP. This replaces CDR-SB / CogIndex / MMSE, which the frozen KAN used as direct inputs and which are **not allowed** in T0–T3. At T0 use sex and education (width shrinks to 12). Stream inputs must be out-of-fold specialist scores, as in the paper. |
| **F-CB** | CatBoost on concatenated inputs | Tier features + MRI specialist inputs; frozen config. Robustness check for R5. |
| **F-KAN-mask** | F-KAN with MRI inputs masked at inference | Matches the paper's masking analysis |

Expected run count: 5 tiers (+1 sensitivity) × {C, F-KAN, F-CB, F-KAN-mask} × 5 seeds × 5 folds, plus M once. The KAN ensemble dominates the cost. Time it in the dry run.

## 7. Metrics and statistics

**Four-stage:** macro-F1 (primary), balanced accuracy, accuracy, per-class F1, mean stage-distance error, multiclass log-loss.
**Binary views:** AD-vs-rest AUC; CN-vs-MCI AUC (EMCI + LMCI vs CN, AD excluded); EMCI-vs-LMCI AUC computed on true-MCI subjects only, from p(LMCI) / (p(EMCI) + p(LMCI)).

**Statistics:** for each system, average OOF probabilities across the 5 seeds per subject, then compute metrics. Paired Δ between systems uses a subject-level paired bootstrap (2,000 resamples, 95% percentile CI, fixed bootstrap seed 0). Also report the seed-level spread (min–max of per-seed macro-F1) so seed variance is visible. On validation, report point estimates and bootstrap CIs (expect wide intervals; n = 122, LMCI ≈ 24).

## 8. Phases

### Phase 0 — Codebase audit (read-only) → STOP
Report:
- repo root, branch, last commit, and whether the working tree is clean;
- locations of: the clinical feature table, the master split file, the clinical pipeline (CatBoost + SHAP selection), the MRI specialist and its inputs, the cross-fitted ALZ scores, the brain-age gap file, the KAN fusion code, the frozen artefact directory and its freeze guard, and the earlier `fusion-feasibility` ablation (if present);
- the **full list of columns** available in the clinical feature table and in the 36-feature set, each tagged *label-defining / blacklisted / MRI / eligible*;
- **whether Logical Memory variables (`LDELTOTAL`, `LIMMTOTAL` or equivalents) are present anywhere in the current feature sets** (the paper reviewer asked this directly);
- whether the paper's CV numbers can be reproduced from saved OOF predictions without re-running.

### Phase 1 — Label-definition audit
1. For every candidate clinical column, compute the univariate CV AUC for each one-vs-rest stage and the Spearman correlation with stage (train only).
2. **Label reconstructability:** fit a depth-3 decision tree and a multinomial logistic regression on only {MMSE, CDR global, Logical Memory delayed recall, education} (whichever exist) to predict the 4-class label, using the same CV. Report macro-F1 and the confusion matrix. A high value supports the circularity argument in the paper. If Logical Memory is not available in the data, report that and run with the remaining variables.
3. Report how many training subjects have every variable needed to reconstruct the label rule.

### Phase 2 — Manifest + dry run → STOP
- Write `manifest.yaml`: the exact column list per tier, n features, per-column missingness (train), subjects per tier after missingness handling (impute as in the paper; do not drop subjects), and the KAN input width per tier.
- Commit `PREREGISTRATION.md` (Section 3, verbatim) and record the hash.
- Dry run: one tier (T1), one seed, one fold, all systems. Check the shapes, assertions (subject disjointness, no test rows, no excluded column in any tier's design matrix), and that outputs land in the right place.
- Report runtime per system and a projected total. If the projected total exceeds 12 hours, propose (do not apply) a reduced KAN ensemble for screening, and wait.
- Run the **reproduction gate** at Full (this may use saved OOF predictions if Phase 0 found them valid).

### Phase 3 — Full run
Run every tier and system. Save per-subject OOF probabilities (`oof_{tier}_{system}_seed{s}.parquet`), per-fold logs, and validation-holdout predictions. Log runtime and any warnings.

### Phase 4 — Analysis and report
Compute every metric and Δ, apply R1–R5 mechanically, write the outputs, then write `SHARE_THIS.md`.

## 9. Output spec

```
experiments/exp2_reduced_clinical/
├── PREREGISTRATION.md          # committed before Phase 3
├── manifest.yaml
├── SHARE_THIS.md               # the only file the user pastes back
├── results.json                # every number in SHARE_THIS.md, machine-readable
├── tables/                     # CSVs behind each table
├── figures/                    # delta_by_tier.png, confusion matrices T1/T2/Full
├── oof/                        # per-subject predictions
└── logs/
```

### `SHARE_THIS.md` template (follow exactly; keep under ~200 lines; no interpretation beyond the "Observations" block)

```markdown
# EXP2 — Reduced clinical inputs: results

## Run metadata
Branch / commit (code) / commit (preregistration) / date / total runtime / seeds / bootstrap resamples / hardware

## Phase 0–1 audit
- Logical Memory present in frozen feature set: YES/NO (columns: …)
- Label-defining variables in frozen 12-feature set: …
- Label reconstructability (tree / logistic on {…}): macro-F1 … ; confusion matrix (4×4)
- Earlier fusion-feasibility ablation: found at … / not found; variables it removed: …

## Tier definitions
| Tier | n features | Columns | Max missingness (train) |

## Reproduction gate
| System | Paper CV macro-F1 | Reproduced | Δ | PASS/FAIL |

## Main results — CV (569 train, seed-averaged OOF)
| Tier | C macro-F1 | M macro-F1 | F-KAN macro-F1 | F-CB macro-F1 | Δ F-KAN−C [95% CI] | Δ F-CB−C [95% CI] |
| Tier | C AD-AUC | M AD-AUC | F-KAN AD-AUC | Δ F-KAN−C [95% CI] |
| Tier | C EMCI-vs-LMCI AUC | M | F-KAN | Δ [95% CI] |
| Tier | C CN-vs-MCI AUC | M | F-KAN | Δ [95% CI] |

## Validation holdout (fit on 569, evaluate on 122)
| Tier | C macro-F1 | F-KAN macro-F1 | Δ [95% CI] | C AD-AUC | F-KAN AD-AUC | Δ [95% CI] |

## Per-class F1 (CV) at T1, T2, Full
| Tier | System | CN | EMCI | LMCI | AD |

## MRI masking inside fusion (CV)
| Tier | Δ macro-F1 (F-KAN − F-KAN-mask) [95% CI] |

## Sensitivity: T2 + APOE4
(one row per main table)

## Pre-registered verdicts
| Rule | Tier | Result | Verdict (MET / NOT MET / KAN-specific) |

## Seed spread
| Tier | System | min–max per-seed macro-F1 |

## Sanity checks
- [ ] No test rows loaded (assertion count: …)
- [ ] No excluded column in any T0–T3 design matrix
- [ ] Subject-disjoint folds asserted (n checks)
- [ ] Frozen artefacts unchanged (hash check before/after)
- [ ] Reproduction gate passed

## Deviations from protocol
(none / list each with reason)

## Anomalies and warnings
(convergence warnings, NaNs, unexpected class counts …)

## Observations (max 5 bullets, factual, no claims beyond the tables)

## Files
(paths to tables, figures, OOF)
```

## 10. Out of scope
Conversion-based validation of the severity index, new MRI models, CNN retraining, hyperparameter search, any use of the test split, and changes to the frozen system or the paper.

## 11. Commit policy

All work happens on `exp/reduced-clinical`. Never commit to `master` or any other branch, and never push. The user pushes after reviewing `SHARE_THIS.md`.

| Commit | When | Contents | Message |
|---|---|---|---|
| 1 | Start of Phase 2, before any Phase 3 run | `PREREGISTRATION.md` only | `exp2: preregistration` |
| 2 | End of Phase 2, after the dry run passes | Experiment code, `manifest.yaml`, `.gitignore` additions | `exp2: code and manifest` |
| 3 | End of Phase 4 | `SHARE_THIS.md`, `results.json`, `tables/`, `figures/` | `exp2: results` |

- Make each commit only after its gate passes. If a phase fails or stops, leave the work uncommitted and report it.
- Before commit 2, add `experiments/exp2_reduced_clinical/oof/` and `experiments/exp2_reduced_clinical/logs/` to `.gitignore`, along with any file over 10 MB.
- Commit 1 must contain nothing but the preregistration, so its hash proves the rules came before the results.
- Do not amend, squash or rebase these commits.
- Record all three hashes in the Run metadata block of `SHARE_THIS.md`. Commit 3's own hash can't be inside itself, so report it in the final message instead.
