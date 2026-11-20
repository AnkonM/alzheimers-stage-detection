# EXP2 — Reduced clinical inputs: preregistration

Branch `exp/reduced-clinical` (from `9bac529`). Written 2026-10-09, before any
Phase 1, 2 or 3 result exists. Spec: `docs/exp2_reduced_clinical.md`.
The frozen fusion-v3 system (`analysis/fusion_v3/frozen/`, `freeze.md`) is
read-only; the test split is not used.

Results already known when this was written (Phase 0 audit, read from saved
paper outputs): paper CV macro-F1 recomputed from `streams.npz` /
`kan_runs/32.npz` = clinical 0.7435, KAN 0.7562 (mean of per-seed values);
0.7415 / 0.7627 (macro-F1 of seed-averaged probabilities). No tier result
exists.

## 1. Decision rules (spec Section 3, verbatim)

| Rule | Definition |
|---|---|
| **R1 – MRI adds value at tier T** | Paired Δ macro-F1 (Fusion-KAN − Clin-only) on seed-averaged CV OOF predictions has a 95% bootstrap CI lower bound > 0 **and** the validation-holdout Δ point estimate is > 0. |
| **R2 – MRI adds AD-detection value at tier T** | Same as R1, using AD-vs-rest AUC. |
| **R3 – Trend (H2)** | Spearman correlation between tier order (T0=0 … Full=4) and Δ macro-F1 is ≤ −0.8. With 5 points this is descriptive, so report it as such. |
| **R4 – MRI beats demographics (H3)** | At T0, MRI-only AD-vs-rest AUC minus Clin-only AD-vs-rest AUC has a CV CI lower bound > 0. |
| **R5 – Robustness** | A tier counts as positive in the summary only if Fusion-CatBoost (Section 6) shows a Δ with the same sign. Otherwise label it "KAN-specific". |
| **Reproduction gate** | The Full-tier Clin-only and Fusion-KAN CV macro-F1 must reproduce the paper's values (0.744 and 0.756) within ±0.010. If not, **stop**; do not run the other tiers. |

## 2. Phase 0 decisions (from the user, verbatim)

D0. The full load of step1_labeled.csv during the audit is accepted. Record it under "Deviations" in SHARE_THIS.md exactly as you described it. From now on, no DataFrame may hold test rows.

C1. Approved. Build the line-by-line extractor that keeps only train/val RIDs, asserts no test RID is ever written, and writes trainval_extra.parquet under experiments/exp2_reduced_clinical/. All later code reads only from that parquet and fusion_trainval.csv.

C2 + C3. Run Full twice:
  - Full-paper: the paper's step-4 table, paper preprocessing, fixed KAN extras (CDRSB, CogIndex, MMSE). Used ONLY for the reproduction gate and to trace the paper's numbers.
  - Full-spec: raw values, in-fold imputation (same imputation rule as the paper, fitted per fold), KAN extras chosen by the spec rule (AGE + in-fold SHAP top-3 non-age). This is the Full row used in R3 and in all tier-trend tables.
  T0–T3 use the Full-spec pipeline. Record the preprocessing difference as a deviation.

C4. Primary: the paper's masking (retrain the KAN with MRI masked in train and evaluation), reported as "F-KAN-noMRI". Also report inference-only masking of the normal F-KAN as "F-KAN-mask-inf", since it needs no retraining. Both go in the masking table.

C5. Approved. Primary metric is Spec §7 (seed-averaged OOF probabilities), and R1–R5 use it. Also report the paper-style figure (mean of per-seed macro-F1) for C, F-KAN and Δ at Full-paper and Full-spec, labelled as such.

C6. F-CB = tier top-K ∪ {AGE} + MRI specialist inputs (6 volume ratios, ALZ, BAG, SEX). AGE is added explicitly because the top-K at T1–T3 may drop it. Keep BAG missing where its reliability rule fails, and let CatBoost handle the NaN natively.

C7. Approved. T3 = T2 + ADAS11, ADAS13, Q4SCORE, RAVLT_{immediate, learning, forgetting, perc_forgetting}, TRABSCOR. DIGITSCR is dropped. List the redundancy between Q4 and ADAS13 in the report caveats.

C8. Approved. Import pure functions only, write only under experiments/exp2_reduced_clinical/. Additionally, hash analysis/fusion_v3/config_log.csv, kan_runs/ and streams.npz before Phase 3 and after Phase 4, and add the comparison to the sanity checks.

C9. Acknowledged. The validation holdout is fit on 569 and evaluated on 122, the same as the paper's kan_runs val predictions. Use the real counts (LMCI = 25) everywhere.

Phase 1 addition (exploratory, not part of any rule). Besides the depth-3 tree and logistic model, apply the ADNI-GO/2 enrolment cut-offs directly as a rule-based classifier on train+val, and report its agreement with the label (accuracy, macro-F1, 4×4 confusion matrix). The cut-offs are:
  CN = MMSE 24–30, CDR global 0, LM delayed ≥9 (16+ yrs education) / ≥5 (8–15) / ≥3 (0–7)
  EMCI = MMSE 24–30, CDR 0.5, LM delayed 9–11 / 5–9 / 3–6
  LMCI = MMSE 24–30, CDR 0.5, LM delayed ≤8 / ≤4 / ≤2
  AD = MMSE 20–26, CDR 0.5 or 1
Treat these cut-offs as unverified. Check them against the ADNI2 procedures manual if you can access it, and say in the report whether you did. Our values are baseline-visit scores, while the cut-offs applied at screening, so some disagreement is expected; say so.

## 3. Implementation details (fixed before any result; 2026-10-09)

These resolve choices the spec leaves open. They do not change Sections 1–2.

**Data.** `trainval_extra.parquet` (C1) holds the raw step-1 values for the
691 train+val subjects. Source → name mapping as `src/step2_feature_selection.py`
`FEATURE_MAP`, plus `Q4SCORE`, `TRABSCOR`, `CDGLOBAL__CDR → CDGLOBAL`,
`CDMEMORY`, `LDELTOTL`, `LIMMTOTL`, `MPACCTRAILSB`. MRI inputs (ratios, ALZ,
has_mri) come from `fusion_trainval.csv`; brain age from
`analysis/brain_age/brainage_trainval.csv` and
`analysis/kan_fusion/brainage_reliability.csv` (both train/val only; asserted).

**Tiers (exact columns).**
- T0: AGE, PTGENDER, PTEDUCAT (3).
- T1: T0 + FAQ + EcogPt{Mem,Lang,Visspat,Plan,Organ,Divatt,Total} + EcogSP{same} (18).
- T2: T1 + MOCA (19).
- T3: T2 + ADAS11, ADAS13, Q4SCORE, RAVLT_immediate, RAVLT_learning, RAVLT_forgetting, RAVLT_perc_forgetting, TRABSCOR (27).
- Full-spec: the paper's 29 clinical features (raw values; CogIndex built in-fold) (29).
- Full-paper: the paper's 29 features from `fusion_trainval.csv` (step-4 processed).
- Sensitivity S: T2 + APOE4 (20).
- Excluded from T0–T3 and S (asserted on every design matrix: C, F-CB, KAN extras): MMSE, CDRSB, CDGLOBAL, CDMEMORY, CogIndex, LDELTOTL, LIMMTOTL, MPACCTRAILSB, the 17 blacklist columns, FreeSurfer raw volumes.

**Preprocessing (Full-spec pipeline, every fit, on that fit's rows only).**
The paper's step-4 rule: median imputation (numeric), mode (PTGENDER),
missingness indicator if > 10% missing on the fit rows, z-score with the fit
rows' mean and SD (ddof = 1); Full-spec only: CogIndex = z(MMSE) − z(CDRSB) − z(ADAS13).

**Folds and nesting** (as `analysis/fusion_v3/step2_streams.py`). Outer:
StratifiedKFold(5, shuffle, random_state = s) over the 569 train subjects in
`fusion_trainval.csv` order, s = 42–46. Inner: the same splitter on each
outer-train set. CV: specialists fitted on outer-train score the outer-test
fold; the KAN trains on inner OOF stream scores. Val: specialists fitted on
the 569 score the 122; the KAN trains on the 569 outer OOF scores.

**C (clinical).** Frozen CatBoost (MultiClass, depth 6, l2 10, lr 0.1, 265
trees, PTGENDER categorical, random_seed = s). Ranking: CatBoost on all tier
columns of the fit rows → mean |SHAP| over those rows → stable descending
order → top K = min(12, n) → refit. T0: all 3, no selection.

**M (MRI).** `specialists.MRI("logreg", s, C=0.1)` unchanged, computed once.
It is recomputed with the folds above and must match `streams.npz` to
< 1e-8; if it does not, the mismatch is reported and the recomputed scores are used.

**F-KAN.** Config 32 (K2 residual, width [n_in, 5], G 5, order 3, mu 1e-3,
class-balanced CE, ord 0.5, scale, no temperature, dropout 15%, KAN seeds
42–46 averaged), using `kan_fusion.fit_predict`. Inputs: 4 centred clin
log-probs + 4 centred MRI log-probs + has_mri + extras.
- Extras, T0: AGE, SEX (0/1), PTEDUCAT (n_in = 12).
- Extras, other spec tiers: AGE + the first 3 non-AGE columns in the SHAP
  ranking of the clinical specialist fitted on the KAN's training rows
  (outer-train for CV, the 569 for val) (n_in = 13). PTGENDER is encoded as 0/1.
- Extras are preprocessed with the rule above, fitted on the KAN's training rows.
- Full-paper: extras AGE, CDRSB, CogIndex, MMSE from `fusion_trainval.csv` (n_in = 13).

**F-KAN-noMRI.** F-KAN with `no_mri = True` (retrained). **F-KAN-mask-inf.**
The F-KAN models applied with has_mri = 0 (neutral MRI logits, flag 0) on the
evaluation rows.

**F-CB.** Frozen CatBoost on C's top-K of the same fit rows ∪ {AGE}
(preprocessed) + 6 ratios, ALZ, BAG, SEX (not imputed). BAG uses the
in-fold (α, β) and is NaN when the reliability rule fails or there is no brain
age. All MRI inputs are NaN for subjects without a scan. Not stacked.

**Metrics.** Classes from argmax.
- AD AUC uses p(AD).
- CN-vs-MCI uses rows with y ∈ {CN, EMCI, LMCI} and score (p_E + p_L)/(p_C + p_E + p_L).
- EMCI-vs-LMCI uses rows with y ∈ {EMCI, LMCI} and score p_L/(p_E + p_L).
- Stage-distance error = mean |pred − y|.
- Log-loss clips at 1e-12.
- M and any comparison with M use subjects with a scan (567 CV / 121 val).

**Statistics.**
- Primary: metrics of the seed-averaged probabilities.
- Bootstrap: `np.random.default_rng(0)`, 2,000 subject resamples drawn once per evaluation set (CV-569, CV-567, val-122, val-121) and shared by every comparison. The CI is the 2.5–97.5th percentile of Δ. Resamples where an AUC is undefined are dropped and counted.
- Paper-style figure (C5): mean of per-seed macro-F1, and its Δ CI as `analysis/fusion_v3/evaluate.py` computes it.

**Rule application.**
- R1/R2 apply to T0, T1, T2, T3 and Full-spec. Sensitivity S is reported but not in R3.
- R3 uses the CV Δ macro-F1 (F-KAN − C) of T0, T1, T2, T3 and Full-spec against 0–4.
- R4 is evaluated on the 567 CV subjects with a scan.
- R5: for each tier where R1 (or R2) is MET, compare the CV Δ point estimate of F-CB − C against F-KAN − C. The same sign makes the tier positive; otherwise it is labelled "KAN-specific".
- Reproduction gate: evaluated on Full-paper. It passes only if C and F-KAN are each within ±0.010 of 0.744 / 0.756 under both definitions (spec §7 and paper-style), both from the saved paper OOF and from this experiment's code.

**Phase 1 (exploratory).**
- Univariate: for each candidate column, a 1-D logistic regression per one-vs-rest class, fitted in-fold (median imputation, z-score) on the CV above. The AUC is computed on the seed-averaged OOF probabilities. Spearman ρ with stage is computed on raw train values (pairwise complete).
- Reconstructability: inputs are {MMSE, CDGLOBAL, LDELTOTL, PTEDUCAT}.
  - Models: DecisionTreeClassifier(max_depth = 3, random_state = s), and multinomial LogisticRegression(C = 1, max_iter = 5000) on in-fold z-scored inputs.
  - Evaluation: same CV, seed-averaged OOF; macro-F1 and confusion matrix.
- Rule-based classifier: applied to the 691 train+val baseline values.
  - Education bands: 0–7, 8–15, ≥ 16 years.
  - A subject matching exactly one class is assigned to it. One matching several classes is "ambiguous"; one matching none is "unmatched".
  - Reported: the 4×4 confusion matrix of uniquely matched subjects, ambiguous and unmatched counts per true class, and accuracy and macro-F1 both over all 691 (ambiguous and unmatched count as errors) and over the uniquely matched.

**Safety.**
- Every loader asserts that no RID is in `test_rids.csv` and counts the assertions.
- `frozen/MANIFEST.sha256` is verified before and after.
- sha256 of `analysis/fusion_v3/config_log.csv`, `kan_runs/*` and `streams.npz` is recorded before Phase 3 and after Phase 4.
