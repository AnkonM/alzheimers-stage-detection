# Reviewer response: pre-registration (Part B analyses)

Branch `paper/reviewer-response`, created from the fusion-v3 freeze commit `8f351b6`. Date 2026-11-18.
This file is committed **before** any Part B analysis script is written or run. The rules below may
not be changed after results exist. If a rule proves unworkable, it stays as written and the problem
goes under "Deviations" in results.md.

## 0. General rules

- **Test split:**
  - No model is fitted, selected or tuned on test, and no new inference is run on test.
  - Every script except `b4_operating.py` aborts if any RID from `dataset/processed/test_rids.csv` appears in a loaded table.
  - `b4_operating.py` reads only the already-stored frozen test predictions (`adc4ccb:analysis/fusion_v3/frozen_run/test_kan.npz`) and the labels in `adc4ccb:analysis/fusion_v3/frozen_run/test_streams.npz`.
- **Frozen artefacts:**
  - Files in `analysis/fusion_v3/frozen/` and the existing result files are read-only.
  - `sha256sum -c analysis/fusion_v3/frozen/MANIFEST.sha256` (manifest sha256 `5e266705…cde92`) is run before and after.
- **CV protocol (the paper's):**
  - 569 clinical-train subjects; outer seeds s = 42–46; StratifiedKFold(5, shuffle=True, random_state=s).
  - Outer and inner fold indices are taken from `analysis/fusion_v3/streams.npz`, or regenerated identically and asserted equal.
- **Point estimate (Table II convention):** a metric is computed per outer seed on that seed's out-of-fold predictions, then averaged over the 5 seeds. EXP2 keeps its own §7 convention (metrics of seed-averaged probabilities).
- **CIs:**
  - Paired subject-level bootstrap: 2,000 resamples, `np.random.default_rng(0)`, the same resample indices for every comparison on the same subjects.
  - The statistic in each resample is the per-seed metric averaged over seeds (as `analysis/fusion_v3/evaluate.py` with `metrics.macro_f1_boot`).
  - 95% percentile interval.
- **AD-vs-rest AUC** uses p(AD) from the 4-class (stage) output, for every system.
- **Subject IDs** never appear in any output file. Subject-level intermediates go in the gitignored `analysis/reviewer_response/private/`.

## B1 Fusion baselines (train CV only; val and test unread)

- **Inputs:** exactly the 13 inputs the residual KAN receives (config 32).
  - For outer seed s and fold f, the stacker trains on the inner out-of-fold stream scores `clin_inner_{s}_{f}`, `mri_inner_{s}_{f}` of the outer-train rows. It predicts the outer-test fold from `clin_oof[s]`, `mri_oof[s]` (`streams.npz`).
  - Extras: AGE, CDRSB, CogIndex, MMSE from `analysis/kan_fusion/fusion_trainval.csv`.
  - Input construction is `kan_fusion.Inputs` (imported, unchanged) with config 32 settings: centred log-probabilities, neutral MRI logits plus flag 0 for missing scans, z-scored on the training rows, no temperature.
- **(i) Logistic-regression stacker:**
  - sklearn `LogisticRegression` (multinomial, L2, C = 1.0, lbfgs, max_iter 5000, class_weight = "balanced"), deterministic.
  - No tuning. C = 1.0 is fixed here.
- **(ii) MLP stacker:**
  - torch, one hidden layer of h ReLU units, 5 outputs (4 stage logits + 1 ADD logit; the stage output is used).
  - Loss: class-balanced CE (weights n / (4·n_k) from the fitting rows) + 0.5 × BCE(ADD), as in the KAN minus the KAN-specific terms.
  - Training: Adam lr 0.01, full batch, ≤ 400 steps, early stopping on a stratified 15% holdout of the training rows (patience 40, best state restored).
  - Ensemble: seeds 42–46 per outer unit, probabilities averaged, as the KAN.
- **MLP hidden size:**
  - h ∈ {4, 8, 16, 32}, chosen **separately for each outer unit**.
  - Selection uses inner StratifiedKFold(5, shuffle=True, random_state=s) over that unit's outer-train rows (training data: the same inner OOF stream scores, with Inputs refitted on each inner-train part).
  - Criterion: mean inner macro-F1 of the 5-seed ensemble. A tie goes to the smaller h.
  - The outer-test fold is never used for selection.
- **KAN reference:**
  - The stored config 32 predictions `analysis/fusion_v3/kan_runs/32.npz` (no refit).
  - Check: their CV macro-F1 must equal 0.7562 (± 5e-5, `config_log.csv` row 32).
- **Comparison rule** (Δ = stacker − KAN, CI from §0):

  | Metric | Margin m |
  |---|---|
  | macro-F1 | 0.02 |
  | AD-vs-rest AUC | 0.01 |

  - "stacker non-inferior to KAN" if CI lower bound > −m;
  - "stacker better" if CI lower bound > 0;
  - "KAN better" if CI upper bound < 0;
  - "stacker inferior beyond margin" if CI upper bound < −m;
  - otherwise "no difference resolved".
  - Every label that applies is reported.

## B2 Fold-wise preprocessing

- **Raw values:**
  - A line-by-line extractor of `dataset/processed/step2_features.csv` keeps the 569 train + 122 val rows only.
  - It parses nothing from a test row beyond its RID and asserts the absence of test RIDs.
- **Gate B2-a (code check, run before any fold-wise fit):**
  - The step-4 rule fitted on all 569 train must reproduce `fusion_trainval.csv` (the 29 clinical features incl. CogIndex, the 6 ratios) and `step4_{train,val}.csv` (36 features) within max |diff| 1e-9.
  - The step-4 rule here is `src/step4_processing.py` functions: median/mode imputation, > 10% missingness indicators, z-score, CogIndex from the z-scored MMSE, CDRSB, ADAS13; ratios: volume / ICV, train-median imputation, z-score.
  - If B2-a fails, B2 stops and the failure is reported.
- **Fold-wise rule:**
  - The same transforms are fitted on exactly the rows each model is fitted on:
    - outer-train rows for outer models;
    - inner-train rows for inner (stream-score) models;
    - all 569 for the val fits;
    - for the KAN extras, the KAN's training rows.
  - Nothing else changes: hyperparameters, per-fit SHAP top-12, the CatBoost F/R procedure (`analysis/kan_fusion/catboost_stage.py`, K = 12), brain-age gap (already in-fold), ALZ (cross-fitted CNN scores, unchanged; out of scope), the KAN (config 32 and control 34, `kan_fusion.fit_predict` imported unchanged).
- **Systems (Table II CV column):** clinical specialist, MRI specialist (567 with a scan), KAN K2 config 32, CatBoost F, CatBoost R, and config 34 (MRI masked).
- **Comparisons:** KAN − clinical, KAN − CatBoost F, KAN − config 34.
- **Report and rule:**
  - Each number is reported next to the original.
  - A **reading changes** if a comparison's CI changes between "excludes 0" and "includes 0".
  - Any system whose CV macro-F1 shifts by > 0.010 in absolute value is flagged **noteworthy**.
- **Experiment 2 (all tiers):**
  - The code of `880f4a7:experiments/exp2_reduced_clinical/code/` is copied into `analysis/reviewer_response/exp2_foldwise/`, with a header naming the source commit.
  - Exactly two changes are made:
    - (a) the MRI stream's 6 ratios are raw (volume / ICV) and imputed/scaled in fold, inside the MRI specialist and in F-CB;
    - (b) tier Full-paper uses in-fold Prep for its 29 features and its KAN extras (AGE, CDRSB, CogIndex, MMSE).
  - All tiers are re-run on CV and val.
  - R1–R5 and the reproduction gate are applied exactly as in EXP2's PREREGISTRATION.md (commit `de368a1`, addendum `3744036`).
  - Every tier whose R1–R5 outcome differs from `880f4a7:experiments/exp2_reduced_clinical/results.json` is listed.
  - Deviation from EXP2: the reproduction gate is reported but does **not** stop the run, since changing preprocessing is the point.

## B3 KAN edge curves (frozen 25-model ensemble, no refitting)

- **Models:** `MultKAN(width=[13, 5], grid=5, k=3)` per `frozen/kan_config.json`, loaded from `frozen/kan_{s}_{ks}.pt` with scalers `frozen/kan_inputs_{s}_{ks}.pkl`.
- **Validity gates** (all must pass, otherwise B3 stops and the reason is reported):
  - (a) Reloaded models reproduce the stored per-model val predictions (`frozen_run/val_kan.npz`) within 1e-5.
  - (b) For the 691 fitting rows, summing the per-edge activations (pykan `spline_postacts` / per-edge outputs incl. mask) and applying the output node affine reproduces `model(x)` within 1e-5.
- **Evaluation grid:**
  - 101 equally spaced points per input over its [min, max] across the 691 fitting rows, in raw (unscaled) input units, mapped through each model's own scaler.
  - MRI logits use rows with a scan only. `has_mri` is evaluated at {0, 1} only.
- **Output:**
  - For each of the 65 edges (13 inputs × 5 outputs): the median and 10th/90th percentiles across the 25 models, as CSV + figure.
  - Pruned or masked edges contribute 0.
  - Curves are a model description, not an explanation.

## B4 Operating characteristics (stored predictions only, no new inference)

- **Threshold:**
  - The app defines no AD-vs-rest probability threshold. The stage shown is the argmax (`src/assessment/report_builder.py:120`). The AD probability shown is the KAN ADD head with no cut-off (`report_builder.py:278`), both at `ffbabe6`.
  - Primary: ADD-head probability ≥ 0.5 (as instructed when none is defined).
  - Secondary: the app's stage call (argmax stage == AD).
- **Sensitivity and specificity** (AD vs rest), with bootstrap CIs:
  - on CV out-of-fold predictions (`kan_runs/32.npz`: `cv_add`, `cv`; per seed, then averaged);
  - on the stored frozen test predictions (`test_kan.npz`: `add`, `p`), labelled **"post-hoc metric on already-read test predictions; no model change"**.
- **Reliability of model probabilities** (CV out-of-fold only):
  - Brier score for the ADD head; multiclass Brier (sum over 4 classes) for the stage output.
  - ECE with 10 equal-width bins on [0, 1]: ADD head (fraction AD vs mean probability) and stage output (top-label confidence vs accuracy). Per seed, then averaged; bootstrap CIs.
  - Reliability diagram from the 5 seeds' predictions pooled, with bin counts.
  - Descriptive only: no pass/fail rule. The words "calibrated"/"calibration" are not used to describe the models.
