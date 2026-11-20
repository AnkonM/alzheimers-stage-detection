# Reproducing the analyses

Prerequisite: your own ADNI access (and, for the external validation, the OASIS-1 download). Nothing here can be
run without them; see [../DATA.md](../DATA.md). Use the environments in `env/locks/` rather than one merged environment:
the scikit-learn pickles, CatBoost, pykan, TensorFlow and SynthSeg have conflicting pins.

Order of the steps (each folder under `analysis/` has its own report and, where applicable, a pre-registration):

1. **Clinical pipeline.** `src/step0_join_tables.py` to `src/step5_catboost_shap.py`: join tables, integrity checks,
   feature selection, subject-level split, processing, CatBoost with SHAP.
2. **MRI preprocessing.** `src/mri/m1_preprocess.py` (bias correction, affine registration to ICBM152, template mask);
   scan selection and unpacking scripts are in `dataset/mri_raw/` and `dataset/mri_manifest/`.
3. **CNN and cross-fitting.** `src/mri/m3_train.py`, then `analysis/fusion_feasibility/` (5 folds x 3 seeds).
4. **Brain age and volumes.** `analysis/brain_age/`, `analysis/mri_morphometry/`, `analysis/volume_agreement/`.
5. **Fusion.** `analysis/kan_fusion/`, then `analysis/fusion_v3/` (tuning log in `config_log.csv`, freeze in `freeze.md`).
   The test set is read once, after the freeze; `freeze_guard.py` enforces this.
6. **Explanations.** `analysis/explainability/`.
7. **External validation.** `src/validation/`, `analysis/external_validation/`, `tools/build_oasis_tables.py`.
8. **Reviewer-response and experiment 2.** `analysis/reviewer_response/` (`exp2_foldwise/run_all.sh`),
   `experiments/exp2_reduced_clinical/`.
9. **Figures.** `figures/run_all.sh`.

Shell scripts that call a Python interpreter read it from the `PY` environment variable (default `python`).
