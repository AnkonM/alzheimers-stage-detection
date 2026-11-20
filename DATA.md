# Data and model availability

## Not distributed

| Item | Reason |
|---|---|
| ADNI tables (clinical, diagnostic, FreeSurfer, scan lists) and anything derived at subject level (features, splits, predictions, SHAP values, per-subject volumes) | The ADNI data use agreement does not allow redistribution. Download from LONI IDA (adni.loni.usc.edu) after approval. |
| T1 scans, preprocessed volumes, QC renders, overlays and attribution maps of individual subjects | Same; derived images are covered too. |
| OASIS-1 images and tables | Available from the OASIS project under its own terms; not redistributed. |
| Trained weights (CatBoost, logistic regression, KAN, CNN checkpoints, DeepBrainNet) | The ADNI agreement does not clearly cover trained models, and models fitted on a few hundred subjects can leak information about them. Weights can be shared with people who hold their own ADNI approval; contact the authors. |

Everything that is distributed is code, documentation, synthetic test fixtures or aggregate results (group-level tables,
metrics with confidence intervals, group-averaged maps on the ICBM152 template).

## How the code expects the data

- ADNI tables go under `dataset/raw/` (file names are listed in `src/step0_join_tables.py`).
- ADNI T1 NIfTI files: set `ADNI_NIFTI_GLOB` to a glob such as `adni_nifti/ADNI/*/*/*/I*/*.nii`.
- OASIS-1 working area: set `OASIS_WORK` (default `oasis_work/`).
- Weights restored to their original locations (`analysis/fusion_v3/frozen/`, `analysis/fusion_feasibility/runs/`)
  are checked against `src/assessment/artifacts/registry.json` (SHA-256) before use.

## Checking a release

`tools/dua_scan.py --tree <dir> --allow tools/dua_allow.txt` flags files that look like subject-level data, images,
weights or archives; the export of this repository was required to pass it. It reports aggregate counts only.
The thirteen figures it lists for manual review are aggregate plots (confusion matrices, group-average
maps on the template, SHAP and KAN function plots).
