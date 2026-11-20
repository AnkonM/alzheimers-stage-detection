# Alzheimer's disease stage detection from clinical data and T1 MRI

Code, analysis scripts and written reports for a four-stage classifier (CN, EMCI, LMCI, AD)
that fuses a clinical model and an MRI model, developed on ADNI and externally checked on OASIS-1.

**Research code. This is not a diagnostic device and must not be used for clinical decisions.**

## Method in brief

- **Clinical specialist.** Five-seed CatBoost ensembles on the twelve top-ranked features of a 29-feature clinical
  set (`analysis/fusion_v3/`).
- **MRI specialist.** Logistic regression on six FreeSurfer-style volume ratios, a 3D ResNet AD score
  (cross-fitted: 5 folds x 3 seeds), a DeepBrainNet brain-age gap, age and sex (`src/mri/`, `analysis/fusion_feasibility/`,
  `analysis/brain_age/`, `analysis/mri_morphometry/`).
- **Fusion.** A residual Kolmogorov-Arnold network (KAN) over the specialist outputs, compared with CatBoost
  and stacking baselines (`analysis/kan_fusion/`, `analysis/fusion_v3/`).
- **Explanations.** SHAP for the tabular models, learned-function plots for the KAN, and LayerCAM for the CNN
  with pre-registered faithfulness checks, which the CNN maps only partly pass (`analysis/explainability/`).

The ADNI subject split (569 train / 122 validation / 123 test) is made once at subject level. The final system was
frozen on commit `8f351b6` before the test set was read; the test-time scripts refuse to run otherwise.

## Main results

All numbers below come from `docs/final_results.md` and `docs/validation/`, which list their sources.

| | CV macro-F1 (train, 5 seeds) | Test macro-F1 (n = 123) |
|---|---|---|
| Clinical specialist | 0.744 | 0.771 |
| MRI specialist | 0.424 | 0.438 |
| KAN fusion | 0.756 | 0.764 |
| CatBoost on the fusion inputs | 0.758 | 0.746 |

Every paired difference between the fusion model and the single-stream or CatBoost alternatives has a bootstrap
interval that includes zero. The fusion model is therefore reported as matching the best single stream, not
exceeding it, and the contribution of the MRI inputs inside the fusion is not reliably above zero in cross-validation.

On OASIS-1 (pre-registered, `docs/external_validation_protocol.md`) the CNN AD score separated CDR 0 from CDR >= 1
in the primary comparison with AUC 0.874 (95% CI 0.795 to 0.941; 30 vs 98 subjects). Full tables, secondary analyses,
the domain-shift analysis and the exclusion accounting are in `docs/validation/oasis_results_tables.md`.

## Repository layout

| Path | Contents |
|---|---|
| `src/step0_*.py` to `src/step5_*.py` | clinical data pipeline: table joins, integrity checks, features, subject-level split, CatBoost + SHAP |
| `src/mri/` | MRI preprocessing, dataset and 3D CNN |
| `src/assessment/`, `src/api/`, `frontend/` | inference engine, report builder, local web application |
| `src/validation/` | OASIS-1 external validation code |
| `analysis/` | one folder per experiment: pre-registration, scripts, aggregate results and reports |
| `experiments/exp2_reduced_clinical/` | reduced-clinical-input experiment (pre-registration, code, aggregate tables) |
| `analysis/reviewer_response/` | additional analyses requested in review (stacking baselines, fold-wise preprocessing, operating points) |
| `docs/` | final results, model card, external-validation protocol, experiment reports |
| `figures/` | scripts that rebuild the paper figures from saved result files |
| `tests/` | unit and integration tests |
| `tools/` | model-card and table generators, the data-release scanner (`dua_scan.py`) |
| `env/locks/` | exact package versions of the separate Python environments |

## What is not in this repository

No ADNI or OASIS-1 data, no subject-level tables, no scan-derived images and no trained model weights are
distributed. See [DATA.md](DATA.md) for what is withheld, why, and how to obtain the inputs.
Consequently the experiments cannot be re-run from a plain clone; [docs/REPRODUCING.md](docs/REPRODUCING.md)
describes the order of the steps for someone who has their own ADNI access.

## Setup and tests

The work uses several Python environments on purpose (scikit-learn pickles, CatBoost, pykan, TensorFlow for
DeepBrainNet and the SynthSeg build have conflicting pins). Lockfiles for each are in `env/locks/`, with
rebuild notes in `env/locks/README.md`. `requirements.txt` covers the clinical and analysis environment.

    python -m venv .venv-app
    .venv-app/bin/pip install -r requirements.txt pytest
    .venv-app/bin/python -m pytest tests --ignore=tests/e2e --ignore=tests/parity

Tests that need weights or restricted data are skipped automatically when those are absent (`tests/conftest.py`).

## Pre-registration and audit trail

Analyses were specified in dated pre-registration documents committed before the corresponding results
(`analysis/*/preregistration.md`, `docs/external_validation_protocol.md`, `experiments/exp2_reduced_clinical/PREREGISTRATION.md`,
`docs/reviewer_response_prereg.md`). Deviations are recorded as dated addenda in the same files, and generated
tables state which script produced them. This is a fresh repository: the original private development
history, which contains restricted data, is not published, and commit hashes quoted in the documents refer to it.

## Limitations

- ADNI-GO/2 cohort of 814 subjects with a four-class label; the held-out test split has 123 subjects (24 LMCI),
  so test intervals are wide (about +/-0.07 macro-F1).
- MRI volumes at inference time come from SynthSeg, whereas the MRI specialist was fitted on FreeSurfer 5.1
  values; agreement was measured on a small sample (`analysis/volume_agreement/`), so the MRI contribution may be unreliable on new scans.
- The model outputs have not been calibrated; see the model card for what has and has not been measured.
- External validation is on one dataset (OASIS-1), with CDR-based labels instead of the four ADNI stages.

## Data acknowledgements

Data used in the development of this work were obtained from the Alzheimer's Disease Neuroimaging Initiative
(ADNI) database (adni.loni.usc.edu). The investigators within ADNI contributed to the design and implementation
of ADNI and/or provided data but did not participate in the analysis or writing of this work. A complete listing
of ADNI investigators is at
https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf.

OASIS-1 data: Marcus DS, Wang TH, Parker J, Csernansky JG, Morris JC, Buckner RL. Open Access Series of Imaging
Studies (OASIS): cross-sectional MRI data in young, middle aged, nondemented, and demented older adults.
J Cogn Neurosci 2007;19(9):1498-1507. Principal investigators: D. Marcus, R. Buckner, J. Csernansky, J. Morris;
P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382.

## Citation and licence

Cite via [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button). Code and documentation are under
the MIT licence ([LICENSE](LICENSE)); the licence does not extend to ADNI or OASIS-1 data or to models derived from them.

Authors: Ankon Mukherjee, Affan Shaikh, Eshani Kane, Aneeket Banerjee.
