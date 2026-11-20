# SynthSeg vs training volumes: descriptive agreement check (not a gate)

Written 2026-10-28, before any SynthSeg run on these subjects. It is a check, not a pre-registered gate: there is no pass/fail, no harmonisation and no retraining. Its numbers are quoted in a caveat on every report that uses SynthSeg volumes.

## Question
How closely do SynthSeg volumes of a native ADNI scan agree with the FreeSurfer 5.1 volumes (ADNIMERGE2 UCSFFSX51, joined in step 2) that the MRI specialist was trained on, for the six model-input ratios?

## Sample
25 subjects from the 569 clinical-train subjects only (never val or test). Seed 0: `default_rng(0)`, 7 CN + 6 EMCI + 6 LMCI + 6 AD (the 7 extra is CN), drawn among train subjects that have a selected baseline scan (`dataset/mri_raw/selected_scans.csv`) and a complete FreeSurfer row (all six volumes and ICV). If a tool run fails for a subject, the failure is reported and the subject is not replaced.

## Inputs
- SynthSeg: the **native** selected baseline scan (the N3m-processed NIfTI under `datasetnew_extracted`), i.e. what an uploaded scan would be. FreeSurfer 7.4.1 `mri_synthseg` 2.0, `--parc --cpu --threads 16 --crop 160 192 160`, run alone (about 11.5 GB RAM), in `.venv-synthseg`. This is the invocation of `analysis/mri_morphometry/run_synthseg.py`.
- Reference: the six volumes and ICV from `step2_features.csv` (read-only, FreeSurfer 5.1).
- Segmentation sanity: report the fraction of scans whose segmentation touches the analysis patch boundary (a sign of clipping).

## Label mapping (from `analysis/mri_morphometry/smoke_analysis.py`)
Hippocampus = L+R hippocampus; Ventricles = L+R lateral + inferior lateral ventricle; Entorhinal, Fusiform, MidTemp = ctx-lh + ctx-rh of the same names; WholeBrain = every non-cortical-parcel brain label except CSF and the four ventricle labels; ICV = SynthSeg "total intracranial".

## Metrics, on the six model-input ratios (volume / ICV, each tool with its own ICV)
- ICC(2,1) (two-way random, absolute agreement) with a 95% bootstrap CI (2000 resamples, seed 0).
- Bland-Altman bias as a percentage of the FreeSurfer mean, and the 95% limits of agreement.
- Also reported for context: the same two metrics on raw volumes and on ICV.

## Use of the result
- Aggregate numbers only are committed (`results/agreement.md`, `results/agreement.json`). Per-subject values stay in gitignored `private/`.
- The report caveat quotes them: "volumes from SynthSeg; agreement with training volumes: ICC(2,1) …".
- Rule for the report text (declared now): if the ICC(2,1) point estimate of any of Entorhinal, Fusiform or MidTemp ratio is below 0.50, reports using SynthSeg volumes add that the MRI-specialist contribution may be unreliable for that scan.

## Amendment, 2026-10-28, before any run on the sample
The raw native NIfTIs were not kept after m1 preprocessing: only 73 of the 811 selected baseline scans are still on disk (under `datasetnew_extracted`, matched by Image ID), of which 50 belong to clinical-train subjects with a complete FreeSurfer row (CN 7, EMCI 23, LMCI 14, AD 6). The sample is therefore drawn from these 50 (same seed and 7/6/6/6 quota, which is feasible: all 7 CN and all 6 AD are taken, 6 EMCI and 6 LMCI drawn with `default_rng(0)`). Everything else is unchanged.

## Amendment, 2026-10-29: display rule for SynthSeg reports (declared before any SynthSeg report is built)
Result of the check (`results/agreement.md`): 25 of the 73 raw native scans still on disk were used, so the confidence intervals are wide (for example Fusiform ICC 0.28 [-0.21, 0.66], WholeBrain 0.52 [0.16, 0.70]). The numbers are descriptive, not a validation. Rules for reports whose volumes come from SynthSeg:
1. A volume percentile is shown only for ratios with ICC(2,1) >= 0.50 **and** |bias| <= 10%. On the committed numbers that is WholeBrain (0.52, +1%) and Hippocampus (0.85, +9%). Ventricles (bias +21%), Entorhinal, Fusiform and MidTemp show the raw ratio labelled "low agreement with training volumes, no percentile".
2. The "MRI changed the stage" flag is suppressed.
3. The agreement caveat is a WARN-level item, first in the MRI QC section/panel, quoting the ICC and bias numbers.
4. These rules are enforced by the schema validators (`src/assessment/schema.py`): a SynthSeg report cannot carry a percentile for a low-agreement ratio. `agreement.percentile_ok()` derives the set from the committed numbers, and a test asserts it equals the schema constant.
(There is no separate `src/assessment/volumes/PLAN.md`; this file is the plan for the volume provider.)
