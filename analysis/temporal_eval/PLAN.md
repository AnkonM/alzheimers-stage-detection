# Temporal-scan evaluation of the frozen fusion-v3 system — PLAN (preregistered)

Branch `eval/temporal-scans` (from `9bac529`). Written 2026-10-11, committed
before any prediction is computed and before anything is run on TEST subjects.
Read-only evaluation: no retraining, retuning, refitting of any scaler /
imputer / selector, no hyperparameter change, no change to any file in
`analysis/fusion_v3/frozen/` or to `analysis/fusion_v3/freeze.md`.

Anything decided after results exist goes in the **Post-hoc** section at the
end, dated, and never replaces a rule above it.

## 0. Already done before this commit

- Frozen checksum list `analysis/fusion_v3/frozen/MANIFEST.sha256`:
  109 / 109 files OK (`sha256sum -c`). It is re-checked at the start of every
  run script; any failure stops the run.
- Code read only (no data rows inspected per group): the frozen test-time
  chain, step 0–4, `visits_index.csv` schema, ADNI table schemas. Finding used
  below: `DXSUM.DIAGNOSIS` takes only CN / MCI / Dementia at every visit in
  every phase, including ADNI-GO / ADNI-2. EMCI / LMCI exist only through the
  enrollment arm (`ARM.csv`), which step 0 applies to the baseline diagnosis.

## 1. Data, groups, in-sample status

- **Scans:** the temporal ingestion is `dataset/mri_prep_patha/visits_index.csv`
  (Path A, cohort `patha_bl_m6_m12_m24`, slots bl / m06 / m12 / m24). A scan is
  *usable* if `present & prep_ok`; the Path A masks (`DUPLICATE_SCAN`,
  `DATE_OUT_OF_ORDER`, `REG_FAILED`, `REG_FAILED_LAYOUT`, `NO_ROW`, manifest
  `MISSING`) are counted and reported as preprocessing failures, never
  silently dropped. The images are those already preprocessed by the unchanged
  m1 chain (`dataset/mri_prep_patha/{vol96,mni,qc}`); they are not
  reprocessed.
- **Baseline vs follow-up:** slot `bl` = baseline scan; m06 / m12 / m24 =
  follow-up scan.
- **Groups** (by RID, from the saved split files
  `dataset/processed/{train,val,test}_rids.csv`): TEST, VAL, TRAIN, or
  NOT_IN_COHORT (RID not among the 814). Every subject is in exactly one group;
  the script asserts this.
- **In-sample:** the frozen final system was fitted on TRAIN + VAL (all 691
  subjects, `freeze.md`). TRAIN and VAL rows are **IN-SAMPLE** in every table
  and are never pooled with out-of-sample rows. Out-of-sample = TEST,
  NOT_IN_COHORT, and **OOS = TEST + NOT_IN_COHORT** (the out-of-sample estimate).

## 2. Ground truth

1. **Visit match.** Each scan is matched to the DXSUM row of the same RID
   (non-missing `DIAGNOSIS` and `EXAMDATE`) with the nearest `EXAMDATE` to the
   scan date, if within **±90 days** (ties → the earlier visit). Sensitivity
   run: **±180 days**. No match → excluded, counted.
2. **Mapping** (the main-analysis mapping, `src/step0_join_tables.py`
   `combine_dx_bl` + `src/step1_data_integrity.py` `LABEL_MAP`):
   - matched visit is the subject's baseline visit (DXSUM `VISCODE2` in
     {`sc`, `scmri`, `bl`}): true stage = the main analysis's baseline label,
     `combine_dx_bl(arm category, DXSUM baseline diagnosis)` mapped CN=0,
     EMCI=1, LMCI=2, AD=3; SMC or plain MCI → UNDEFINED.
   - any other visit: CN → CN, Dementia → AD, **MCI → UNDEFINED** (plain MCI;
     EMCI / LMCI are not defined by ADNI after the enrollment visit and are not
     guessed, e.g. not carried from the arm).
   - UNDEFINED scans are excluded from all four-stage metrics and counted.
     Consequence stated in advance: **follow-up four-stage metrics cover only
     CN and AD scans.**
   - For cohort subjects' baseline scans the derived true stage is compared
     with the frozen label (`step2_features.label`); disagreements are reported.
3. **Secondary 3-class truth** (CN / MCI / AD, from DXSUM at the matched visit,
   MCI kept): used only for the pre-specified secondary 3-class metrics
   (§5), so follow-up MCI scans are not invisible.

## 3. Inputs per scan (frozen preprocessing, no refitting)

### 3.1 Clinical record at the same visit

- **Baseline scans of cohort subjects:** the subject's row of
  `dataset/processed/step2_features.csv` — the exact record the frozen system
  used.
- **Follow-up scans, and baseline scans of NOT_IN_COHORT subjects:** a
  per-visit builder (`build_visits.py`) that applies the step 0 / step 2
  derivations to the rows of the **matched DX visit only** (same RID and same
  `VISCODE2`; for a baseline visit, step 0's baseline codes per table). No row
  from any other visit is used — in particular **a baseline clinical record is
  never used for a follow-up scan**, and there is no last-observation-carried-
  forward. Sources:
  - CDRSB: `dataset/raw/CDR.csv` `CDRSB`;
  - ADAS11 / ADAS13: `dataset/raw/ADAS_ADNIGO2.csv` `TOTSCORE` / `TOTAL13`;
  - MMSE: `MMSE.csv` `MMSCORE`;
  - FAQ: `FAQ.csv` `FAQTOTAL`;
  - RAVLT immediate / learning / forgetting / percent forgetting: from
    `NEUROBAT.csv` (AVTOT1–5, AVDEL30MIN), using ADNIMERGE's definitions;
  - MOCA and the 14 ECog scores: step 0's own derivations (`derive_moca_total`,
    `derive_ecog_scores`) on that visit's row;
  - FreeSurfer volumes: §3.2;
  - subject-level constants (sex, education, APOE4) from ADSL as in step 0;
    these are not visit records;
  - AGE = ADSL `AGE` + (scan date − `ENRLDT`) / 365.25 (age at the visit).
- **Builder check (part of gate G1c):** on the baseline visit of the 814
  cohort subjects the builder must reproduce `step2_features.csv` per feature
  (|diff| ≤ 1e-6, or both missing) for **≥ 98 %** of subjects for every
  feature. A feature below 98 % → STOP and report, unless the cause is a
  documented step-0 earliest-visit fallback (then logged in Post-hoc and the
  affected feature's agreement on non-fallback rows must be ≥ 98 %).
- **Clinical record present** (same gate as step 1's CogIndex rule): CDRSB
  non-missing and at least one of MMSE / ADAS13 non-missing at that visit.
  Otherwise the scan is excluded from the full-pipeline and clinical-only
  evaluations (counted) and kept for the MRI-only evaluation. Other missing
  features are imputed by the frozen medians/modes (frozen preprocessing);
  per-feature missingness is reported for baseline vs follow-up.

### 3.2 FreeSurfer volumes

- Baseline scans of cohort subjects: the volumes in `step2_features.csv`.
- Other scans: `UCSFFSX51.csv` row with `IMAGEUID` equal to the scan's image ID;
  else the row with the same RID and the scan's `VISCODE2` (earliest
  `EXAMDATE`, step 0's tie rule); else **unavailable**. No QC filter (step 0 used
  none).
- Unavailable volumes: imputed by the frozen ratio medians, exactly as the
  frozen preprocessing does; the scan is kept, flagged and counted.
  Sensitivity: all metrics rerun excluding those scans.

### 3.3 Scan-derived inputs

- **ALZ** (frozen CNNs, not retrained), from `vol96/<PTID>__<slot>`:
  - TEST, VAL, NOT_IN_COHORT: mean over the 5 cross-fit fold models of each
    model's 3-seed mean sigmoid(ALZ logit) — the frozen test-time rule
    (`scan_inputs.py`).
  - TRAIN: the out-of-fold model only — the 3-seed mean of the fold model that
    did not train on that subject (fold from `alz_crossfit.csv`). This is how
    the frozen system's own TRAIN ALZ was made (cross-fitted), so it is used
    for all of a TRAIN subject's scans. (VAL's frozen ALZ was the 5-fold mean;
    same here.)
- **Brain age:** DeepBrainNet, frozen weights (sha256 checked), on
  `mni/<PTID>__<slot>`; age at scan = ADSL AGE + (scan date − ENRLDT)/365.25.
  No brain age if `corr_vs_mni` < 0.15.
- **Reliability rule** (frozen): unreliable if the mni volume has no zero voxel
  inside the template mask, or > 10 % of voxels ≥ 0.999. Unreliable / missing
  brain age → BAG NaN → frozen median imputation (frozen rule). Counted.
- `has_mri` = 1 for every usable scan.

## 4. Systems (inference only, frozen code paths)

All use `analysis/fusion_v3/frozen/` through the functions of `final_features.py`,
`step4_processing.py`, `test_streams.py`, `test_kan.py`, `kan_fusion.py`.

1. **Full pipeline:** clinical specialist + MRI specialist + residual KAN,
   mean of the 25 KANs (COG probabilities).
2. **MRI-only:** the MRI specialist (6 ratios, ALZ, BAG, AGE, SEX), mean of
   the 5 seeds.
3. **Clinical-only:** the clinical specialist, mean of the 5 seeds, on the same
   records as (1).
4. **Always-EMCI:** p = one-hot EMCI.
5. **Stage prior:** p = TRAIN+VAL (691) class prior; hard label drawn from the
   prior with `numpy.random.default_rng(20261012)` in the scan order of §6.
   Its expected accuracy Σ prior_k · share_k is also reported.

Evaluation sets: FULL set = usable scans with a defined four-stage truth and a
clinical record (systems 1–5); MRI set = usable scans with a defined truth
(systems 2, 4, 5). Paired differences use the FULL set.

## 5. Metrics

For each system × group (TEST, VAL [IN-SAMPLE], TRAIN [IN-SAMPLE],
NOT_IN_COHORT, OOS) × timepoint (baseline, follow-up, all):
- n scans, n subjects, per-stage counts;
- accuracy; balanced accuracy and macro-F1 **averaged over the classes present
  in the evaluated truth** (identical to `metrics.py` when all four are
  present); the `metrics.py` 4-class definitions are also printed; per-class
  precision / recall / F1;
- 4×4 confusion matrix (counts);
- mean |predicted − true stage|; share of errors that are adjacent (1 stage)
  vs ≥ 2 stages;
- AD-vs-rest AUC from P(AD);
- **cluster bootstrap over subjects**: 2,000 resamples, subjects with
  replacement (all of a subject's scans in the evaluated cell travel together),
  `numpy.random.default_rng(20261012)`, percentile 95 % intervals on accuracy,
  macro-F1, AD AUC. A resample with no AD or no non-AD scan is skipped for AUC;
  the number of valid resamples is reported;
- full − clinical-only and full − MRI-only differences in accuracy, macro-F1,
  AD AUC, with **paired** cluster-bootstrap intervals (same resampled subjects
  for both systems).
- Secondary (pre-specified): 3-class CN / MCI / AD accuracy and macro-F1
  (`metrics3` collapse, EMCI+LMCI → MCI), on all usable scans with a defined
  3-class truth, including follow-up MCI.

## 6. Sampling

- Seed **20261012**.
- **Primary:** ALL eligible scans per group.
- **Random subsample (as requested):** within each group, for each true stage
  (CN, EMCI, LMCI, AD), draw up to **40** scans without replacement (fewer if
  the cell is smaller; shortfall reported). Scans are sorted by (PTID, scan
  date) and drawn with one `numpy.random.default_rng(20261012)` in the fixed
  order TEST, VAL, TRAIN, NOT_IN_COHORT × CN, EMCI, LMCI, AD. Drawn once from
  the FULL set and once (fresh generator, same seed) from the MRI set. The
  subsample is stage-balanced by design, so its accuracy is not a
  natural-prevalence estimate; it is reported beside, never instead of, the
  all-scan results, with the same baseline / follow-up split.
- No metric is reported on any hand-picked subset. The only subsets are the
  pre-specified ones in this plan.

## 7. Sanity gate (must pass before any metric is reported)

- **G1a (strict).** The evaluation's inference wrapper, fed the frozen stored
  TEST baseline inputs (`step2_features` rows, `scan_inputs_test.csv`,
  `brainage_test.csv`), reproduces `frozen_run/test_streams.npz` (all keys) and
  `frozen_run/test_kan.npz` (`p`, `add`) with max |diff| ≤ **1e-6** and
  identical argmax for every subject.
- **G1b (strict).** The ALZ scorer on the original baseline volumes
  (`dataset/mri_prep/vol96`) reproduces `scan_inputs_test.csv` ALZ to
  ≤ **1e-5**, and the brain-age wrapper on `dataset/mri_prep/mni` reproduces
  `brainage_test.csv` pred_age to ≤ **0.001 years**, for every TEST subject.
- **G1c (strict).** The builder check of §3.1.
- **ALZ provenance check:** for TRAIN (VAL) baseline scans, the ALZ computed
  from the *original* baseline volume by the §3.3 rule equals
  `alz_crossfit.csv` ALZ to ≤ 1e-5 on the first 20 subjects of each, which
  confirms the frozen system's TRAIN (cross-fitted) and VAL (5-fold mean) ALZ.
- **G2 (descriptive, not pass/fail).** The ingestion's own TEST baseline scans
  are a different IDA derivative file of the same session (Path A Stage B:
  median r = 0.987 against the original volumes), so their predictions cannot
  match the stored ones to saved precision. Reported: argmax agreement with the
  stored test predictions, max and median |Δp|, ALZ and brain-age differences.
- If G1a, G1b or G1c fails: STOP, report what differs, no metrics.

## 8. Interpretation rules (fixed now)

- Reference values (frozen test, 123 subjects): full pipeline accuracy 0.764,
  macro-F1 0.764; MRI-only accuracy 0.504.
- OOS result **"consistent with"** the earlier test result if the earlier value
  lies inside the OOS 95 % interval; **"below"** if the interval's upper end is
  below it; **"above"** if the lower end is above it. No other success/failure
  wording.
- OOS (TEST + NOT_IN_COHORT) overlaps the subjects of the earlier test; it is
  descriptive (§9), not a new confirmatory test.
- In-sample overstatement = in-sample (TRAIN, VAL) minus OOS, per metric,
  descriptive.
- Follow-up vs baseline: difference with a cluster-bootstrap interval
  (unpaired across scans, subjects resampled within the OOS group); "drops" only
  if the interval excludes 0. Because follow-up truth is CN/AD only, the
  comparison is also made on baseline CN/AD scans only (like-for-like).

## 9. Outputs and data-use agreement

- Per-scan / per-subject outputs go to `analysis/temporal_eval/private/`
  (gitignored). Committed: this plan, code, aggregate tables, confusion
  matrices (counts), the gate summary, the report.
- Commits: (1) this plan; (2) code; (3) results + `report.md`.

## 10. Limitations known in advance

- TEST subjects were read twice before (an earlier clinical model, then the
  frozen fusion); TEST here is descriptive.
- Follow-up diagnoses may differ from baseline; follow-up truth has no
  EMCI / LMCI.
- Label definitions differ by ADNI phase.
- Scans from one subject are correlated (hence the subject-level bootstrap).
- Follow-up visits have a smaller test battery (missing features imputed with
  frozen medians), a distribution shift by design.

## Post-hoc

(none yet)
