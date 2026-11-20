# External validation of the frozen fusion-v3 system on OASIS-1 (cross-sectional): pre-registered protocol

**Status: final text prepared 2026-11-14 (earlier revisions are in the git history). Frozen by the local annotated tag `oasis-protocol-v1`; the tag message records the commit and the pinned hashes. At the time of tagging no OASIS image has been passed through any model.**
The body is frozen when the local tag `oasis-protocol-v1` is created, which happens before any model inference on OASIS data (Stage D, the analysis script, and the merge of addendum A1 into section 4a are completed before it). After the tag, changes are made only as dated addenda at the end of this file.

Data were provided by OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, R. Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382.
Citation: Marcus et al., J Cogn Neurosci 2007;19(9):1498-1507, https://doi.org/10.1162/jocn.2007.19.9.1498

## 1. Purpose
Evaluation feedback required a test on an external dataset. This protocol tests whether the MRI-derived outputs of the frozen system generalise to an independent cohort, a different scanner population and a different diagnostic label (CDR). It is a generalisation test, not a tuning step.

## 2. Frozen-system declaration
- Code commit: **HEAD at tag**, the HEAD of the `val/oasis1` branch at the moment the tag `oasis-protocol-v1` is created (recorded in the tag message and, as `git_commit`, in the run header). Engine parity tag `engine-parity-pass` = 79e53b5dffb22d4b6549684be1c98e57bb47c5bf. Freeze commit 8f351b6.
- Hashes the tag pins (each is verified by a test and again immediately before the run, and logged):
  - `analysis/fusion_v3/frozen/MANIFEST.sha256` (109 entries): sha256 `5e266705ca96290c4e6daaf1476317aeb0c71a07346ff16d02b5b360d7cbde92`.
  - artifact registry `src/assessment/artifacts/registry.json` (127 entries): sha256 `2e0f4ebabdd33024969c96b0e8139956485971c612d76edc741b2c78940b1448`.
  - DeepBrainNet weights (a registry entry): sha256 `9257e98e9aff88bfcd83acdb1e74ffde762967503ef6df16c9270aa39d9941ea`.
  - `inputs.csv` of the manifest (data area, not in git): sha256 `7ede554d051a4442ff39dd912b06f4115ea651c0a008882f26c86d08011a30fd`. This is the Stage A file after the `orientation` column was set to `oasis1_primary` (addendum A1; before that change its hash was `1dc4f56c4b380884eefc8f5662f63bede9fe45da74185a3ad908be6fd8ddb874`).
  - `labels.csv` of the manifest (data area, not in git): sha256 `b78d1258d72e30e9d112ad845ae6b1e422bc4ae0990eaadac247a5d860252196`.
  `sha256sum -c` of the manifest and `python -m src.assessment.registry verify` must pass immediately before the run. The run header pins the git commit, the `inputs.csv` hash and the registry hash, and the analysis script refuses to open `labels.csv` unless the run is complete, its pinned `inputs.csv` hash matches the file, and the `labels.csv` hash equals the value above.
- **No training, tuning, threshold selection, calibration or recalibration on OASIS data.** No frozen artifact, golden parity result or earlier verdict is altered. The parity run (`python -m tests.parity.run_parity`) is repeated at the run commit and must pass with unchanged tolerances.
- The run commit may differ from the tag commit. Any change under `src/` after the tag is recorded as a dated addendum that names the run commit, and the gates (G-freeze, pytest, oasis_guard, G-adni-bytes, G-parity) are re-run at that commit before Stage C.
- Preprocessing (`src/mri/m1_preprocess.py`) is the unchanged code used for ADNI. It is not bit-exact (identical reruns agree at voxel r 0.997), the ADNI scans were preprocessed on another machine with an unrecorded ANTsPy version, and no test covers native scan to `vol96`. This is a known uncertainty and is not corrected.
- All inference runs through the assessment engine (`predict-batch`), MRI_ONLY mode, one scan at a time (batch size 1). Single-scan ALZ differs from the frozen batch run by up to about 5e-5 (`tests/parity/DRIFT.md`), which is immaterial for the analyses below.
- **Memory and time.** SynthSeg alone peaks at **12.62 GiB** resident on one ADNI validation scan (13,231,404 kB by `/usr/bin/time -v`, 16 threads) and at up to **13.61 GiB** over a 10-scan ADNI soak (peak memory varies by scan), on a VM with 19,999 MB RAM and 32 GB swap. With the four stage workers resident (6.8 GB, of which DeepBrainNet 4.4 GB and the MRI worker 1.7 GB) it does not fit. The runner therefore **stops the mri and dbn workers before every SynthSeg call** (they restart on demand afterwards; no report content changes) and checks `MemAvailable` immediately before the call against a **floor of 15.2 GiB** (the largest measured peak, 13.61 GiB, plus 1.5 GiB, rounded up to 0.1 GiB). Below the floor it removes the half-done case, records a halt (sizes only), stops without a completion marker and exits non-zero; `--resume` continues. Measured through the real engine on one ADNI validation scan with this release logic: **about 79 s per scan warm and 104 s cold**, with `MemAvailable` of 16.3 to 16.8 GiB before SynthSeg. A 10-scan ADNI soak through `predict-batch` with the same release logic (run with the earlier floor of 14.5 GiB; it was not repeated at 15.2 GiB, but every reading below was above 15.2 GiB) finished with no halt, took 75 s per scan on average (69 to 88 s), had `MemAvailable` of 16.1 to 17.2 GiB before every SynthSeg call without downward drift, left only the streams and kan workers resident at each call and left no worker behind. That is **about 4.4 h for 198 scans** (cold bound 5.7 h). These timings and memory figures were not measured on OASIS scans. The full run is strictly sequential and requires that **no other memory-heavy process runs on the machine** for its duration.

## 3. What can and cannot be tested
- **Clinical stream: untestable.** The ADNI clinical features do not exist in OASIS, and CDR is the label, so using it as a feature would be label leakage. MMSE and demographics are not model inputs. Age and sex are the only clinical inputs used (by the MRI specialist and for brain-age gap).
- **No 4-stage output.** MRI_ONLY mode yields no stage, severity or flags, so CN/EMCI/LMCI/AD staging cannot be tested. EMCI and LMCI do not exist in OASIS. OASIS documents CDR 0.5 as "very mild dementia", a clinical diagnosis of probable AD (section 3a); other cohorts may call the same clinical stage MCI. This protocol uses the OASIS wording and the neutral class name "CDR0.5".
- **Testable:** ALZ (the CNN's AD-vs-rest score), brain-age gap (BAG), and the MRI specialist's P(AD) (AD-vs-rest, from a 4-class model) on SynthSeg volumes.
- Labels are CDR-based, not ADNI diagnostic criteria. The label definitions differ.
- Acquisition, defacing, labels, timing and left/right: see the documented facts (section 3a) and the domain-shift sources (section 3b). ADNI training scans are ADNI2/GO, mostly 3T, GradWarp + N3m or N3m.
- Probabilities are called "model probability". Calibration is neither claimed nor tested.

## 3a. Documented facts about OASIS-1
Sources: Marcus et al. 2007 (the paper; page numbers are the journal pages 1498-1507) and the OASIS fact sheet rev. 2007-8-20 (the fact sheet). Both were supplied by the project owner and read in full. The last column says how each fact is checked against the downloaded files; counts are aggregate and generated at Stage A (section 4).

| Item | Documented | Source | Check against the downloaded files |
|---|---|---|---|
| Scanner | 1.5-T Siemens Vision (Erlangen) | paper p.1500, Image Acquisition | Not in any file; the documentation is the only source. Resolves open item O2. |
| Sequence | MP-RAGE; TR 9.7 ms, TE 4.0 ms, flip 10 degrees, TI 20 ms, TD 200 ms; sagittal; 1.25 mm thickness, no gap; 128 slices; 256 x 256 in-plane (1 x 1 mm) | paper p.1500, Table 1; fact sheet Table 1 (Sag, 256x256x128, 1x1x1.25) | Per-session text file: TR, TE, TI, flip, orientation, voxel size, field of view. Raw headers: dimensions and voxel size. TD is documentation only. Disagreeing session counts are reported, nothing else. |
| Scans per session | 3 or 4 T1 scans in a single session | paper p.1500; fact sheet Summary | Count of raw scans per session (generated). |
| Image format | 16-bit big-endian Analyze 7.5, converted from Siemens IMA by a custom program | fact sheet "Image data"; paper p.1501, Postprocessing | Raw headers: endianness, data type, bits per pixel (generated). |
| Defacing | All released images were defaced. The method "first masks out all nonbrain voxels in the image, dilates and smoothes the mask in an iterative fashion, and finally applies the enlarged mask to the original image to safely remove voxels outside the cranial vault". In 12 cases an atlas-based method (Morphometry BIRN) was used instead. eTIV and nWBV changed by 0.024% and 0.059% on average. The fact sheet attributes the face removal to the Brain Extraction Tool at the fMRIDC; the paper names "software provided by the fMRI Data Center" | paper p.1501 (Postprocessing) and p.1502 (Anatomical Characteristics); fact sheet Summary | Not checked from images: no OASIS image is viewed. The two documents name the tool differently. |
| Dementia labels | CDR 0 no dementia; 0.5 very mild; 1 mild; 2 moderate. All participants with CDR > 0 were diagnosed with probable AD. The diagnosis is clinical ("based solely on clinical methods", standard criteria, McKhann et al. 1984). The paper states this clinical diagnosis "is confirmed by histopathological examination in 93% of the individuals (Berg et al., 1998), even for those in the earliest symptomatic stage (CDR 0.5) of AD who elsewhere may be considered to represent mild cognitive impairment" | paper p.1499-1500, Clinical Assessment; fact sheet "Additional data" | The 93% figure is a literature value for the parent research-centre cohort, cited from Berg et al. 1998. It was not measured in this release, and no neuropathology is in the download. |
| Subject exclusions in the release | Primary cause of dementia other than AD, active neurological or psychiatric illness, serious head injury, clinically meaningful stroke, psychoactive drugs, and gross anatomical abnormality were excluded; age-typical changes were accepted | paper p.1499 | Not checkable. |
| Image-quality screening | Each image rated 1 to 3; rating 3 excluded. 10 individual scans and 15 complete sessions were excluded from the release; borderline images remain | paper p.1501, Quality Control | Not checkable. Selection effect: the release is cleaner than an unscreened cohort. |
| Clinical-to-scan interval | MRI "typically ... obtained within a year of a subject's clinical assessment (mean = 105 days, range = 0-314 days)". Three AD subjects were scanned after 635, 449 and 443 days | paper p.1499, Subjects | Not analysable: the downloaded files contain no clinical assessment date. Stated as a limitation (section 3b). |
| Left/right convention | Not documented beyond "Sagittal" (paper Table 1, fact sheet Table 1). The paper says a vitamin E capsule was placed over the left forehead as a reference marker of anatomical side (p.1500); how the files encode left and right, and whether the marker survives defacing, is not documented | paper p.1500; fact sheet Table 1 | **The left/right convention is not documented in the provided materials.** No image is inspected for the marker. The primary convention is a third-party documented one (section 4a, addendum A1). |
| Age | Integer years in the per-session text file; cohort range 18 to 96 | fact sheet Summary; per-session text | Integer years only, so age has a rounding uncertainty of about half a year, which enters brain-age gap and the MRI specialist. |

**Older subjects, Table 3 of the paper (p.1503): age 60 or older.**

| | CDR 0 | CDR 0.5 | CDR 1 | CDR 2 |
|---|---|---|---|---|
| Number | 98 | 70 | 28 | 2 |
| Sex (female/male) | 72/26 | 39/31 | 19/9 | 1/1 |
| Age, mean +/- SD (range) | 75.9 +/- 9.0 (60-94) | 76.4 +/- 7.0 (63-92) | 77.2 +/- 7.5 (62-96) | 82 +/- 5.7 (78-86) |
| MMSE, mean +/- SD (range) | 29.0 +/- 1.2 (25-30) | 25.6 +/- 3.5 (14-30) | 21.7 +/- 3.8 (15-29) | 15.0 +/- 0.0 (15) |

The paper's footnote: "Full clinical data were not available for 23 subjects". The generated comparison of our recount with this table is in section 4. In short: our recount agrees with Table 3 in n, sex and MMSE for every group and in age for CDR 0 and CDR 2, but the printed age statistics for CDR 0.5 and CDR 1 differ, while the recount agrees with the paper's Table 2 in all eight decade-by-dementia cells (counts, CDR split and mean age to two decimals). The cause is not resolved. **The analysis uses the age in the per-session text file** (integer years), which reproduces Table 2.

## 3b. Domain-shift sources and limitations (stated before any result)
- **Field strength.** OASIS-1 is 1.5 T (section 3a). The ADNI scans the frozen system was trained on are ADNI2/GO, predominantly 3 T (an earlier project audit found 802 of 811 training subjects in the 3 T metadata table; the row-level match to the scan visit was not done). Contrast, noise and bias-field differ between field strengths. This is a source of domain shift that is tested, not corrected.
- **Defacing.** Every OASIS image has voxels outside the cranial vault set to zero (section 3a). m1 registers the whole head, including skull and scalp, to a whole-head template before registering brain to brain, so a defaced input can change the first registration stage and therefore corr_vs_mni and the affine. The effect has not been measured. It is reported through the Stage D domain-shift diagnostics, and m1 is not changed.
- **Acquisition geometry.** 128 sagittal slices of 1.25 mm with 1 x 1 mm in-plane resolution. m1 resamples every input to 1.0 x 1.0 x 1.2 mm in its own working space.
- **Image-quality screening.** The release excludes images the providers rated 3 (section 3a); the ADNI inputs were screened differently.
- **Clinical-to-scan interval.** The label (CDR) belongs to a clinical assessment that preceded the scan by a mean of 105 days (range 0 to 314, three AD subjects more than a year). This is a limitation of the label, not analysable from our files; its direction is not asserted.
- **Left/right.** Not documented by OASIS in the materials supplied (section 3a). A mirrored brain registers almost as well as the correct one, so registration quality cannot reveal a wrong convention. Stage B0 could not decide it from the data (addendum A1, Part 2). The primary convention is the third-party documented conversion described in section 4a; if its left-right sign is wrong the CNN input is mirrored for the whole cohort. Sensitivity S6f (section 7) quantifies this for a defined subset.
- **Age.** Integer years, and the cohort reaches 96 while the ADNI training ages are 54.9 to 90.3.
- **Labels.** CDR, a different instrument from the ADNI diagnostic categories.

## 4. Cohort
- Source: the 12 downloaded OASIS-1 archives (discs 1-12), read from the data area outside this repository.
- **Included:** the MR1 session of each subject, age 60 or older at scan, with a recorded CDR (0, 0.5, 1 or 2). MR2 retest sessions are excluded (one session per subject).
- Counts by CDR, exclusions, the comparison with Table 3 of the paper and the S4 age-tertile edges are generated at Stage A (block below); QC verdict counts are generated later. The CDR >= 1 group is small and confidence intervals will be wide.

<!-- STAGE_A:BEGIN (generated by src/validation/oasis_manifest.py from docs/validation/stage_a_counts.json) -->

Stage A (no inference). Archives: 12; sessions 436; subjects 416; retest sessions 20 (largest age 34, so all are excluded by the age rule; the paper describes the retest subjects as being in their twenties).

Exclusions applied in this order: not first session 20; age missing 0; age under 60 218; cdr missing 0; sex missing 0; no raw scan 0; cohort 198 of 436 sessions. Sessions whose first raw scan is not `mpr-1`: 0.

Cohort n = 198; by CDR 0 / 0.5 / 1 / 2 = 98 / 70 / 28 / 2. First raw scan is `mpr-1` in 198 of 198. MMSE missing: 0; education or SES missing: 18.

| CDR | n (ours / paper) | female/male (ours / paper) | age mean +/- SD, range (ours) | age (paper) | MMSE mean +/- SD, range (ours) | MMSE (paper) | agree |
|---|---|---|---|---|---|---|---|
| 0 | 98 / 98 | 72/26 / 72/26 | 75.9 +/- 9 (60-94) | 75.9 +/- 9 (60-94) | 29 +/- 1.2 (25-30) | 29 +/- 1.2 (25-30) | n yes, sex yes, age yes, MMSE yes |
| 0.5 | 70 / 70 | 39/31 / 39/31 | 76.2 +/- 7.2 (62-92) | 76.4 +/- 7 (63-92) | 25.6 +/- 3.5 (14-30) | 25.6 +/- 3.5 (14-30) | n yes, sex yes, age NO, MMSE yes |
| 1 | 28 / 28 | 19/9 / 19/9 | 77.8 +/- 7 (65-96) | 77.2 +/- 7.5 (62-96) | 21.7 +/- 3.8 (15-29) | 21.7 +/- 3.8 (15-29) | n yes, sex yes, age NO, MMSE yes |
| 2 | 2 / 2 | 1/1 / 1/1 | 82 +/- 5.7 (78-86) | 82 +/- 5.7 (78-86) | 15 +/- 0 (15-15) | 15 +/- 0 (15-15) | n yes, sex yes, age yes, MMSE yes |

Check against Table 2 of the paper (decade by dementia status; counts, CDR 0.5/1/2 split and mean age to two decimals): 60s agrees; 70s agrees; 80s agrees; 90+ agrees.

Where Table 3's printed age statistics differ from the recount (the rows marked NO above), the recount agrees with Table 2 and Table 3's MMSE statistics; the cause is not resolved here. The analysis uses the age in the per-session text file.


Disagreement with the documented acquisition (sessions, over all sessions in the archives): per-session text none. Raw headers: 1688 raw scans; sessions with a disagreeing scan 0. Sessions by number of raw scans: 3 scans: 59, 4 scans: 375, 5 scans: 1, 6 scans: 1; sessions outside the documented 3 or 4 scans: 2.

S4 age-tertile edges (fixed here from the CDR 0 cohort members, n = 98): e1 = 71, e2 = 80; groups are age <= e1, e1 < age <= e2, age > e2 (sizes 35 / 31 / 32).

Manifest sha256 as written at Stage A, when the `orientation` column was still `UNSET` (the pinned `inputs.csv` hash after addendum A1 is in section 2): inputs.csv 1dc4f56c4b380884eefc8f5662f63bede9fe45da74185a3ad908be6fd8ddb874; labels.csv b78d1258d72e30e9d112ad845ae6b1e422bc4ae0990eaadac247a5d860252196.

Gate A: PASS sessions in the archives = 436; PASS distinct subjects = 416; PASS retest (MR2) sessions = 20; PASS every MR2 session is under age 60 (excluded by the age rule); PASS cohort by CDR = 98/70/28/2; PASS every cohort case has a raw scan, age and sex; PASS every session that meets the age and CDR rule has a raw scan and a sex; PASS no duplicate case ids; PASS no duplicate sessions in the cohort; PASS every session text file has an index entry; PASS second code path (awk) reproduces n, age mean/SD, MMSE mean/n, sex by CDR group and the session counts.

<!-- STAGE_A:END -->

- Scan: **the first raw scan (`mpr-1`) of the session** only. Analyze 7.5 files are converted by the explicit-orientation loader. The registered, masked and atlas-space images (`PROCESSED/`) and the tissue segmentations (`FSL_SEG/`) are never model input; the loader refuses those paths.
- Orientation convention: the primary convention `oasis1_primary` is third-party documented, not OASIS-documented; the mirrored convention is the sensitivity S6f. Section 4a (merged from addendum A1) gives the decision, its provenance and what is unverified.
- QC policy: the primary analysis keeps engine registration verdicts PASS and WARN, and excludes FAIL (corr_vs_mni below 0.15, MRI withheld by the engine) and engine hard failures. Counts of PASS, WARN, FAIL and failures are reported by CDR group. A sensitivity analysis includes everything the engine could score, and a worst-case bound counts every failure as an uninformative score (tied at the median).

## 4a. Orientation convention (addendum A1, final text 2026-11-14, merged into this body on 2026-11-14)
Provenance: Stage B0 was run on 2026-11-14 under rules committed before any B0 computation (`1008ede`); its generated results and the pre-registered rules are kept unchanged in `docs/external_validation_addendum_A1.md`, which is the dated record. This section is that addendum's decision and outcome.

**Stage B0 outcome (recorded as it was).** Rule outcome: stop, no convention proposed. E1 (reader behaviour on synthetic Analyze files) was consistent but cannot say which convention is right. E-geo (the raw-to-atlas transform and the XML/TXT of 20 sessions) found a positive determinant everywhere and no statement of orientation, so it gave no absolute convention. E3 (template chirality) was uninformative: the template m1 loads is the symmetric ICBM152 2009a variant (the project plan had assumed an asymmetric one), the pre-registered numeric check (reflection correlation 0.9827 against a limit of 0.99) wrongly classified it as asymmetric, and the ADNI positive control scored 4 of 20 where 18 were needed, which is below chance. No OASIS scan was processed by m1 in B0.

**Primary convention: third-party documented, not OASIS-documented.** The loader convention `oasis1_primary` maps voxel (i, j, k) to RAS with the sform rows [0 0 -1.25 0], [1 0 0 0], [0 1 0 0], [0 0 0 1]: i = posterior to anterior, j = inferior to superior, k = right to left (axis codes A, S, L), zero translation. Source: A. M. Winkler, "Converting OASIS brains to NIFTI", brainder.org, 2011-08-13 (https://brainder.org/2011/08/13/converting-oasis-brains-to-nifti/), step 4 of a recipe for converting the raw OASIS-1 Analyze files. OASIS did not state this convention in any material supplied to the project. It was adopted on the project owner's instruction on 2026-11-14, after B0 had stopped. It is **not derived from the B0 evidence**: none of E1, E-geo or E3 supports or contradicts it.
- The recipe page gives commands, not a prose description of the axes, and does not say how its author verified the left-right sign. That is unverified. The page was read through a web-fetch tool.
- **Analyze 7.5 specification check: unverified.** The ANALYZE 7.5 document (Mayo Clinic Biomedical Imaging Resource, 1986-1995; text extracted from the copy at https://eeg.sourceforge.net/ANALYZE75.pdf) defines `orient` as "slice orientation for this dataset" with codes 0 to 5 (2 = sagittal unflipped), says the field "determines whether the Movie program will attempt to flip the images before displaying a movie sequence", and contains none of the words left, right, anterior, posterior, superior, inferior or radiological. It neither confirms nor contradicts the recipe.
- **Withdrawn:** the earlier statement that "`orient` carries no left-right sign" is withdrawn as an established fact. What is established is narrower: the specification text read gives the slice plane and a display flip and states no axis direction, and OASIS's own materials supplied to the project state no left-right convention.
- The in-plane axes (i = posterior to anterior, j = inferior to superior) come from the same recipe and are untested; the Stage B registration verdict rule (section 10) is the first check.

**Mirrored convention.** `oasis1_mirrored` (the k column reversed: k = left to right, axis codes A, S, R) is the pre-declared sensitivity S6f (section 7). It never replaces a primary result.

**Replacing the primary convention.** An authoritative statement from OASIS, if one is obtained, may replace the primary convention only by a dated addendum made **before Stage C** starts. After Stage C starts the convention is fixed and a later statement is reported as a limitation.

**Implementation and manifest (dated note, 2026-11-14).** The loader (`src/assessment/readers/analyze75.py`) takes the exact sform rows above, requires header orient code 2 and voxel size exactly 1 x 1 x 1.25 mm, and still needs an explicit `orientation` argument (no default). The `orientation` column of `inputs.csv` was set from `UNSET` to `oasis1_primary` in all 198 rows; no other column changed (hashes in section 2).

## 5. Labels
Plugin `cdr_3class`: CDR 0 is "CN-like", CDR 0.5 is "CDR0.5" (OASIS: very mild dementia, a clinical diagnosis of probable AD; other cohorts may call it MCI), CDR 1 or higher is "dementia". The binary H1 contrast is CDR 0 against CDR 1 or higher. Labels are held in a separate file that the inference runner never opens; the analysis step opens it only after the run is complete.

## 6. Volume source
Primary analyses (H1, S1-S4) use ALZ and BAG, which do not depend on volumes. The MRI specialist (S5) uses **SynthSeg 7.4.1 volumes computed from the native scan**. These are **not** the FreeSurfer 5.1 volumes on which the specialist was trained. The six ratios have known weak agreement with FreeSurfer 5.1 (ADNI, n = 25, ICC(2,1)): Ventricles 0.89, Hippocampus 0.85, WholeBrain 0.52, MidTemp 0.33, Fusiform 0.28, Entorhinal 0.15, with biases of +21.5%, +8.8%, +1.0%, -11.9%, +2.6% and +31.1% (`analysis/volume_agreement/results/agreement.md`). All six frozen ratios are used, because the frozen model cannot drop a feature without a refit. Every specialist result states its volume source. OASIS-provided FreeSurfer volumes, if ever obtained, enter only as a later optional addendum (S6b), with the FreeSurfer version stated and the statement that it is not the ADNI FS 5.1 pipeline; equivalence is not claimed.

## 7. Hypotheses, metrics and decision thresholds
One primary hypothesis. Everything else is secondary or exploratory, and no multiplicity correction is applied (stated here).

**H1 (primary).** ALZ discriminates CDR 0 from CDR >= 1 (a higher ALZ means more impaired).
- Metric: AUC (ties count half) with a subject-level percentile bootstrap 95% CI, 2000 resamples, bootstrap seed 20261113. Positives and negatives are resampled separately. The AUC and the CI limits are compared with the thresholds unrounded, and the CI lower bound is compared strictly (a lower bound of exactly 0.65 is not above 0.65).
- **PASS:** AUC >= 0.75 and CI lower bound > 0.65.
- **PARTIAL:** 0.65 <= AUC < 0.75, or AUC >= 0.75 with CI lower bound <= 0.65.
- **FAIL:** AUC < 0.65, which includes AUC < 0.5 ("inverted").
- Threshold provenance: the limits 0.75 and 0.65 were fixed before any OASIS output exists and are not derived from OASIS. The ADNI train-only references below were computed afterwards, for context only. The limits were not adjusted after seeing them.
- With about 30 positive cases the CI half-width is expected to be about 0.10.

**ADNI reference values (generated, train split only).**

<!-- ADNI_REFERENCE:BEGIN (generated by analysis/external_validation/adni_reference.py) -->

ADNI train-split reference values (n = ALZ 567, BAG 566, specialist 567; out-of-fold or cross-fitted predictions; the frozen test split is not read). AUC with 2000-resample subject-level percentile bootstrap 95% CI, seed 20261113.

| Quantity | CN vs AD AUC [95% CI] | AD vs rest AUC [95% CI] |
|---|---|---|
| ALZ (cross-fitted) | 0.805 [0.744, 0.860] (n 104 / 133) | 0.764 [0.712, 0.814] (n 104 / 463) |
| Brain-age gap (corrected) | 0.795 [0.737, 0.850] (n 104 / 133) | 0.710 [0.655, 0.764] (n 104 / 462) |
| MRI specialist P(AD) (OOF, mean of 5 seeds) | 0.961 [0.936, 0.980] (n 104 / 133) | 0.880 [0.847, 0.909] (n 104 / 463) |

| Group | ALZ mean (SD) | BAG mean (SD), years | Specialist P(AD) mean (SD) |
|---|---|---|---|
| CN (n 133) | 0.379 (0.172) | +0.00 (5.47) | 0.066 (0.081) |
| EMCI (n 216) | 0.386 (0.191) | +2.20 (5.10) | 0.112 (0.139) |
| LMCI (n 114) | 0.490 (0.220) | +4.02 (5.13) | 0.247 (0.224) |
| AD (n 104) | 0.617 (0.209) | +6.80 (6.26) | 0.430 (0.215) |

BAG bias fit (CN train, n 133): predicted age = 0.582 x age + 29.97; ADNI train age at scan 54.9 to 90.3 years.

<!-- ADNI_REFERENCE:END -->

The frozen ADNI test-set value for the MRI specialist, 0.873, is an AD-vs-rest AUC (CN, EMCI and LMCI as negatives; `analysis/fusion_v3/frozen_run/test_results.md`). It is not a CN-vs-AD value and not an ALZ value.

**Secondary analyses** (reported with CIs; direction and descriptive statements only unless stated). All are computed by `src/validation/analyze_oasis.py` (see "Analysis implementation" below). Unless stated, they use the primary-analysis rows (registration verdict PASS or WARN, a score present).
- **S1** ALZ, CDR 0 against CDR 0.5: AUC with CI, no pass/fail beyond direction (AUC above 0.5).
- **S2** Trend of ALZ and of BAG across CDR 0, 0.5 and >= 1: one-sided Jonckheere-Terpstra (alpha 0.05) and Spearman against ordinal CDR. "Direction matches ADNI" means a monotone increase with severity (see the reference table): the group medians do not decrease with severity and Spearman rho is positive. PASS if the direction matches and the one-sided p < 0.05. OASIS BAG is not centred (the corrected gap is zero-mean only in ADNI CN), so only rank statistics are used and the CDR 0 offset is reported descriptively.
- **S3** Spearman correlation of ALZ and of BAG with MMSE, with bootstrap CI. PASS if rho <= -0.30 for ALZ (inclusive); BAG is descriptive.
- **S4** Age-adjusted versions of S2 and S3 (partial Spearman, rank-based, controlling for age, with bootstrap CI); AUC of ALZ for CDR 0 against CDR >= 1 within the age tertiles with edges fixed at Stage A (age <= e1, e1 < age <= e2, age > e2, the bins closed on the right); and two descriptive benchmarks for H1 computed from demographics and labels only, with no model output and no QC filter, on all CDR 0 and CDR >= 1 cases: an age + sex logistic model scored by repeated stratified 5-fold cross-validation, 20 repeats, L2 penalty 1.0 (standardised features; the same objective as scikit-learn with C = 1), reported as the AUC of the repeat-averaged out-of-fold predictions with a bootstrap CI, so that it is not optimistic; and the AUC of nWBV alone (lower nWBV scoring as more impaired, a sign fixed in advance).
- **S5** MRI specialist P(AD) on SynthSeg volumes (volume source stated) for CDR 0 against CDR >= 1: AUC with CI. Reported as a shift relative to the ADNI references above, with the version-mismatch statement. No pass/fail.
- **S6** Sensitivities (the letters are stable references; S6a, session-level averaging of repeated scans, is not part of this protocol): (b) OASIS-provided FreeSurfer volumes instead of SynthSeg, only if obtained, as a later optional addendum; (c) include WARN, FAIL and failed cases: every case with a score, plus a worst-case bound in which failed and unscored cases are tied at the median of the kept scores; (d) sex-stratified; (e) age restricted to 60-90.3 (the ADNI training range, both ends included); (f) **S6f, mirrored left-right convention** (pre-declared): all cases with CDR 1 or higher (30 cases) plus the first 30 CDR 0 cases in sorted case-id order, converted with `oasis1_mirrored` (k reversed) and run through the same engine path after the main run has finished, as a labelled sensitivity. Reported descriptively for these 60 cases: the AUC for CDR 0 against CDR >= 1 under each convention with a bootstrap CI (seed 20261113, 2000 resamples) and the paired per-case difference in ALZ (median and interquartile range). No pass/fail, and it never replaces or modifies the primary result; it shows how far the result depends on the third-party left-right convention. The S6f manifest is `inputs.csv` with only the `orientation` column changed to `oasis1_mirrored` and restricted to the 60 cases; its sha256 is recorded in a dated addendum before the S6f run starts.
- **Descriptive only:** distributions of ALZ, BAG and P(AD) against ADNI; domain-shift diagnostics (intensity statistics, corr_vs_mni and QC counts, brain-mask volume in mm3, acquisition parameters). No calibration is claimed.
- Recalibration, threshold shifts or any re-fitting appear only as a clearly labelled secondary analysis and never replace a primary result.

**Analysis implementation.** `src/validation/analyze_oasis.py`, tested on synthetic data only, defines every threshold once (`src/validation/stats.py`); a test parses this section and asserts that they match. The script reads `labels.csv` only after a `RUN_COMPLETE` marker of a full run exists, the run's pinned `inputs.csv` hash equals the file on disk and every case has a finished status; otherwise it refuses before opening the labels. It reads scores through the score view of the report reader and never reads the specialist's argmax-stage field. Its output is an aggregate JSON written to `docs/validation/` by a writer that refuses case ids, OASIS identifiers and per-subject vectors.

## 8. Interpretation of outcomes
- **PASS:** the MRI CNN score separates CDR 0 from dementia in an external cohort at the stated level. The model card, paper and slides state this with the CI, the small positive group, and the label and scanner differences. It does not validate staging, the clinical stream, calibration or severity.
- **PARTIAL:** partial generalisation, reported with the CI and without softening. Limiting factors (age confound, QC exclusions, domain shift) are examined only through the pre-declared secondaries.
- **FAIL:** reported as a limitation of generalisation in the model card, paper and slides. The primary result is not replaced by a secondary or recalibrated analysis. Domain-shift diagnostics are reported to inform the cause but are not used to reinterpret H1.

## 9. Exclusions
MR2 retest sessions; age under 60; CDR missing; sessions without a raw scan; engine hard failures; QC FAIL (primary analysis only). All are counted and reported by CDR group.

## 10. Staged-run protocol
Stage A manifest (no inference; counts reconcile) -> B0 orientation and geometry (m1 and QC numbers only, no model) -> D analysis script tested on synthetic data -> addendum A1 moved into this body -> local tag `oasis-protocol-v1` -> B five-subject plumbing dry run (first five sorted case ids; scores suppressed; outputs deleted; QC-plumbing rule below) -> C full run, one shot, runner blind to labels -> S6f mirrored sensitivity run (after the main run, labelled) -> analysis, aggregate only -> E reporting. A failed gate stops the run and is recorded as an addendum.

**Stage B QC-plumbing rule.** In the five dry-run cases the registration verdict must be PASS or WARN in every case. The verdict counts are plumbing information and are read through the plumbing view; no score value is read or printed. A FAIL, or a case that fails to run, stops the run before Stage C and is recorded as an addendum (a systematic registration failure would also refute the in-plane axis assumption of section 4a).

## 11. Reporting commitments
All pre-specified analyses are reported whether or not they are favourable. All numbers are generated by the committed analysis script, none typed by hand. Per-subject values never enter the repository. Deviations appear only as dated addenda. The OASIS acknowledgement and DOI accompany every document that contains OASIS results.
- **Model card correction (Stage E):** the "MRI specialist (MRI-only mode)" row in `docs/model_card.md` reports 4-class accuracy (0.504), which does not apply to an MRI_ONLY report that has no stage. It is corrected as a dated correction in Stage E, not silently.
- **Separate issue, recorded and not changed:** in MRI_ONLY mode the engine reports the specialist's argmax stage in `drivers.mri_target` (`src/assessment/report_builder.py`). This validation never reads that field, and engine behaviour is not changed for it here.

## 12. Data-use compliance
Non-commercial academic use only. No sharing of OASIS data or derivatives. No OASIS images, derived volumes or subject-level rows in git, enforced by `tools/oasis_guard.py`, a versioned pre-commit hook and a CI workflow. Aggregate metrics only. Figures use skull-stripped images only: no raw-head slices and no 3D head renders. All derivatives live in a data area outside every repository and are securely deleted at project end, with a cleanup log.

## 13. Open items (to be resolved from primary sources, none assumed)
- **O1 DeepBrainNet training data. Status: unresolved; brain-age gap independence from OASIS is not established (checked 2026-11-14; nothing inferred).** Sources read through a web-fetch tool, not directly by the analyst: Bashyam et al., Brain 2020;143(7):2312-2324 (PMC7364766 and the Oxford Academic page; abstract, introduction, methods, results, Table 1 and discussion verbatim, reference list not read), its supplementary PDF (15 pages, 8,235 characters of text extracted locally), and the 2021 corrigendum (it corrects an author-name spelling only).
  - The model's training set is "normal control subjects (n = 11 729) covering individuals of ages 3 to 95", called "LifespanCN"; the introduction says it is "derived from 18 studies". The only held-out set named is SHIP (n = 2739).
  - Table 1 (columns: Study, n (male/female), n Controls, n Disease, Mean age, Age range, Experiments) marks these 12 rows "LifespanCN": ADC 76, AIBL 446, BLSA 1.5 T 90, BLSA 3 T 952, CARDIA 719, PAC-WASH 247, PAC-WISC 127, PAC-JHU 95, PING 398, PNC 1396, PENN PMC 41, SHIP 2739. UK BioBank (4402 scans, 4403 controls) is marked "Residual Analysis"; ADNI-1 (747) and ADNI-2 (952) and PHENOM (835) are marked "Residual analysis and transfer learning". Supplement S.1 ("LifespanCN racial demographics") lists 11 datasets as "the LifespanCN training set": ADC, AIBL, BLSA-1.5T, BLSA-3T, CARDIA, PAC-JHU, PAC-WASH, PAC-WISC, PING, PNC, SHIP. Supplement S.8 says "The development set contains 2 studies from LifespanCN ... The aging set contains 12 studies from LifespanCN".
  - **OASIS is not named among the training data in any text read (main text, Table 1, supplement). ADNI is not named among the LifespanCN training data either; ADNI-1 and ADNI-2 are named for residual analysis and transfer learning.**
  - **Why this stays unresolved:** the paper's own study counts do not agree (18 studies in the introduction; 2 + 12 studies in the supplement; 12 rows in Table 1; 11 datasets in S.1), and the LifespanCN rows of Table 1 add up to 7,326 scans (summed by the analyst; the paper does not state this total) against the stated 11,729. (7,326 + the 4,403 UK Biobank controls = 11,729, but Table 1 does not mark them as training data; the paper also states 14,468 = 11,729 + 2,739, which counts SHIP again.) So the list of studies behind the 11,729 scans is not fully given, and the absence of OASIS cannot be concluded from what was read.
  - Consequence: brain-age gap results are reported with the statement "OASIS is not named among the training cohorts; the training list is incomplete; independence of the brain-age model from OASIS is not established". The DeepBrainNet release README (n = 11,729 scans from "different studies") gives no cohort list.
- **O2 Field strength** of OASIS-1 acquisition: **resolved 2026-11-14** from the paper (p.1500): 1.5-T Siemens Vision (section 3a).
- **O3 Orientation convention:** Stage B0 stopped without a convention (E3 uninformative, E-geo none; `docs/external_validation_addendum_A1.md`, Part 2). **Resolved by owner decision on 2026-11-14:** the primary convention is the third-party documented conversion in section 4a (labelled as such; not OASIS-documented, not confirmed by the Analyze 7.5 specification text), the mirrored convention is sensitivity S6f, and an authoritative OASIS statement may replace the primary only by a dated addendum before Stage C.
- **O4 OASIS FreeSurfer version**, only if the optional volumes addendum is ever used.

## Addenda
- **2026-11-14.** The test that enforced "no tag yet" expired when the tag was created and was replaced by a frozen-body check (tests/ only; no change under `src/` or to the body above the Addenda heading). The Stage C run commit is therefore the commit that follows tag `oasis-protocol-v1`; the run header pins it. Gates were re-run at that commit.
- **2026-11-16 (S6f manifest, pre-registration).** The S6f subset was selected from the labels after the main run had completed and before any score was read, computed or printed, by `tools/make_s6f_manifest.py` (commit `e59ef81`), which reads only the case-id and CDR columns of `labels.csv`: all 30 cases with CDR 1 or higher plus the first 30 CDR 0 cases in sorted case-id order, 60 cases. The S6f manifest is `inputs.csv` with only the `orientation` column set to `oasis1_mirrored`, restricted to those 60 of the 198 pinned rows; its sha256 is `2bc286cc5acc00263cf2c865b4f306871048ca1eb30a6f985bb160ffd22dbc26` (data area folder `s6f_manifest`, no labels file). The S6f run uses its own run root, `stageC_s6f`, in the data area. S6f is a labelled sensitivity and never replaces or modifies the primary result.
- **2026-11-16 (results).** The analysis was run once with `src/validation/analyze_oasis.py` at commit `16d309f` (the analysis script and the protocol body were not changed after the tag) on the main run (run commit `5c6a6ae`, `inputs.csv` hash as pinned in section 2) and the S6f run (run root `stageC_s6f`, run commit `16d309f`, manifest hash as recorded above). The script assigned the H1 outcome class **PASS**. Result files, aggregate only: `docs/validation/oasis_analysis.json` sha256 `a96f564548c391f239dc617602847b6a6d33cbbd469c940496a2559439b894aa`; `docs/validation/oasis_domain_shift.json` sha256 `b4241f907849643c74564950f895e21b48fde146f93362609ab5e0514bfcbd12`. Deviations: the domain-shift diagnostics were produced by a helper that was not pre-registered (committed afterwards as `tools/oasis_domain_shift.py`, commit `f2a5ddc`); its first output had an empty acquisition section because of a parsing bug in the helper (it matched scan number 1 instead of mpr-1), and it was regenerated once with no change to the features, the QC counts or the analysis file; regenerating it from the committed tool reproduced the regenerated file byte for byte. The analysis script was run exactly once. The numbers are in the generated files above and in `docs/validation/oasis_results_tables.md`; none is typed here.
