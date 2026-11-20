# Addendum A1: left-right convention for the OASIS-1 Analyze scans (final text)

**Status: FINAL TEXT, dated 2026-11-14. Not tagged. It is moved into the protocol body before the local tag `oasis-protocol-v1`; an authoritative statement from OASIS, if one is obtained, may replace the primary convention only by a further dated addendum made before Stage C (Part 0.5).**
The document keeps the history in order: Part 0 is the decision, Part 1 the rules pre-registered before any Stage B0 computation (unchanged except for one dated note), Part 2 the generated Stage B0 results as they were, Part 3 how B0 ended and how it was resolved.

Data were provided by OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, R. Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382.
Citation: Marcus et al., J Cogn Neurosci 2007;19(9):1498-1507, https://doi.org/10.1162/jocn.2007.19.9.1498

## Part 0. Decision

### 0.1 Primary convention: third-party documented, not OASIS-documented
The raw OASIS-1 files are converted with the loader convention `oasis1_primary`: voxel (i, j, k) maps to RAS with the sform rows `[0 0 -1.25 0]`, `[1 0 0 0]`, `[0 1 0 0]`, `[0 0 0 1]`, that is i = posterior to anterior, j = inferior to superior, k = right to left (axis codes A, S, L), with zero translation.

Source: A. M. Winkler, "Converting OASIS brains to NIFTI", brainder.org, 2011-08-13, https://brainder.org/2011/08/13/converting-oasis-brains-to-nifti/ (a third-party recipe, step 4: `fslorient -setsform 0 0 -1.25 0  1 0 0 0  0 1 0 0  0 0 0 1`). **This is a third-party documented convention. OASIS did not state it in any material supplied to this project.** It was adopted on the project owner's instruction on 2026-11-14, after Stage B0 had stopped without a convention (Part 3).

What the recipe page does and does not say (the page was read through a web-fetch tool, not directly; the commands below were relayed verbatim):
- It gives the commands, not a prose description of the axes. The later steps are consistent with the sform as I read them: `fslswapdim ... RL PA IS` reorders the axes to x = right to left (old k), y = posterior to anterior (old i), z = inferior to superior (old j), and `fslorient -setqform -1.25 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1` compensates.
- It says Analyze 7.5 "suffers from not including orientation information" and that "some care has to be taken to ensure that there are no left-right flips". It says the vitamin E capsule on the left side (Marcus et al. 2007) is "very helpful to identify the correct side".
- **It does not say how the author verified the left-right sign of the sform.** That is unverified here.
- A 2020 comment on the page reports that some OASIS-3 files (a different data set) did not work with the script. Noted only.

### 0.2 In-plane axes
The in-plane directions (i = posterior to anterior, j = inferior to superior) come from the same recipe. They have not been tested. Stage B reports the registration QC verdict counts of its five scans; a systematic registration failure there would refute the in-plane assumption and stops the run.

### 0.3 Check (a): what the Analyze 7.5 specification says about `orient` 2
Source: the ANALYZE 7.5 file format document (Mayo Clinic Biomedical Imaging Resource, copyright 1986-1995), 11 pages, as hosted at https://eeg.sourceforge.net/ANALYZE75.pdf; the text was extracted locally from the PDF and searched.
- Verbatim: "The data_history substructure is not required, but the orient field is used to indicate individual slice orientation and determines whether the Movie program will attempt to flip the images before displaying a movie sequence." and "orient: slice orientation for this dataset.", followed by the list (one entry per line) "0 transverse unflipped", "1 coronal unflipped", "2 sagittal unflipped", "3 transverse flipped", "4 coronal flipped", "5 sagittal flipped". The sample program comments read "Movie flag OFF: 0 = transverse, 1 = coronal, 2 = sagittal; Movie flag ON: 3 = transverse, 4 = coronal, 5 = sagittal".
- The text contains none of the words left, right, anterior, posterior, superior, inferior or radiological.
- **Result: unverified.** The specification defines `orient` as the slice plane together with a display flip for the Movie program. It states no direction for any voxel axis and no left-right sense for code 2. It therefore neither confirms nor contradicts the recipe. A secondary source (the nibabel documentation for its Analyze reader) says Analyze images are conventionally stored fastest-changing to slowest-changing in R to L, P to A, I to S order, that its reader assumes R to L by default, and that it does not interpret the `orient` byte; that is a convention for the usual transverse layout, not a definition for code 2.

### 0.4 Statement withdrawn
The earlier statements in this project that "`orient` carries no left-right sign" (the B0 plan, the Stage B0 report and Part 1.1 below) are **withdrawn as an established fact**. What is established is narrower: the specification text read gives the slice plane and a display flip and states no axis direction (0.3), and OASIS's own materials supplied here state no left-right convention.

### 0.5 Mirrored convention: sensitivity S6f; replacing the primary
The loader convention `oasis1_mirrored` (the same with the sign of the k column reversed: k = left to right, axis codes A, S, R) is a pre-declared **sensitivity S6f**, defined in protocol section 7: all cases with CDR 1 or higher plus the first 30 CDR 0 cases by sorted case id, converted with the mirrored convention and run through the engine after the main run, reported as a labelled sensitivity. It never replaces a primary result.

If an authoritative statement from OASIS (the data owners) on the left-right convention of the raw files is obtained, it may replace the primary convention **only by a dated addendum made before Stage C** starts. After Stage C starts the convention is fixed, and any later statement is reported as a limitation.

### 0.6 Implementation and manifest (dated note, 2026-11-14)
- Loader: `src/assessment/readers/analyze75.py`, conventions `oasis1_primary` and `oasis1_mirrored` take the exact sform rows above (header voxel size required to be exactly 1 x 1 x 1.25 mm, header orient code 2); an explicit `orientation` argument is still required, with no default. Tests on synthetic files cover the exact affine, a marker voxel on each side, both byte orders, the refusal paths, and that nibabel and ITK read the written NIfTI identically. The legacy B0 names `radiological` and `neurological` are kept so that Stage B0 stays reproducible; `radiological` has the same rotation and scale as `oasis1_primary`.
- Manifest: the `orientation` column of `inputs.csv` was set from `UNSET` to `oasis1_primary` in all 198 rows; no other column changed. `inputs.csv` sha256 before (Stage A) `1dc4f56c4b380884eefc8f5662f63bede9fe45da74185a3ad908be6fd8ddb874`, after `7ede554d051a4442ff39dd912b06f4115ea651c0a008882f26c86d08011a30fd`. `labels.csv` is unchanged, `b78d1258d72e30e9d112ad845ae6b1e422bc4ae0990eaadac247a5d860252196`.

## Part 1. Rules pre-registered before any Stage B0 computation (committed in 1008ede; unchanged)

### 1.1 The question
The raw OASIS scans are Analyze 7.5 files whose header says "sagittal" (orient code 2) and gives no sign for the slice axis, which is the left-right axis. *[Dated note 2026-11-14: the claim that the header gives no sign is withdrawn as an established fact; see Part 0.3 and 0.4.]* The documentation supplied (paper, fact sheet) does not state a left-right convention (protocol section 3a). The loader (`src/assessment/readers/analyze75.py`) therefore requires an explicit convention, and there are two candidates that differ only in that sign:

- **radiological**: voxel axes point towards (A, S, L); the slice index increases towards the patient's left.
- **neurological**: voxel axes point towards (A, S, R); the slice index increases towards the patient's right.

The in-plane directions (A and S) are my reading of the Analyze specification and are not documented in the materials supplied. They are checked by the in-plane sanity rule (1.5). A mirrored brain registers almost as well as a correct one, so registration quality alone cannot decide left-right.

### 1.2 Samples
- **B0 sample**: the first 20 cohort sessions in sorted order of case id (MR1, age 60 or older, first raw scan `mpr-1`). Nothing about a scan other than its position in the sort is used to choose it.
- **ADNI control sample**: the first 20 ADNI subjects in ascending PTID order among subjects in the ADNI train split that have a stored m1 `mni` volume. The test split is not used.

### 1.3 Evidence line E1: reader behaviour on synthetic files (no OASIS data)
Build synthetic Analyze files at run time (256 x 256 x 128, big-endian int16, orient code 2, voxel size 1 x 1 x 1.25 mm) with a single marker voxel placed off-centre on all three axes. For each reader, report (a) the world direction in which each array axis increases and (b) the world coordinates of the marker voxel:

1. nibabel loading the `.hdr` directly;
2. ITK through `ants.image_read` loading the `.hdr` directly (the behaviour m1 would see without the loader);
3. the NIfTI written by the loader under each candidate convention, read by nibabel and by `ants.image_read`.

Then pass each ANTs-read image through the orientation step m1 uses (`reorient_image2` to "IAL", resample to 1.0 x 1.0 x 1.2 mm) and report the output-array position of the marker and which side of the output midline it lies on, per the world frame the reader assigned.

E1 states **which reader behaviour m1 relies on** (read from `src/mri/m1_preprocess.py`: `ants.image_read` on the file it is given, so ITK's reading of the header, followed by header-driven reorientation). E1 never votes for a convention. It is **consistent** if the loader's NIfTI, read by ITK, gives the same world directions as nibabel reads for the same file and the marker's side after the m1 orientation step is the mirror image between the two candidates. If it is not consistent, that is a pipeline defect and Stage B0 stops.

### 1.4 Evidence line E-geo: the t4 transform and the XML/TXT (B0 sample only)
For the B0 sample, read as text only the session XML and TXT, the `mpr-1` to `711-2C` transform file (`..._mpr-1_anon_to_711-2C_t4`), and the 348-byte header of the atlas-space image (`..._t88_gfc.hdr`). No image payload is read.

Report, as aggregate counts over the 20 sessions: the sign of the determinant of the transform's 3 x 3 block; the signed-permutation pattern of that block (for each atlas axis, which raw axis dominates and with which sign); the atlas header's orient code and voxel size; and any statement of orientation, frame or handedness in the XML, TXT or transform file text (quoted verbatim, once, if present).

What is derivable and what is not is fixed in advance:
- **Derivable**: whether the raw-to-atlas transform preserves handedness between the two coordinate frames it connects (determinant sign), and how raw axes map to atlas axes.
- **Not derivable without a documented basis**: which anatomical side the raw slice axis points to. That follows only if the frame convention of the transform's raw side and atlas side (the relation between the transform's coordinates and the Analyze voxel axes) is stated in the transform or XML text or in the supplied documents. If it is not, E-geo is declared "no absolute convention", and no left-right claim is made from it.

### 1.5 Evidence line E3: template chirality test on QC numbers (no model, m1 only)
**Template check first.** Load the template exactly as m1 does (`m1_preprocess.load_template(96)`; ICBM152 2009a via nilearn) together with its brain mask. The left-right axis of the 96 x 96 x 96 arrays is the array axis whose ITK direction column has the largest absolute x component. Let F_s be reflection along that axis about plane s, with s searched over the centre plane plus or minus 8 voxels, using the template only. Take s* that maximises the in-mask correlation between the template and its reflection, and call that correlation r_T.
- If **r_T >= 0.99**, the template is symmetric within resampling tolerance. **E3 is declared uninformative and is not run on OASIS data.** The ADNI control below is still computed (no OASIS data involved) to document this.
- If r_T < 0.99, the template is asymmetric and E3 proceeds.

**Statistic.** For a template-aligned, brain-masked 96-cube volume M (m1's `vol96`), D(M) = corr(M, T) - corr(F_s*(M), T), with both correlations computed inside the brain mask by the function m1 uses for `corr_vs_mni`. A positive D means the volume matches the template's lateral asymmetry better than its mirror image does.

**ADNI positive control.** For each of the 20 control volumes (stored full-resolution `mni`, down-sampled to 96 cubed as m1 does), compute D. The control passes if D > 0 in at least 18 of 20. E3 is informative only if the template is asymmetric and the control passes.

**OASIS test (only if E3 is informative).** For each of the 20 B0 scans and each candidate convention c, convert with the loader under c into the data area, run m1 only (no engine, no worker, no model, no SynthSeg), and compute D(c). Convention c **wins** a scan if D(c) is greater than D of the other convention. A convention is **supported** if it wins at least 15 of the 20 scans.

**In-plane sanity.** Over the 20 scans, take for each scan the larger of its two `corr_vs_mni` values. If the median of these is below 0.30 (m1's PASS limit), the assumed in-plane axes are considered refuted: Stage B0 stops, and the candidate set is not widened without a new dated addendum.

### 1.6 Decision rule
No documentation gives left-right, so the evidence lines must agree:
1. If E1 is not consistent, or the in-plane sanity rule fails: **STOP and report.**
2. If E3 is uninformative **and** E-geo gives no absolute convention: **STOP and report.** No convention is picked.
3. If E3 is informative and supports convention c, and E-geo either gives the same convention or gives none: **propose c**, stating which lines support it.
4. If E-geo gives an absolute convention with a documented basis and E3 is uninformative: **propose that convention**, stating that E3 did not corroborate it.
5. If E-geo and E3 are both informative and disagree, or E3 supports neither candidate (neither reaches 15 of 20): **STOP and report.**

In every case the outcome is a proposal for the project owner. Nothing is applied to the protocol body, the manifest or the loader until it is confirmed.

### 1.7 Handling and safeguards
- m1 is the only code that processes an OASIS scan in Stage B0. It runs one scan at a time with no worker process running, and does not start when less than 6 GB of memory is available.
- No image is rendered, saved as a picture or viewed. The m1 QC picture is not generated.
- Per-subject values (QC numbers, D, transform entries) are written only to the data area. Standard output and anything in git show agreement counts and aggregate statistics.
- Converted NIfTI files and m1 outputs stay in the data area and are covered by the end-of-project deletion plan.
- The B0 code is committed before it is run, and the commit is recorded in Part 2.

## Part 2. Results

<!-- B0_RESULTS:BEGIN -->

Run of 2026-11-14 with the Stage B0 code at commit `59ef158`. Generated from `docs/validation/b0_summary.json`; per-subject values are in the data area only.

### E1: readers on a synthetic Analyze file (no OASIS data)

| Reader | Array axes increase toward | Marker side |
|---|---|---|
| ITK (ants.image_read), .hdr read directly | L, P, S | R |
| loader NIfTI (neurological), ITK | A, S, R | R |
| loader NIfTI (neurological), nibabel | A, S, R | R |
| loader NIfTI (radiological), ITK | A, S, L | L |
| loader NIfTI (radiological), nibabel | A, S, L | L |
| nibabel, .hdr read directly | L, A, S | R |

After the orientation and resampling steps of m1 (marker array position on the left-right axis, array length 133): radiological 28 (L); neurological 104 (R).

- PASS: after the m1 step the conventions differ only along the left-right array axis (mirror within 1 voxel)
- PASS: nibabel and ITK agree on the loader NIfTI (axes and marker side), both conventions
- PASS: the two conventions put the marker on opposite sides after the m1 orientation step

m1 relies on ITK's reading (`ants.image_read`) of the file it is given, followed by header-driven reorientation. Read directly, the two readers disagree with each other about the front-back axis and both put the first array axis on left-right, which is not the layout of a sagittal scan; this is why the explicit loader exists. E1 shows that the loader's NIfTI is read identically by both readers under each convention and that the two conventions give mirror images after m1's orientation step. **E1 does not say which convention is right.**

### E-geo: raw-to-atlas transform and XML/TXT text (20 sessions)

- Determinant sign of the 3 x 3 block: + in 20 sessions; absolute determinant 0.7539 to 1.0583.
- Signed-permutation pattern of the block (read as atlas = M x raw, an assumption): `atlas_x<--raw_y,atlas_y<-+raw_z,atlas_z<--raw_x` in 20 sessions.
- Atlas header: orient code 0, dims 176x208x176x1, voxel 1x1x1x0 (20 sessions).
- Text in the transform files other than numbers (digits masked): `$Id: <revision line> $` (20); `<path>/nil-tools/imgreg_#dfp <path>/nil-tools/refdir/young#+#_atrophy_avg_# <path>/nil-t` (20); `t#` (20); `scale:      #` (19); `scale:     #` (1).
- Orientation-like lines in the XML and TXT (digits masked): `<xnat:flip>#</xnat:flip>` (77); `<xnat:orientation>Sag</xnat:orientation>` (77); `Flip:         #` (77); `Orientation:  Sag` (77); `HAND:         Right` (20). The `flip` fields are the flip angle, not a left-right flip.

Derivable: the raw-to-atlas transform preserves handedness (positive determinant) in every session. Not derivable: the pattern shows the transform's raw frame is ordered differently from the Analyze voxel axes, and the relation between that frame and the voxel axes, and the left-right sense of the atlas x axis in these files, are not stated in the transform text, the XML, the TXT or the documents supplied. Stated outcome: **E-geo gives no absolute convention** (convention read from it: none; documented basis: no).

### E3: template chirality test

- Template: the ICBM152 2009 files m1 loads through nilearn are named `nlin_sym_09a` (symmetric). Pre-registered numeric check: in-mask correlation of the template with its reflection r_T = 0.9827 at plane 95 (centre plane 95), limit 0.99. This is below the limit, so by the numeric rule alone the template counts as asymmetric. The files are symmetric by construction, so the limit was a poor discriminator at 96 voxels; it was not adjusted afterwards. The ADNI control below decides in any case.
- ADNI positive control (first 20 train-split PTIDs with a stored volume): 4 of 20 volumes match their stored orientation better than the mirror image (pass needs 18). This is below chance, in the wrong direction.
- E3 informative: **no**. It was therefore not run on OASIS data: no OASIS scan was processed by m1 in Stage B0.

### Decision (rule 1.6)

**Outcome: STOP. Proposed convention: none.** Reason: E3 is uninformative and E-geo gives no absolute convention. No convention is picked, and the loader, the manifest (`orientation` is `UNSET`) and the protocol body are unchanged.

Not tested at all in Stage B0: the in-plane axis directions assumed by the loader (A, S). The in-plane sanity rule is attached to E3, which did not run. A registration-quality check of the in-plane axes does not depend on the left-right sign and can be done on a few scans in either convention, but only under a new dated addendum.

<!-- B0_RESULTS:END -->

## Part 3. How Stage B0 ended and how it was resolved
- **Rule 1.6 outcome: STOP, no convention proposed.** E1 was consistent; E-geo gave no absolute convention; E3 was uninformative. No OASIS scan was processed by m1 in Stage B0.
- **Template.** The ICBM152 files that m1 loads through nilearn are the symmetric 2009a variant (`nlin_sym_09a`). My project plan had assumed an asymmetric template; that was wrong. My pre-registered numeric check (in-mask reflection correlation 0.9827 against a limit of 0.99) classified the template as asymmetric, which was also wrong: the limit was a poor discriminator at 96 voxels. It was recorded as it was and not adjusted.
- **ADNI positive control.** 4 of 20 stored volumes matched their stored orientation better than the mirror image (18 were needed), which is below chance. With a symmetric template the statistic carries no left-right information, and here it is also biased. E3 is recorded as **uninformative**.
- **Resolution.** The primary convention in Part 0 was **not derived from the Stage B0 evidence**: none of E1, E-geo or E3 supports or contradicts it. It rests on the third-party recipe alone, which the Analyze 7.5 specification text neither confirms nor contradicts. If its left-right sign is wrong, ALZ is computed on mirrored brains for the whole cohort; sensitivity S6f measures how much the result depends on this for a defined subset, and Part 0.5 states how an authoritative statement would replace the convention.

