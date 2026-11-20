# EXP2 — Preregistration addendum (post-Phase 1, pre-Phase 3)

Written 2026-10-09, after the Phase 1 label audit and the Phase 2 dry run and
reproduction gate, and **before any Phase 3 result exists** (no T0, T1, T2, T3,
Full-spec or S-T2+APOE4 system has been run beyond the single T1 dry-run fold).
**Nothing in this addendum changes R1–R5 or the reproduction gate** in
`PREREGISTRATION.md` (commit `de368a1`). This commit also adds the spec,
`docs/exp2_reduced_clinical.md`. It intentionally amends the spec's 3-commit
policy (recorded under Deviations).

Known when this was written: the Phase 1 results (rule-based classifier on the
691 train+val subjects: 575 uniquely matched, 108 ambiguous (71 LMCI+AD,
37 EMCI+AD), 8 unmatched) and the Full-paper gate values.

## E1 — Rule-ambiguity subsets (exploratory; not part of R1–R5)

From the user, verbatim:

> Using the Phase 1 rule-based classifier, label each train subject as UNAMBIGUOUS (exactly one rule class), AMBIGUOUS (MMSE 24–26 / CDR 0.5 overlap, matching both MCI and AD rules), or UNMATCHED.
>
> For every tier, report C, M, F-KAN and F-CB on the AMBIGUOUS and UNAMBIGUOUS subsets separately, using the same seed-averaged CV OOF predictions: n, accuracy, macro-F1 over the classes present, AD-vs-rest AUC, and paired Δ F-KAN−C with 95% bootstrap CI.
>
> Also report the same on validation subjects, giving n and point estimates only if n < 30.
>
> Report subset sizes and class composition per subset.
>
> The question is whether MRI's contribution is larger in the AMBIGUOUS zone than in the UNAMBIGUOUS zone. Report it descriptively. Make no verdict.

### Implementation (fixed now)

- **Subsets.** The classifier is `code/phase1_label_audit.py:rule_classifier`, applied to the raw baseline MMSE, CDR global, LM delayed recall and education of each subject. UNAMBIGUOUS means exactly one class matches, whether or not it equals the label. AMBIGUOUS means more than one matches; it is asserted that every such subject has CDR 0.5 and MMSE 24–26 and matches an MCI rule plus the AD rule. UNMATCHED means none matches. UNMATCHED is reported for size and composition only.
- **Tiers:** T0, T1, T2, T3, Full-spec, S-T2+APOE4, and Full-paper as reference.
- **Systems:** C, M, F-KAN and F-CB. M is evaluated on subset subjects with a scan.
- **Predictions:** the seed-averaged CV OOF probabilities of the main analysis, restricted to the subset. Nothing is refitted.
- **Metrics:**
  - n and accuracy.
  - Macro-F1 averages per-class F1 over the true classes present in the subset. In a bootstrap resample, it averages over the true classes present in that resample.
  - AD-vs-rest AUC from p(AD); NA if the subset lacks AD or non-AD subjects.
- **Δ F-KAN − C:** computed for macro-F1 and AD-vs-rest AUC with a paired subject bootstrap. 2,000 resamples are drawn within the subset from `np.random.default_rng(0)` and shared across tiers. The CI is the 95% percentile interval. Resamples with an undefined AUC are dropped and counted.
- **Validation:** the same metrics on the 122 val subjects' holdout predictions, per subset. If the subset has n < 30, only n and point estimates are given; otherwise CIs are given as for CV.
- **Description only:** the per-tier Δ in AMBIGUOUS versus UNAMBIGUOUS, and their difference, are reported with no verdict.
