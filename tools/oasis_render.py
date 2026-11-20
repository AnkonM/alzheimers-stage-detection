"""Render the OASIS-1 external-validation results as Markdown, from the committed aggregate JSON files only.
Every number comes from docs/validation/oasis_analysis.json, docs/validation/oasis_domain_shift.json and
analysis/external_validation/adni_reference.json; the prose here carries no metric. Used by
tools/build_oasis_tables.py (docs/validation/oasis_results_tables.md) and tools/build_model_card.py (model card section).
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validation.ack import ACK_TEXT, CITATION, DOI, MARKER, notice  # noqa: E402

CLASS_LABEL = {"CN-like": "CDR 0", "CDR0.5": "CDR 0.5", "dementia": "CDR 1 or higher"}
ABSOLUTE_LEVELS = "Absolute score levels differ from ADNI, and no ADNI cutoff applies to OASIS."
TASKS_NOT_IDENTICAL = (
    "The two tasks are not identical: OASIS positives are CDR 1 or higher (clinical diagnosis of probable AD) against CDR 0 elders from a "
    "research-centre cohort with screened images, whereas the ADNI reference is CN vs AD under ADNI criteria, from cross-fitted predictions "
    "on the training split. The comparison shows that the ranking transfers; it is not a measure of relative performance.")


def load():
    j = lambda p: json.loads((ROOT / p).read_text())  # noqa: E731
    return (j("docs/validation/oasis_analysis.json"), j("docs/validation/oasis_domain_shift.json"),
            j("analysis/external_validation/adni_reference.json"))


def f(x, nd=3):
    return f"{x:.{nd}f}"


def p(x):
    return f"{x:.1e}" if x < 1e-3 else f"{x:.3f}"


def ci(c, nd=3):
    return f"[{f(c[0], nd)}, {f(c[1], nd)}]"


def auc_row(name, d, nd=3):
    return f"| {name} | {f(d['auc'], nd)} | {ci(d['ci95'], nd)} | {d['n_pos']} | {d['n_neg']} |"


def table(header, rows):
    return "\n".join([header, "|" + "---|" * (header.count("|") - 1), *rows])


def h1_block(a):
    h, c = a["H1"], a["constants"]
    return "\n".join([
        f"H1 (primary): ALZ for {h['contrast']}, higher score means more impaired. Outcome class **{h['verdict']}** "
        f"(registered thresholds: AUC at least {f(c['h1_pass_auc'], 2)} and CI lower bound above {f(c['h1_pass_ci_lower_above'], 2)}; "
        f"FAIL below an AUC of {f(c['h1_fail_auc_below'], 2)}).",
        "", table("| Quantity | AUC | 95% CI | n positive | n negative |", [auc_row("H1 ALZ", h)]),
        "", f"Subject-level percentile bootstrap, {h['n_boot']} resamples, seed {h['seed']}. "
        f"Cases kept by the primary quality rule: {h['qc']['n_kept']} of {h['qc']['n_total']}."])


def secondary_tables(a):
    s1 = a["S1"]["CDR0.5"]
    out = ["**S1 to S5 (pre-registered secondaries)**", "",
           table("| Analysis | AUC | 95% CI | n positive | n negative |",
                 [auc_row("S1 ALZ, CDR 0 vs CDR 0.5", s1),
                  auc_row("S5 specialist P(AD) (model probability), SynthSeg volumes", a["S5"]),
                  auc_row("S4 benchmark: nWBV alone", a["S4"]["benchmarks"]["nwbv_alone"]),
                  auc_row("S4 benchmark: age + sex logistic, repeated 5-fold cross-validation", a["S4"]["benchmarks"]["age_sex_logistic_repeated_cv"])]),
           ""]
    rows = []
    for key, name in (("S2_alz", "S2 ALZ"), ("S2_bag", "S2 BAG")):
        d = a[key]
        rows.append(f"| {name} | {d['outcome']} | {', '.join(f(m) for m in d['group_median'])} | {', '.join(str(n) for n in d['group_n'])} | "
                    f"{p(d['jt']['p'])} | {f(d['spearman_vs_rank']['rho'])} {ci(d['spearman_vs_rank']['ci95'])} |")
    out += [table("| Trend | Outcome | Group medians (CDR 0, CDR 0.5, CDR 1 or higher) | n | Jonckheere-Terpstra p (one-sided) | Spearman rho [95% CI] |", rows), ""]
    rows = [f"| S3 ALZ with MMSE | {a['S3_alz']['outcome']} | {f(a['S3_alz']['rho'])} {ci(a['S3_alz']['ci95'])} | {a['S3_alz']['n']} |",
            f"| S3 BAG with MMSE | descriptive | {f(a['S3_bag']['rho'])} {ci(a['S3_bag']['ci95'])} | {a['S3_bag']['n']} |"]
    for k, name in (("partial_spearman_vs_rank_age_adjusted", "S4 ALZ with CDR rank, age-adjusted"),
                    ("partial_spearman_vs_mmse_age_adjusted", "S4 ALZ with MMSE, age-adjusted")):
        d = a["S4"][k]
        rows.append(f"| {name} | partial Spearman | {f(d['rho'])} {ci(d['ci95'])} | {d['n']} |")
    out += [table("| Correlation | Outcome | rho [95% CI] | n |", rows), ""]
    e1, e2 = a["S4"]["age_edges"]
    names = (f"age up to {e1}", f"age above {e1} up to {e2}", f"age above {e2}")
    out += [f"S4 AUC of ALZ (CDR 0 vs CDR 1 or higher) within age tertiles (edges fixed at Stage A):", "",
            table("| Tertile | AUC | 95% CI | n positive | n negative |", [auc_row(names[t['tertile']], t) for t in a["S4"]["auc_by_age_tertile"]]), ""]
    ref = a["S5"]["adni_reference"]
    out += [f"S5 ADNI references (train split): CN vs AD AUC {f(ref['cn_vs_ad_auc'])}, AD vs rest AUC {f(ref['ad_vs_rest_auc'])}; "
            f"OASIS minus ADNI CN vs AD: {f(ref['shift_vs_cn_vs_ad'])}. Volume source: {', '.join(a['S5']['volume_source'])}; "
            "these are not the FreeSurfer 5.1 volumes the specialist was trained on. No pass or fail is assigned to S5."]
    return "\n".join(out)


def sensitivity_tables(a):
    s6c, s6e, s6f = a["S6c"], a["S6e"], a["S6f"]
    rows = [auc_row("S6c all scorable cases", s6c["include_all_scorable"]),
            auc_row("S6c worst-case bound (failed or unscored tied at the median)", s6c["worst_case_bound"]),
            auc_row("S6d female", a["S6d"]["F"]), auc_row("S6d male", a["S6d"]["M"]),
            auc_row(f"S6e age {f(s6e['age_range'][0], 1)} to {f(s6e['age_range'][1], 1)}", s6e),
            auc_row("S6f primary convention (60-case subset)", s6f["auc_primary_convention"]),
            auc_row("S6f mirrored convention (60-case subset)", s6f["auc_mirrored_convention"])]
    d = s6f["paired_difference_mirrored_minus_primary"]
    return "\n".join([
        "**S6 sensitivities**", "", table("| Analysis | AUC | 95% CI | n positive | n negative |", rows), "",
        f"S6e excluded {s6e['n_excluded_by_age']} cases outside the age range. S6c: {s6c['include_all_scorable']['n_scored']} scored, "
        f"{s6c['include_all_scorable']['n_not_scored']} not scored.", "",
        f"S6f (mirrored left-right convention, {s6f['n_subset']} cases, {s6f['n_paired_kept']} paired): paired ALZ difference "
        f"(mirrored minus primary) median {f(d['median'], 4)}, interquartile range [{f(d['q25'], 4)}, {f(d['q75'], 4)}]; "
        f"{d['n_lower']} lower and {d['n_mirrored_higher']} higher under the mirrored convention; Spearman between conventions "
        f"{f(s6f['spearman_between_conventions'])}. Labelled sensitivity; it never replaces the primary result.",
        "", "S6a (session-level averaging) is not part of the protocol and S6b (OASIS FreeSurfer volumes) was not run."])


def cohort_table(dom):
    by = dom["qc_new"]["by_group"]
    rows = []
    for g in sorted(by):
        c = by[g]
        rows.append(f"| {g} | {sum(c.values())} | {c['PASS']} | {c['WARN']} | {c['FAIL']} |")
    t = dom["qc_new"]["all"]
    rows.append(f"| all | {dom['qc_new']['n']} | {t['PASS']} | {t['WARN']} | {t['FAIL']} |")
    return table("| CDR | n | registration PASS | WARN | FAIL |", rows)


def descriptive_tables(a, adni):
    d = a["descriptive"]["score_distributions_by_class"]
    names = {"alz": "ALZ", "bag": "Brain-age gap (years)", "p_ad": "Specialist P(AD) (model probability)", "corr_vs_mni": "Registration correlation (corr_vs_mni)"}
    rows = []
    for k in ("alz", "bag", "p_ad", "corr_vs_mni"):
        for cls in ("CN-like", "CDR0.5", "dementia"):
            s = d[k][cls]
            rows.append(f"| {names[k]} | {CLASS_LABEL[cls]} | {s['n']} | {f(s['mean'])} | {f(s['sd'])} | {f(s['median'])} | {f(s['q25'])} | {f(s['q75'])} |")
    out = [table("| Score | OASIS group | n | mean | SD | median | Q25 | Q75 |", rows), "", "ADNI reference (train split, cross-fitted or out-of-fold predictions):", ""]
    rows = []
    for key, name in (("ALZ", "ALZ"), ("BAG", "Brain-age gap (years)"), ("MRI_specialist_Pad", "Specialist P(AD) (model probability)")):
        for g in ("CN", "EMCI", "LMCI", "AD"):
            s = adni[key]["by_group"][g]
            rows.append(f"| {name} | {g} | {s['n']} | {f(s['mean'])} | {f(s['sd'])} |")
    out += [table("| Score | ADNI group | n | mean | SD |", rows), "", ABSOLUTE_LEVELS]
    return "\n".join(out)


def domain_tables(dom):
    feats = dom["features"]
    rows = []
    for k in ("corr_vs_mni", "nz_mean", "nz_std", "p25", "p50", "p75", "p95", "nonzero_frac", "mask_volume_mm3"):
        x = feats[k]
        rows.append(f"| {k} | {f(x['new']['median'], 4)} | {f(x['reference']['median'], 4)} | {f(x['shift']['smd'], 2)} | {f(x['shift']['ks_stat'], 2)} | {p(x['shift']['ks_p'])} |")
    acq = dom["acquisition_new"]
    arows = [f"| {k} | {'; '.join(f'{v} (n {n})' for v, n in sorted(c.items()))} |" for k, c in sorted(acq.items())]
    ref = dom["qc_reference"]["all"]
    return "\n".join([
        table("| Feature (m1 outputs) | OASIS median | ADNI median | standardised mean difference | KS statistic | KS p |", rows), "",
        f"Mask volume is the tissue-filled volume in template space after registration and the fixed template mask, not native brain volume. "
        f"OASIS n {dom['scope']['n_new']}; ADNI train n {dom['qc_reference']['n']}; ADNI registration verdicts PASS {ref['PASS']}, WARN {ref['WARN']}, FAIL {ref['FAIL']}.", "",
        "Acquisition parameters of scan mpr-1 (per-session text):", "", table("| Parameter | Value (cases) |", arows)])


def tables_markdown():
    a, dom, adni = load()
    parts = [
        "# OASIS-1 external validation: generated results tables", "",
        "Generated by `tools/build_oasis_tables.py` from `docs/validation/oasis_analysis.json`, `docs/validation/oasis_domain_shift.json` and "
        "`analysis/external_validation/adni_reference.json`. No number in this file is typed by hand.", "",
        "> " + notice(), "",
        "## Primary result", "", h1_block(a), "",
        "## Cohort and quality accounting (main run)", "", cohort_table(dom), "",
        "## Secondary analyses", "", secondary_tables(a), "",
        "## Sensitivity analyses", "", sensitivity_tables(a), "",
        "## Score distributions next to the ADNI reference", "", descriptive_tables(a, adni), "",
        "## Domain-shift summary", "", domain_tables(dom), ""]
    return "\n".join(parts)


def card_section():
    a, dom, adni = load()
    ref_rows = [f"| ALZ, CDR 0 vs CDR 1 or higher | {f(a['descriptive']['adni_reference_comparison']['alz']['oasis_h1_auc'])} | "
                f"{f(adni['ALZ']['CN_vs_AD']['auc'])} (ADNI CN vs AD) |",
                f"| Specialist P(AD), SynthSeg volumes | {f(a['S5']['auc'])} | {f(adni['MRI_specialist_Pad']['CN_vs_AD']['auc'])} (ADNI CN vs AD) |"]
    return f"""## External validation (OASIS-1, cross-sectional)
<!-- {MARKER} -->
{ACK_TEXT}. Citation: {CITATION}

The frozen system was run, without any training, tuning, threshold selection or adjustment, on the OASIS-1 cross-sectional cohort of
people aged 60 or older (one scan per person), under the pre-registered protocol `docs/external_validation_protocol.md`
(local tag `oasis-protocol-v1`). All numbers below are generated from `docs/validation/oasis_analysis.json`,
`docs/validation/oasis_domain_shift.json` and `analysis/external_validation/adni_reference.json`; full tables are in
`docs/validation/oasis_results_tables.md`.

{h1_block(a)}

{secondary_tables(a)}

{sensitivity_tables(a)}

Same task in the two cohorts (AUC):

{table('| Task | OASIS-1 | ADNI reference (train split) |', ref_rows)}

{TASKS_NOT_IDENTICAL}

{ABSOLUTE_LEVELS}

### What this does and does not show
- Tested: the MRI CNN score (ALZ), the brain-age gap, and the MRI specialist's model probability, for separating CDR 0 from CDR 1 or higher
  and for trends across CDR groups.
- Not tested: the 4-stage output (MRI-only reports have no stage; EMCI and LMCI do not exist in OASIS), the clinical stream (OASIS has no
  such inputs, and CDR is the label), and the fusion layer. These results do not validate staging, severity or the clinical stream.
- Probabilities are model probabilities; their reliability as frequencies was not examined.

### Limitations of this validation
- One scan per person (the first of three or four in the session); averaging over scans was not part of the protocol.
- The left-right convention for the OASIS image files is third-party documented, not documented by OASIS; sensitivity S6f above shows how
  far the result depends on it, for a subset of cases only. S6f does not establish which convention is correct.
- The specialist result (S5) uses SynthSeg volumes, not the FreeSurfer 5.1 volumes it was trained on; agreement between the two is weak for several ratios
  (see the section on SynthSeg volumes).
- Independence of the brain-age model from OASIS is not established: OASIS is not named among its training cohorts, but the published
  training list is incomplete. This also applies to the specialist result (S5), which uses the brain-age gap as an input.
- OASIS-1 is a single cohort from a single research centre and scanner; this is one external check, not a multicentre validation.
- Labels (CDR) and scanner, field strength, acquisition, defacing and image-quality screening differ from ADNI; registration and intensity
  shifts against ADNI are in the domain-shift table of `docs/validation/oasis_results_tables.md`.
- The positive group is small, so intervals are wide; age and sex were not matched, and age-adjusted analyses are reported as S4.
- Clinical-to-scan interval of the CDR label is not analysable from the downloaded files.
- Non-commercial academic use of OASIS data; no OASIS data or subject-level values are part of this repository.

### Dated correction (2026-11-16)
The row "MRI specialist (MRI-only mode)" in the frozen internal results table reports 4-class accuracy, balanced accuracy and macro-F1.
These do not apply to MRI-only reports, which carry no stage; only the AD-versus-rest AUC in that row is relevant to what an MRI-only report shows.
The table is left as generated from the frozen run; this note is the correction.
"""
