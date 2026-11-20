"""Assemble an AssessmentReport from the engine's outputs. Every number comes from an engine array or a committed
results file (analysis/inference_refs/results, analysis/explainability/results, analysis/fusion_v3); nothing is
rounded here, and no metric is written as a literal (the M9 bug). A test enforces the last rule.
"""
import csv
import json
from dataclasses import dataclass, field

import numpy as np

from .paths import REPO
from .schema import (ALZ_BATCH_CAVEAT, LOW_AGREEMENT_NOTE, MODE_LABELS, SYNTHSEG_PERCENTILE_OK, ADResult, Ambiguity,
                     AssessmentReport, BrainAge, Caveat, CNNPanel, Completeness, Driver, Drivers, Flag, Meta, QC,
                     RegionDensity, Severity, StageRange, StageResult, StreamContribution, Validation, VolumeItem)

NAMES = ["CN", "EMCI", "LMCI", "AD"]
REFS = REPO / "analysis" / "inference_refs" / "results"
EXPL = REPO / "analysis" / "explainability" / "results"
VOLUMES = ["Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp"]
GROUP_LABEL = {"clinical logits": "clinical specialist", "MRI logits + has_mri": "MRI specialist", "age": "age",
               "CDRSB + CogIndex + MMSE": "CDR-SB / CogIndex / MMSE"}
LABELS = {
    "AGE": "Age (years)", "PTGENDER": "Sex", "PTEDUCAT": "Education (years)", "APOE4": "APOE ε4 alleles",
    "CDRSB": "CDR sum of boxes", "ADAS11": "ADAS-Cog 11", "ADAS13": "ADAS-Cog 13", "MMSE": "MMSE",
    "RAVLT_immediate": "RAVLT immediate recall", "RAVLT_learning": "RAVLT learning",
    "RAVLT_forgetting": "RAVLT forgetting", "RAVLT_perc_forgetting": "RAVLT % forgetting", "FAQ": "FAQ (function)",
    "MOCA": "MoCA", "EcogPtMem": "ECog self: memory", "EcogPtLang": "ECog self: language",
    "EcogPtVisspat": "ECog self: visuospatial", "EcogPtPlan": "ECog self: planning",
    "EcogPtOrgan": "ECog self: organisation", "EcogPtDivatt": "ECog self: divided attention",
    "EcogPtTotal": "ECog self: total", "EcogSPMem": "ECog partner: memory", "EcogSPLang": "ECog partner: language",
    "EcogSPVisspat": "ECog partner: visuospatial", "EcogSPPlan": "ECog partner: planning",
    "EcogSPOrgan": "ECog partner: organisation", "EcogSPDivatt": "ECog partner: divided attention",
    "EcogSPTotal": "ECog partner: total", "CogIndex": "Cognitive index (z MMSE − z CDR-SB − z ADAS13)",
}
MRI_LABELS = {"Ventricles_ICV": "Ventricles / ICV", "Hippocampus_ICV": "Hippocampal volume / ICV",
              "WholeBrain_ICV": "Whole brain / ICV", "Entorhinal_ICV": "Entorhinal cortex / ICV",
              "Fusiform_ICV": "Fusiform gyrus / ICV", "MidTemp_ICV": "Middle temporal gyrus / ICV",
              "ALZ": "CNN AD score (ALZ)", "BAG": "Brain-age gap", "AGE": "Age", "SEX": "Sex"}
_cache = {}


def _json(path):
    if path not in _cache:
        _cache[path] = json.load(open(path))
    return _cache[path]


def refs():
    return _json(REFS / "reference_ranges.json")


def rules():
    return _json(REFS / "rules.json")


@dataclass
class BuildInput:
    report_id: str
    created_at: str
    subject_label: str
    requested_mode: str
    mode: str
    input_provenance: str
    volume_provider: str
    clinical: dict  # validated clinical values as entered
    completeness: list
    n_imputed: int
    qc_gates: list
    vols_mm3: dict | None
    core: dict  # arrays of the one case, see engine.case_arrays
    cnn: dict | None
    has_brain_image: bool
    artifacts: list = field(default_factory=list)
    runtime_s: float | None = None
    code_version: str | None = None
    in_sample: bool = False
    volume_agreement: str | None = None
    visit_label: str | None = None
    extra_caveats: list = field(default_factory=list)  # [{"id","text"}]


def _stage_ranges(table, feat):
    t = table.get(feat)
    if not t or not all(t.get(s) for s in NAMES):
        return None
    return {s: StageRange(p25=t[s]["p25"], p50=t[s]["p50"], p75=t[s]["p75"]) for s in NAMES}


def _percentile(feat_ratio, value):
    q = refs()["cn_volume_quantiles"][feat_ratio]["q"]
    return float(np.interp(value, q, np.arange(len(q)))), 100 * value / q[len(q) // 2]


def volumes(bi):
    if bi.vols_mm3 is None:
        return []
    out = []
    for s in VOLUMES:
        ratio = bi.vols_mm3[s] / bi.vols_mm3["ICV"]
        pct, pm = _percentile(f"{s}_ICV", ratio)
        if bi.volume_provider == "synthseg" and s not in SYNTHSEG_PERCENTILE_OK:
            out.append(VolumeItem(structure=s, ratio_to_icv=ratio, agreement="low", display_note=LOW_AGREEMENT_NOTE))
        else:
            out.append(VolumeItem(structure=s, ratio_to_icv=ratio, percentile=pct, pct_of_cn_median=pm,
                                  agreement="ok" if bi.volume_provider == "synthseg" else "not_applicable"))
    return out


def stage_result(bi):
    c = bi.core
    p = [float(x) for x in c["p"]]
    sev = float(np.dot(p, np.arange(4)))
    top = np.argsort(p)[::-1][:2]
    rule = rules()["ambiguity"]
    margin = p[top[0]] - p[top[1]]
    adjacent = abs(int(top[0]) - int(top[1])) == 1
    means = _json(REPO / "analysis" / "fusion_v3" / "eval_summary.json")["severity"]["KAN K2 (final)"]["cv"]["means"]
    severity = None if bi.mode == "CLINICAL_ONLY" else Severity(
        value=sev, range=tuple(rules()["severity_range"]), stage_means=dict(zip(NAMES, means)))
    return StageResult(model_probability=dict(zip(NAMES, p)), predicted=NAMES[int(np.argmax(p))],
                       clinical_predicted=NAMES[int(np.argmax(c["clin_p"]))], severity=severity,
                       ambiguity=Ambiguity(flag=bool(adjacent and margin < rule["max_margin"]),
                                           top2=(NAMES[int(top[0])], NAMES[int(top[1])]), margin=float(margin),
                                           rule=rule["rule_id"]))


def brain_age(bi):
    c = bi.core
    if c["pred_age"] is None:
        return BrainAge(status="not_available")
    gap = float(c["corrected_gap"])
    return BrainAge(status="unreliable" if c["unreliable"] else "reliable", age_at_scan=float(c["age_at_scan"]),
                    predicted=float(c["pred_age"]), raw_gap=float(c["pred_age"] - c["age_at_scan"]),
                    corrected_gap=gap, extreme=abs(gap) >= rules()["brain_age"]["extreme_gap_years"])


def _validation(key):
    e = _json(EXPL / "eval_status.json")[key]
    return Validation(status="VALIDATED" if e["overall"] == "VALIDATED" else "WEAK",
                      detail="; ".join(e["weak_or_failed"]) or None)


def _fmt_clinical(feat, bi):
    c = bi.core
    if feat == "CogIndex":
        return float(c["cogindex"]), False
    if feat in bi.clinical:
        return bi.clinical[feat], False
    st = _json(REPO / "analysis" / "fusion_v3" / "frozen" / "feature_stats.json")
    return st["median"].get(feat, st["mode"].get(feat)), True


def drivers(bi):
    c, r = bi.core, refs()
    d = Drivers()
    if bi.mode != "MRI_ONLY":
        k = int(np.argmax(c["clin_p"]))
        d.clinical_target = NAMES[k]
        sv = c["cshap"][:, k]
        top = [j for j in np.argsort(-np.abs(sv)) if abs(sv[j]) > rules()["drivers"]["shap_floor"]][: rules()["drivers"]["show_top_clinical"]]
        feats = list(c["clinical_features"])
        for j in top:
            f = feats[j]
            val, imp = _fmt_clinical(f, bi)
            d.clinical.append(Driver(feature=f, label=LABELS[f], value=val, shap=float(sv[j]), imputed=imp,
                                     per_stage_ranges=_stage_ranges(r["clinical"], f)))
        d.clinical_validation = _validation("clinical_shap")
    if bi.mode != "CLINICAL_ONLY" and c["mri_p"] is not None:
        k = int(np.argmax(c["mri_p"]))
        d.mri_target = NAMES[k]
        phi, imp = c["mshap_phi"][:, k], c["mshap_imputed"]
        feats = list(c["mri_features"])
        for j in np.argsort(-np.abs(phi), kind="stable")[: rules()["drivers"]["show_top_mri"]]:
            f = feats[j]
            d.mri.append(Driver(feature=f, label=MRI_LABELS[f], value=_mri_value(f, bi), shap=float(phi[j]),
                                imputed=bool(imp[j]), per_stage_ranges=_stage_ranges(r["mri"], f)))
        d.mri_validation = _validation("mri_shap")
    if bi.mode != "MRI_ONLY" and c.get("kan") is not None:
        kan = c["kan"]
        pred = NAMES[int(np.argmax(c["p"]))]
        t = list(kan["targets"]).index(f"final_{pred}")
        for g, gn in enumerate(kan["groups"]):
            if bi.mode == "CLINICAL_ONLY" and "MRI" in str(gn):
                continue
            v = kan["phi_group"][:, g, t].astype(float)
            same = bool((v > 0).all() or (v < 0).all())
            d.stream_level.append(StreamContribution(group=GROUP_LABEL[str(gn)], mean=float(v.mean()),
                                                     consistent=same and abs(v.mean()) >= rules()["kan_shapley"]["min_effect"]))
    return d


def _mri_value(f, bi):
    c = bi.core
    if f.endswith("_ICV"):
        v = f[:-4]
        return bi.vols_mm3[v] / bi.vols_mm3["ICV"]
    if f == "ALZ":
        return float(c["alz"])
    if f == "BAG":
        return None if c["unreliable"] or c["pred_age"] is None else float(c["corrected_gap"])
    if f == "AGE":
        return bi.clinical.get("AGE")
    return bi.clinical.get("PTGENDER")


def cnn_panel(bi):
    if bi.cnn is None:
        return None
    e = _json(EXPL / "eval_status.json")["cnn"]["cells"]
    ref = {r["region"]: float(r["ad_minus_cn_density"]) for r in csv.DictReader(open(EXPL / "p1_ad_vs_cn.csv"))}
    st = lambda k: {"PASS": "PASS", "FAIL": "FAIL"}.get(e[k]["verdict"], "WARN")  # noqa: E731
    return CNNPanel(n_models=bi.cnn["n_models"], overlay_file="overlay.nii.gz" if bi.cnn.get("overlay") else None,
                    sanity_cascade=st("sanity: cascading randomisation"), sanity_deletion=st("sanity: region deletion"),
                    regions=[RegionDensity(region=n, density=float(d), ad_minus_cn=ref.get(n))
                             for n, d in zip(bi.cnn["regions"], bi.cnn["density"]) if n in ref and n != "other"])


def flags(bi, stage):
    out = []
    sc = _json(EXPL / "card_rules.json")["stage_change"]
    clin, fin = stage.clinical_predicted, stage.predicted
    if clin != fin:
        t = f"Fusion changed the stage: clinical specialist {clin} → final {fin}."
        row = [x for x in sc["A_transitions_train_oof"] if x["from"] == clin and x["to"] == fin]
        if row and row[0]["n"] >= sc["min_transition_n"]:
            x = row[0]
            t += (f" On training subjects (out-of-fold) this change occurred {x['n']} times: {x['corrected']} "
                  f"corrected, {x['broken']} made wrong.")
        out.append(Flag(id="fusion_changed_stage", text=t))
    c = bi.core
    if bi.mode == "FULL" and bi.volume_provider != "synthseg" and c.get("kan") is not None:
        masked = NAMES[int(np.argmax(c["kan"]["p_mri_masked"]))]
        if masked != fin:
            out.append(Flag(id="mri_changed_stage", text=f"MRI changed the stage: without the MRI inputs (scan treated "
                            f"as missing) the system would predict {masked} instead of {fin}."))
    if bi.mode == "FULL" and c["alz"] is not None:
        th = rules()["discordance"]
        k = NAMES.index(clin)
        branch = (k == 3 and c["alz"] < th["P10"]) or (k in (0, 1) and c["alz"] > th["P90"])
        if branch:
            out.append(Flag(id="discordance", text=f"Discordance (exploratory): clinical specialist predicts {clin} while "
                            f"the CNN score is {c['alz']:.2f}. Exploratory and not validated as a predictor of error."))
    if stage.ambiguity and stage.ambiguity.flag:
        a = stage.ambiguity
        out.append(Flag(id="ambiguous", text=f"The top two stages ({a.top2[0]}, {a.top2[1]}) are adjacent and close "
                        f"(rule: {a.rule})."))
    return out


def caveats(bi, ad_alz):
    out = [Caveat(id="research_use", text="Research use only. Not a diagnostic device.")]
    if ad_alz is not None:
        out.append(Caveat(id="alz_batch_effect", text=ALZ_BATCH_CAVEAT))
    if bi.input_provenance == "new_upload":
        out.append(Caveat(id="domain_shift", text="The model was trained on ADNI-processed scans; other scanners and "
                          "protocols are a domain shift, and the result may be less reliable."))
    if bi.in_sample:
        out.append(Caveat(id="in_sample", text="This subject was in the training data of the frozen system (train or "
                          "validation split). Outputs are in-sample and are not performance estimates."))
    if bi.mode == "CLINICAL_ONLY":
        out.append(Caveat(id="no_severity", text="Severity is not shown: this is the fusion model run without scan inputs."))
    if bi.requested_mode != bi.mode:
        out.append(Caveat(id="downgraded", text="MRI failed quality control; the assessment uses clinical inputs only."))
    if bi.n_imputed:
        names = [i.field for i in bi.completeness if i.state == "imputed"]
        out.append(Caveat(id="imputed_inputs", text=f"{bi.n_imputed} clinical inputs were not provided and were replaced "
                          f"by training medians: {', '.join(names)}."))
    out += [Caveat(**x) for x in bi.extra_caveats]
    return out


def build(bi: BuildInput) -> AssessmentReport:
    c = bi.core
    stage = None if bi.mode == "MRI_ONLY" else stage_result(bi)
    if bi.mode == "MRI_ONLY":
        ad = ADResult(model_probability=float(c["mri_p"][3]), source="MRI_SPECIALIST", alz=float(c["alz"]))
    else:
        ad = ADResult(model_probability=float(c["add"]), source="KAN_ADD_HEAD",
                      alz=None if c["alz"] is None else float(c["alz"]))
    has_scan = bi.mode != "CLINICAL_ONLY"
    gates = list(bi.qc_gates)
    rank = {"PASS": 0, "WARN": 1, "FAIL": 2}
    overall = max((g.status for g in gates), key=rank.get, default="PASS")
    return AssessmentReport(
        meta=Meta(report_id=bi.report_id, created_at=bi.created_at, subject_label=bi.subject_label,
                  visit_label=bi.visit_label, mode=bi.mode, mode_label=MODE_LABELS[bi.mode],
                  requested_mode=bi.requested_mode, code_version=bi.code_version, artifacts=bi.artifacts,
                  input_provenance=bi.input_provenance, volume_provider=bi.volume_provider if has_scan else "none",
                  volume_agreement=bi.volume_agreement, runtime_s=bi.runtime_s,
                  has_brain_image=bi.has_brain_image and has_scan),
        qc=QC(gates=gates, overall=overall),
        completeness=Completeness(clinical=bi.completeness, mri_provided=has_scan, n_clinical_imputed=bi.n_imputed),
        stage=stage, ad=ad, brain_age=brain_age(bi) if has_scan else BrainAge(status="not_available"),
        volumes=volumes(bi) if has_scan else [], drivers=drivers(bi),
        cnn=cnn_panel(bi) if has_scan else None, flags=flags(bi, stage) if stage else [],
        caveats=caveats(bi, ad.alz))
