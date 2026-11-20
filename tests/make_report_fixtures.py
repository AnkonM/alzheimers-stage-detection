"""Generate synthetic AssessmentReport fixtures. All values are invented; no ADNI subject is represented."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.assessment.volumes import agreement  # noqa: E402
from src.assessment.schema import ALZ_BATCH_CAVEAT, MODE_LABELS, AssessmentReport  # noqa: E402

OUT = pathlib.Path(__file__).parent / "fixtures" / "reports"
STAGES = ["CN", "EMCI", "LMCI", "AD"]
RANGES = {s: {"p25": a, "p50": b, "p75": c} for s, (a, b, c) in
          zip(STAGES, [(0, 0.5, 1), (0.5, 1, 2), (1, 2, 3.5), (3, 4.5, 6)])}
REGIONS = ["hippocampus", "amygdala", "ventricles", "thalamus", "basal ganglia", "cerebellum", "brainstem",
           "cerebral white matter", "frontal lobe", "temporal lobe", "parietal lobe", "occipital lobe"]
NOTICE = {"id": "research_use", "text": "Research use only. Not a diagnostic device."}
SHIFT = {"id": "domain_shift", "text": "Model trained on ADNI-processed scans; other scanners and protocols are a domain shift."}


def probs(*p):
    return dict(zip(STAGES, p))


def sev(p):
    return round(sum(k * v for k, v in enumerate(p)), 3)


def base(rid, mode, p, pred, **kw):
    r = {
        "meta": {"report_id": rid, "created_at": "2026-01-01T00:00:00Z", "subject_label": f"demo-{rid}",
                 "mode": mode, "mode_label": MODE_LABELS[mode], "requested_mode": kw.pop("requested", mode), "volume_provider": "adni_fs", "input_provenance": "adni_demo", "has_brain_image": mode != "CLINICAL_ONLY",
                 "artifacts": [{"name": "kan_k2", "sha256": "0" * 64}]},
        "qc": {"gates": [
            {"name": "registration corr_vs_mni", "value": 0.62, "threshold": "PASS >= 0.30", "status": "PASS"},
            {"name": "voxel size", "value": "1.0x1.0x1.2 mm", "threshold": "PASS 0.8-1.6 mm", "status": "PASS"}],
            "overall": "PASS"},
        "completeness": {"mri_provided": True, "n_clinical_imputed": 0, "clinical": [
            {"field": "MMSE", "state": "provided", "value": 24}, {"field": "CDRSB", "state": "provided", "value": 3.5},
            {"field": "ADAS13", "state": "provided", "value": 22}]},
        "ad": {"model_probability": (p or (0, 0, 0, 0))[3], "source": "KAN_ADD_HEAD", "alz": 0.41},
        "brain_age": {"status": "reliable", "age_at_scan": 74.0, "predicted": 77.5, "raw_gap": 3.5,
                      "corrected_gap": 4.1},
        "volumes": [{"structure": n, "ratio_to_icv": v, "percentile": pc, "pct_of_cn_median": pm} for n, v, pc, pm in
                    [("Hippocampus", 0.0041, 22.0, 88.0), ("Ventricles", 0.031, 78.0, 124.0),
                     ("WholeBrain", 0.69, 40.0, 97.0), ("Entorhinal", 0.0021, 30.0, 91.0),
                     ("Fusiform", 0.0112, 35.0, 95.0), ("MidTemp", 0.0125, 33.0, 93.0)]],
        "drivers": {
            "stream_level": [{"group": "clinical specialist", "mean": 1.2, "consistent": True},
                             {"group": "MRI specialist", "mean": 0.4, "consistent": True},
                             {"group": "age", "mean": -0.1, "consistent": False}],
            "clinical": [{"feature": "CDRSB", "label": "CDR-SB", "value": 3.5, "shap": 0.9, "per_stage_ranges": RANGES},
                         {"feature": "MMSE", "label": "MMSE", "value": 24, "shap": 0.5, "per_stage_ranges": RANGES}],
            "clinical_validation": {"status": "WEAK", "detail": "CDR-SB sign check failed (pre-registered)"},
            "mri": [{"feature": "Hippocampus_ratio", "label": "Hippocampal volume / ICV", "value": 0.0041,
                     "shap": 0.3, "per_stage_ranges": RANGES}],
            "mri_validation": {"status": "VALIDATED"}},
        "cnn": {"n_models": 15, "overlay_file": "overlay.nii.gz",
                "regions": [{"region": n, "density": round(1 / 12 + 0.003 * i, 4), "ad_minus_cn": 0.001 * (i - 6)}
                            for i, n in enumerate(REGIONS)]},
        "caveats": [NOTICE, {"id": "alz_batch_effect", "text": ALZ_BATCH_CAVEAT}],
    }
    if p is not None and mode != "MRI_ONLY":
        r["stage"] = {"model_probability": probs(*p), "predicted": pred, "clinical_predicted": pred,
                      "severity": {"value": sev(p), "stage_means": probs(0.3, 1.1, 1.9, 2.6)},
                      "ambiguity": {"flag": False, "top2": [pred, STAGES[(STAGES.index(pred) + 1) % 4]],
                                    "margin": 0.4, "rule": "adjacent top-2 and margin < 0.20"}}
    r.update(kw)
    return r


FIX = {}
FIX["cn"] = base("cn", "FULL", (0.78, 0.15, 0.05, 0.02), "CN", ad={"model_probability": 0.02, "source": "KAN_ADD_HEAD", "alz": 0.08})
FIX["lmci_ambiguous"] = base("lmci", "FULL", (0.05, 0.40, 0.44, 0.11), "LMCI",
                             ad={"model_probability": 0.11, "source": "KAN_ADD_HEAD", "alz": 0.35},
                             flags=[{"id": "ambiguous", "text": "Top two stages are adjacent and within 0.20."},
                                    {"id": "mri_changed_stage", "text": "Without MRI inputs the system would predict EMCI."}])
FIX["lmci_ambiguous"]["stage"]["ambiguity"] = {"flag": True, "top2": ["LMCI", "EMCI"], "margin": 0.04,
                                               "rule": "adjacent top-2 and margin < 0.20"}
FIX["ad"] = base("ad", "FULL", (0.01, 0.04, 0.20, 0.75), "AD", ad={"model_probability": 0.75, "source": "KAN_ADD_HEAD", "alz": 0.88})
m = base("mri_only", "MRI_ONLY", None, "CN", ad={"model_probability": 0.63, "source": "MRI_SPECIALIST", "alz": 0.7})
m["meta"]["volume_provider"] = "synthseg"
m["meta"]["input_provenance"] = "new_upload"
m["meta"]["volume_agreement"] = agreement.summary()
m["qc"]["gates"].insert(0, agreement.qc_gate().model_dump())
m["qc"]["overall"] = "WARN"
for v in m["volumes"]:
    if v["structure"] in ("Hippocampus", "WholeBrain"):
        v["agreement"] = "ok"
    else:
        v.update(agreement="low", percentile=None, pct_of_cn_median=None,
                 display_note="low agreement with training volumes, no percentile")
m["completeness"]["clinical"] = []
m["drivers"]["clinical"] = []
m["drivers"]["stream_level"] = []
m["caveats"].append(SHIFT)
m["caveats"] += agreement.caveats()
FIX["mri_only"] = m
c = base("clin_only", "CLINICAL_ONLY", (0.10, 0.45, 0.35, 0.10), "EMCI", ad={"model_probability": 0.10, "source": "KAN_ADD_HEAD"})
c["meta"]["volume_provider"] = "none"
c.update(volumes=[], cnn=None, brain_age={"status": "not_available"})
c["completeness"]["mri_provided"] = False
c["drivers"]["mri"] = []
c["drivers"]["stream_level"] = []
c["qc"] = {"gates": [], "overall": "PASS"}
c["drivers"]["mri_validation"] = None
c["stage"]["severity"] = None
c["caveats"] = [x for x in c["caveats"] if x["id"] != "alz_batch_effect"]
c["ad"].pop("alz", None)
c["caveats"].append({"id": "no_severity", "text": "Severity is not shown: this is the fusion model run without scan inputs."})
FIX["clinical_only"] = c
q = base("qc_fail", "CLINICAL_ONLY", (0.20, 0.40, 0.30, 0.10), "EMCI", requested="FULL",
         ad={"model_probability": 0.10, "source": "KAN_ADD_HEAD"})
q["meta"]["volume_provider"] = "none"
q.update(volumes=[], cnn=None, brain_age={"status": "not_available"})
q["qc"] = {"gates": [{"name": "registration corr_vs_mni", "value": 0.08, "threshold": "FAIL < 0.15", "status": "FAIL"}],
           "overall": "FAIL"}
q["completeness"]["mri_provided"] = True
q["drivers"]["mri"] = []
q["drivers"]["stream_level"] = []
q["drivers"]["mri_validation"] = None
q["stage"]["severity"] = None
q["caveats"].append({"id": "downgraded", "text": "MRI failed quality control; the assessment uses clinical inputs only."})
q["caveats"] = [x for x in q["caveats"] if x["id"] != "alz_batch_effect"]
q["ad"].pop("alz", None)
q["caveats"].append({"id": "no_severity", "text": "Severity is not shown: this is the fusion model run without scan inputs."})
FIX["qc_fail"] = q

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, d in FIX.items():
        rep = AssessmentReport.model_validate(d)
        (OUT / f"{name}.json").write_text(json.dumps(rep.model_dump(mode="json"), indent=2) + "\n")
    print(len(FIX), "fixtures written")
