"""Two deliberately separate views of a report.json.

plumbing_view  statuses, QC verdicts, mode and provider: NO model outputs. Used by the batch runner and the plumbing
               checks, so a dry run can be judged without anybody reading a score.
score_view     the model outputs the protocol analyses (ALZ, specialist P(AD), brain-age gap, registration
               correlation). Used ONLY by the analysis step after a run is complete. The batch runner must never
               import it (a test checks this).

Neither view reads `drivers.mri_target`: in MRI_ONLY mode the engine reports the specialist's argmax stage there, a
known issue recorded in the protocol and not changed.
"""
import json
import pathlib

REGISTRATION_GATE = "registration corr_vs_mni"


def _load(path):
    return json.loads(pathlib.Path(path).read_text())


def plumbing_view(path):
    d = _load(path)
    meta, qc = d["meta"], d["qc"]
    try:
        from .schema import AssessmentReport
        AssessmentReport.model_validate(d)
        valid = True
    except Exception:
        valid = False
    return {"schema_valid": valid, "mode": meta.get("mode"), "requested_mode": meta.get("requested_mode"),
            "volume_provider": meta.get("volume_provider"), "input_provenance": meta.get("input_provenance"),
            "qc_status": {g["name"]: g["status"] for g in qc["gates"]}, "qc_overall": qc.get("overall"),
            "has_alz": (d.get("ad") or {}).get("alz") is not None,
            "brain_age_status": (d.get("brain_age") or {}).get("status"),
            "has_stage": d.get("stage") is not None, "runtime_s": meta.get("runtime_s")}


def score_view(path):
    d = _load(path)
    gates = {g["name"]: g for g in d["qc"]["gates"]}
    reg = gates.get(REGISTRATION_GATE, {})
    ad, ba = d.get("ad") or {}, d.get("brain_age") or {}
    return {"alz": ad.get("alz"), "p_ad": ad.get("model_probability"), "p_ad_source": ad.get("source"),
            "bag": ba.get("corrected_gap"), "bag_status": ba.get("status"),
            "corr_vs_mni": reg.get("value"), "qc": reg.get("status"), "qc_overall": d["qc"].get("overall"),
            "mode": d["meta"].get("mode"), "volume_provider": d["meta"].get("volume_provider")}
