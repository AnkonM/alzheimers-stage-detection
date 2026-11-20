import copy
import json
import pathlib

import pytest
from pydantic import ValidationError

from src.assessment.schema import AssessmentReport, export_json_schema

FIX = sorted((pathlib.Path(__file__).parent / "fixtures" / "reports").glob("*.json"))


@pytest.mark.parametrize("p", FIX, ids=lambda p: p.stem)
def test_fixtures_validate(p):
    AssessmentReport.model_validate_json(p.read_text())


def load(name):
    return json.loads((pathlib.Path(__file__).parent / "fixtures" / "reports" / f"{name}.json").read_text())


def test_rejects_mri_term_in_clinical_driver():
    d = load("ad")
    d["drivers"]["clinical"][0]["label"] = "Hippocampal volume"
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)


def test_alzheimer_word_is_not_an_mri_term():
    d = load("ad")
    d["drivers"]["clinical"][0]["label"] = "Alzheimer's Disease Assessment Scale"
    AssessmentReport.model_validate(d)


def test_mri_only_has_no_stage_and_clinical_only_has_no_mri():
    d = load("mri_only")
    d["stage"] = load("ad")["stage"]
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)
    d = load("clinical_only")
    d["volumes"] = load("ad")["volumes"]
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)


def test_synthseg_requires_agreement_text():
    d = load("mri_only")
    d["meta"]["volume_agreement"] = None
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)


def test_word_calibrated_is_banned():
    d = load("ad")
    d["caveats"].append({"id": "x", "text": "Probabilities are calibrated."})
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)


def test_probabilities_must_sum_to_one_and_match_argmax():
    d = load("ad")
    d["stage"]["predicted"] = "CN"
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)


def test_new_upload_requires_domain_shift_and_demo_forbids_it():
    d = load("mri_only")
    d["caveats"] = [c for c in d["caveats"] if c["id"] != "domain_shift"]
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)
    d = load("ad")
    d["caveats"].append({"id": "domain_shift", "text": "x"})
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)


def test_clinical_only_has_no_severity_and_needs_caveat():
    d = load("clinical_only")
    d["stage"]["severity"] = load("ad")["stage"]["severity"]
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)
    d = load("clinical_only")
    d["caveats"] = [c for c in d["caveats"] if c["id"] != "no_severity"]
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)


@pytest.mark.parametrize("bad", ["/home/user/scan.nii.gz", "C:\\data\\x.nii", "subject " + "000" + "_S_" + "0000"])
def test_no_paths_or_ids_in_body(bad):
    d = load("ad")
    d["meta"]["subject_label"] = bad
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)


def test_reference_population_recorded():
    d = load("ad")["meta"]["reference_population"]
    assert d["n"] == 569 and "not age/sex-adjusted" in d["volume_percentiles"]


def test_no_ece_or_calibration_field_in_schema():
    assert "ece" not in json.dumps(export_json_schema()).lower()


def test_json_schema_exports():
    assert "AssessmentReport" in json.dumps(export_json_schema())


def test_clinical_only_stream_drivers_exclude_mri():
    d = load("clinical_only")
    d["drivers"]["stream_level"] = load("ad")["drivers"]["stream_level"]
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)


def test_mode_label_and_alz_caveat_required():
    d = load("clinical_only")
    assert d["meta"]["mode_label"] == "fusion model run without scan inputs"
    d["meta"]["mode_label"] = "clinical specialist mean"
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)
    d = load("ad")
    d["caveats"] = [c for c in d["caveats"] if c["id"] != "alz_batch_effect"]
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)


def test_clinical_only_has_no_brain_age():
    d = load("clinical_only")
    d["brain_age"] = load("ad")["brain_age"]
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)


def test_synthseg_low_agreement_ratios_show_no_percentile():
    from src.assessment.volumes import agreement
    assert agreement.percentile_ok() == __import__("src.assessment.schema", fromlist=["x"]).SYNTHSEG_PERCENTILE_OK
    d = load("mri_only")
    ent = next(v for v in d["volumes"] if v["structure"] == "Entorhinal")
    assert ent["percentile"] is None and "no percentile" in ent["display_note"]
    ent["percentile"] = 30.0
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)


def test_synthseg_needs_warn_gate_first_and_suppresses_mri_changed_flag():
    d = load("mri_only")
    assert d["qc"]["gates"][0]["status"] == "WARN" and "ICC" in d["qc"]["gates"][0]["value"]
    d["qc"]["gates"] = d["qc"]["gates"][1:]
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)
    d = load("mri_only")
    d["flags"].append({"id": "mri_changed_stage", "text": "x"})
    with pytest.raises(ValidationError):
        AssessmentReport.model_validate(d)
