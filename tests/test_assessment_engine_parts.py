import pytest

from src.assessment import clinical, registry
from src.assessment.mri import qc
from src.assessment.modes import effective_mode


def test_registry_verifies_and_detects_tampering(tmp_path, monkeypatch):
    assert len(registry.verify({"clinical_42.cbm"})) == 1
    monkeypatch.setattr(registry, "EXTRA", [])
    bad = tmp_path / "r.json"
    bad.write_text('{"artifacts":[{"name":"x","root":"repo","path":"does/not/exist","sha256":"0"}]}')
    monkeypatch.setattr(registry, "REGISTRY", bad)
    with pytest.raises(registry.ArtifactError, match="missing: x"):
        registry.verify()


def test_clinical_hard_range_rejects_and_unknown_field():
    with pytest.raises(clinical.ClinicalError, match="MMSE"):
        clinical.validate({"MMSE": 31})
    with pytest.raises(clinical.ClinicalError, match="unknown"):
        clinical.validate({"Hippocampus": 1})
    with pytest.raises(clinical.ClinicalError, match="Ecog"):
        clinical.validate({"EcogSPMem": 5})


def test_ravlt_derivation_matches_adni_definition():
    c, _ = clinical.validate({"ravlt_trials": [3, 5, 6, 7, 8], "ravlt_delayed": 5})
    assert c["RAVLT_immediate"] == 29 and c["RAVLT_learning"] == 5
    assert c["RAVLT_forgetting"] == 3 and c["RAVLT_perc_forgetting"] == 37.5


def test_completeness_imputes_with_median_and_flags_required():
    c, _ = clinical.validate({"MMSE": 24, "CDRSB": 3.5, "PTGENDER": "Male"})
    items, n_imp, fails = clinical.completeness(c)
    by = {i.field: i for i in items}
    assert by["MMSE"].state == "provided"
    assert by["ADAS13"].state == "imputed" and by["ADAS13"].value == 14.0 and by["ADAS13"].note == "training median"
    assert fails == ["required input missing: ADAS13"] and n_imp == len([i for i in items if i.state == "imputed"])


def test_voxel_and_registration_thresholds():
    assert qc.voxel_gate((1, 1, 1.2), (192, 192, 160)).status == "PASS"
    assert qc.voxel_gate((1, 1, 1.8), (192, 192, 160)).status == "WARN"
    assert qc.voxel_gate((1, 1, 2.5), (192, 192, 160)).status == "FAIL"
    assert qc.voxel_gate((1, 1, 1), (192, 192, 60)).status == "FAIL"
    assert qc.voxel_gate((1, 1), (192, 192)).status == "FAIL"
    assert [qc.registration_gate(x).status for x in (0.5, 0.30, 0.2, 0.15, 0.1)] == \
        ["PASS", "PASS", "WARN", "WARN", "FAIL"]
    assert qc.overall([qc.registration_gate(0.5), qc.registration_gate(0.1)]) == "FAIL"


def test_mode_routing():
    assert effective_mode("FULL", True, True, True, True) == ("FULL", [])
    assert effective_mode("FULL", True, True, False, True) == ("CLINICAL_ONLY", ["downgraded", "no_severity"])
    with pytest.raises(ValueError):
        effective_mode("FULL", True, False, True, True)
    with pytest.raises(ValueError):
        effective_mode("FULL", True, True, True, False)
    with pytest.raises(ValueError):
        effective_mode("MRI_ONLY", False, True, False, False)
