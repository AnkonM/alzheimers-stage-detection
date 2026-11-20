"""Step 1.3: MRI_ONLY on a native scan needs AGE and PTGENDER; the engine says so clearly instead of failing with a
TypeError deep in the report builder. Error-only tests: no worker is started and no scan is read."""
import pytest

from src.assessment.engine import AssessmentEngine


@pytest.fixture
def eng(tmp_path):
    e = AssessmentEngine(run_root=tmp_path / "r")
    yield e
    e.close()


@pytest.mark.parametrize("values", [None, {}, {"AGE": 71}, {"PTGENDER": "Female"}])
def test_mri_only_native_scan_needs_age_and_sex(eng, tmp_path, values):
    with pytest.raises(ValueError) as e:
        eng.predict(mode="MRI_ONLY", clinical_values=values, native_path=str(tmp_path / "scan.nii.gz"),
                    report_id="case1", verify_artifacts=False)
    assert str(e.value).startswith("MRI_ONLY needs AGE and PTGENDER")      # the engine guard, not a validator
    assert str(tmp_path) not in str(e.value)                      # the message carries no path


def test_other_modes_keep_their_existing_errors(eng, tmp_path):
    with pytest.raises(ValueError, match="FULL needs both"):
        eng.predict(mode="FULL", clinical_values=None, native_path=None, report_id="c2", verify_artifacts=False)
