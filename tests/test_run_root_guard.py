"""Step 1.2: --run-root and the path guard in the engine and CLI, report-id validation, and the identifier
guard in the report schema (OASIS pattern added; error messages carry the kind of problem, never the text).
No workers are started: every case fails or returns before the first model call."""
import json
import pathlib

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from src.assessment import cli
from src.assessment.engine import RUNS, AssessmentEngine
from src.assessment.paths import REPO
from src.assessment.pathguard import GuardError
from src.assessment.schema import AssessmentReport

ID = "OAS" + "1" + "_" + "0042" + "_" + "MR1"           # planted at runtime
PTID = "123" + "_S_" + "4567"
FIX = pathlib.Path(__file__).parent / "fixtures" / "reports"


def test_engine_default_is_unchanged_and_guard_is_opt_in():
    eng = AssessmentEngine()
    assert eng.run_root == RUNS and isinstance(eng.run_root, pathlib.Path)
    eng.close()


def test_engine_guard_refuses_repo_and_creates_nothing():
    target = REPO / "var" / "guard_test_root"
    with pytest.raises(GuardError):
        AssessmentEngine(run_root=target, guard_outside_git=True)
    assert not target.exists()


def test_engine_guard_accepts_a_folder_outside_git_and_logs_beneath_it(tmp_path):
    eng = AssessmentEngine(run_root=tmp_path / "data", guard_outside_git=True)
    assert eng.pool.log_dir == str(tmp_path / "data" / "worker_logs") and (tmp_path / "data" / "worker_logs").is_dir()
    eng.close()


@pytest.mark.parametrize("rid", ["../escape", "a/b", ".hidden", "", "x" * 65])
def test_invalid_report_id_is_refused_before_any_work(tmp_path, rid):
    eng = AssessmentEngine(run_root=tmp_path / "r")
    with pytest.raises(GuardError):
        eng.predict(mode="CLINICAL_ONLY", report_id=rid or "../")
    assert [p.name for p in (tmp_path / "r").iterdir()] == ["worker_logs"]
    eng.close()


def test_cli_refuses_repo_run_root_without_leaking_path():
    r = CliRunner().invoke(cli.app, ["predict", "--mode", "CLINICAL_ONLY", "--run-root", str(REPO / "var" / "zzz")])
    assert r.exit_code != 0 and "inside the repository" in r.output and "zzz" not in r.output
    assert not (REPO / "var" / "zzz").exists()


def test_cli_refuses_external_dataset_scan_without_run_root(tmp_path):
    r = CliRunner().invoke(cli.app, ["predict", "--mode", "MRI_ONLY", "--nifti", str(tmp_path / ID / "scan.nii.gz")])
    assert r.exit_code != 0 and "--run-root" in r.output and ID not in r.output


def test_cli_refuses_bad_report_id():
    r = CliRunner().invoke(cli.app, ["predict", "--mode", "CLINICAL_ONLY", "--report-id", "../x"])
    assert r.exit_code != 0


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


@pytest.mark.parametrize("text,kind", [(f"note about {ID}", "subject ID"), (f"{ID.lower()}", "subject ID"),
                                       (f"visit of {PTID}", "subject ID"), ("see /home/someone/file", "absolute path")])
def test_schema_string_guard_rejects_ids_and_paths_without_echo(text, kind):
    d = load("ad")
    d["caveats"][0]["text"] = text if d["caveats"] else None
    if not d["caveats"]:
        d["caveats"] = [{"id": "x", "text": text}]
    with pytest.raises(ValidationError) as e:
        AssessmentReport.model_validate(d)
    msg = str(e.value)
    assert kind in msg and ID not in msg and ID.lower() not in msg and PTID not in msg and "someone" not in msg


def test_clean_fixtures_still_validate():
    for p in FIX.glob("*.json"):
        AssessmentReport.model_validate_json(p.read_text())
