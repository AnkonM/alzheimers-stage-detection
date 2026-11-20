"""pathguard: outputs may not live inside the repo or any git work tree; report ids are safe names."""
import os
import subprocess

import pytest

from src.assessment import pathguard as P
from src.assessment.paths import REPO


def test_repo_and_var_are_refused():
    for p in (REPO, REPO / "var" / "runs", REPO / "x" / "y"):
        with pytest.raises(P.GuardError):
            P.assert_outside_git(p)


def test_any_git_work_tree_is_refused(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path / "other")], check=True)
    with pytest.raises(P.GuardError):
        P.assert_outside_git(tmp_path / "other" / "data" / "runs")
    # a plain directory outside every repository is accepted (tmp_path is not inside a work tree)
    assert P.assert_outside_git(tmp_path / "plain" / "runs") == (tmp_path / "plain" / "runs").resolve()


def test_symlink_into_repo_is_refused(tmp_path):
    link = tmp_path / "link"
    os.symlink(REPO / "var", link)
    with pytest.raises(P.GuardError):
        P.assert_outside_git(link / "runs")


def test_error_message_has_no_path_content(tmp_path):
    secret = tmp_path / "secret_dir_name"
    subprocess.run(["git", "init", "-q", str(secret)], check=True)
    with pytest.raises(P.GuardError) as e:
        P.assert_outside_git(secret / "x", what="run root")
    assert "secret_dir_name" not in str(e.value) and "run root" in str(e.value)


@pytest.mark.parametrize("rid", ["X0001", "demo-ad", "a_b-1", "a" * 64])
def test_report_id_ok(rid):
    assert P.check_report_id(rid) == rid


@pytest.mark.parametrize("rid", ["", "../x", "a/b", ".hidden", "a b", "a" * 65, None, "x\n"])
def test_report_id_rejected(rid):
    with pytest.raises(P.GuardError):
        P.check_report_id(rid)
