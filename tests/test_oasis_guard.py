"""tools/oasis_guard.py: every rule fires on a planted pattern, clean content passes, matched
values are never echoed, and the pre-commit hook really blocks a commit.

All OASIS-looking strings are assembled at runtime so this file (and the repo) stays clean.
"""
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GUARD = ROOT / "tools" / "oasis_guard.py"
HOOK = ROOT / "tools" / "githooks" / "pre-commit"
sys.path.insert(0, str(ROOT / "tools"))
import oasis_guard as G  # noqa: E402

SESSION = "OAS" + "1" + "_" + "0042" + "_" + "MR1"   # planted at runtime
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org"}


def sh(cwd, *args, check=True):
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, env=ENV)
    if check and p.returncode:
        raise AssertionError(p.stderr)
    return p


def guard(cwd, *args):
    return sh(cwd, sys.executable, str(GUARD), *args, "--repo", str(cwd), check=False)


@pytest.fixture
def repo(tmp_path):
    sh(tmp_path, "git", "init", "-q")
    sh(tmp_path, "git", "config", "commit.gpgsign", "false")
    return tmp_path


def put(repo, rel, data):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    sh(repo, "git", "add", "-f", rel)


def no_echo(p):
    assert SESSION not in p.stdout + p.stderr and "0042" not in p.stdout + p.stderr


def test_clean_staged_passes(repo):
    put(repo, "docs/validation/summary.md", "aggregate AUC only\n")
    p = guard(repo, "--staged")
    assert p.returncode == 0 and "clean" in p.stdout


def test_planted_id_in_content_fails_without_echo(repo):
    put(repo, "notes.md", f"line1\nsubject {SESSION} here\n")
    p = guard(repo, "--staged")
    assert p.returncode == 1 and "R2 notes.md:2" in p.stdout
    no_echo(p)


def test_planted_id_in_path_fails(repo):
    put(repo, f"out/{SESSION}.txt", "x\n")
    p = guard(repo, "--staged")
    assert p.returncode == 1 and "R1 out/<id>.txt" in p.stdout
    no_echo(p)


@pytest.mark.parametrize("rel", ["oasis_work/x.txt", "OASIS/readme.txt", "a/oasis/b.txt"])
def test_data_area_directory_fails(repo, rel):
    put(repo, rel, "x\n")
    assert guard(repo, "--staged").returncode == 1


@pytest.mark.parametrize("rel", ["a/scan.hdr", "a/scan.img", "a/v.nii.gz", "a/x.t4", "a/arr.npz", "a/fig.png",
                                 "a/m.pkl"])
def test_raw_and_raster_formats_fail_outside_figure_dir(repo, rel):
    put(repo, rel, b"\x00\x01")
    p = guard(repo, "--staged")
    assert p.returncode == 1 and "R4" in p.stdout


def test_group_figure_png_allowed_but_not_other_formats(repo):
    put(repo, "docs/figures/validation/auc.png", b"\x89PNG")
    assert guard(repo, "--staged").returncode == 0
    put(repo, "docs/figures/validation/head.nii.gz", b"\x00")
    p = guard(repo, "--staged")
    assert p.returncode == 1 and "R4 docs/figures/validation/head.nii.gz" in p.stdout


def test_case_id_table_fails(repo):
    put(repo, "t.csv", "case_id,session\nX0001,a\n")
    p = guard(repo, "--staged")
    assert p.returncode == 1 and "R3 t.csv:1" in p.stdout


def test_acknowledgement_required_when_results_marker_present(repo):
    marker = G.RESULTS_MARKER
    put(repo, "docs/validation/a.md", f"{marker}\nAUC 0.8\n")
    p = guard(repo, "--staged")
    assert p.returncode == 1 and "R5 docs/validation/a.md" in p.stdout
    put(repo, "docs/validation/a.md", f"{marker}\nAUC 0.8\n{G.ACK_TEXT}\n{G.DOI}\n")
    assert guard(repo, "--staged").returncode == 0
    put(repo, "docs/validation/a.md", f"{marker}\n{G.ACK_TEXT}\n")      # DOI missing
    assert guard(repo, "--staged").returncode == 1


def test_tree_mode(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "f.txt").write_text(f"a {SESSION}\n")
    (tmp_path / "ok.txt").write_text("fine\n")
    p = sh(tmp_path, sys.executable, str(GUARD), "--tree", str(tmp_path), check=False)
    assert p.returncode == 1 and "R2 x/f.txt:1" in p.stdout and "ok.txt" not in p.stdout
    no_echo(p)


def test_git_ref_diff_and_history_modes(repo):
    put(repo, "a.txt", "clean\n")
    sh(repo, "git", "commit", "-qm", "c1")
    put(repo, "b.txt", f"{SESSION}\n")
    put(repo, "v.nii.gz", b"\x00")
    sh(repo, "git", "commit", "-qm", "c2")
    p = guard(repo, "--git", "HEAD")
    assert p.returncode == 1 and "R2 b.txt:1" in p.stdout
    d = guard(repo, "--diff", "HEAD~1", "HEAD")
    assert d.returncode == 1 and "R4 v.nii.gz" in d.stdout
    sh(repo, "git", "rm", "-q", "b.txt")
    sh(repo, "git", "commit", "-qm", "c3 removes it, history still has it")
    assert guard(repo, "--git", "HEAD").returncode == 0
    h = guard(repo, "--history")
    assert h.returncode == 1 and h.stdout.strip().endswith(": 2")   # the add and the removal
    no_echo(h)


def test_precommit_hook_blocks_commit(repo):
    (repo / "tools" / "githooks").mkdir(parents=True)
    shutil.copy(GUARD, repo / "tools" / "oasis_guard.py")
    shutil.copy(HOOK, repo / "tools" / "githooks" / "pre-commit")
    os.chmod(repo / "tools" / "githooks" / "pre-commit", 0o755)
    sh(repo, "git", "config", "core.hooksPath", "tools/githooks")
    put(repo, "good.md", "ok\n")
    assert sh(repo, "git", "commit", "-qm", "ok", check=False).returncode == 0
    put(repo, "bad.md", f"leak {SESSION}\n")
    bad = sh(repo, "git", "commit", "-qm", "bad", check=False)
    assert bad.returncode != 0
    assert sh(repo, "git", "rev-list", "--count", "HEAD").stdout.strip() == "1"   # no commit created
    no_echo(bad)


def test_guard_source_and_workflow_carry_no_id_literal():
    for p in (GUARD, HOOK, ROOT / ".github" / "workflows" / "oasis_guard.yml", pathlib.Path(__file__)):
        assert not G.ID_RE.search(p.read_bytes()), p


def test_repo_head_has_no_findings():
    if not (ROOT / ".git").exists():
        pytest.skip("not a git work tree")
    p = guard(ROOT, "--git", "HEAD")
    assert p.returncode == 0, p.stdout[:500]


def test_protocol_draft_carries_acknowledgement_and_doi_and_is_clean():
    text = (ROOT / "docs" / "external_validation_protocol.md").read_text()
    assert G.ACK_TEXT in text and G.DOI in text and not G.ID_RE.search(text.encode())
