"""Path guards that keep external-dataset outputs out of git work trees.

OASIS-1 data-use terms: no redistribution, aggregate results only. Per-subject outputs (reports, volumes,
converted scans, logs) therefore live in a data area that is not inside any git work tree, and never inside
this repository, `var/` included. The check resolves symlinks first, so a link into the repo does not help.
"""
import pathlib
import re

from .paths import REPO

EXTERNAL_ID = re.compile(r"OAS[0-9]_", re.I)       # OASIS-1 subject/session id prefix
REPORT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


class GuardError(RuntimeError):
    """A path or identifier violates a data-use rule. The message never contains subject identifiers."""


def real(path) -> pathlib.Path:
    return pathlib.Path(path).expanduser().resolve(strict=False)


def git_work_tree_of(path):
    """Nearest ancestor (or the path itself) that holds a `.git` entry, or None."""
    p = real(path)
    for d in [p, *p.parents]:
        if (d / ".git").exists():
            return d
    return None


def assert_outside_git(path, what="path"):
    """Refuse a path inside any git work tree or inside this repository. Returns the resolved path."""
    p = real(path)
    if p == REPO or REPO in p.parents:
        raise GuardError(f"{what} is inside the repository; per-subject outputs must live outside it")
    tree = git_work_tree_of(p)
    if tree is not None:
        raise GuardError(f"{what} is inside a git work tree; per-subject outputs must live outside every repository")
    return p


def check_report_id(report_id) -> str:
    """Report ids become directory names: no separators, no dots, bounded length."""
    if not isinstance(report_id, str) or not REPORT_ID.fullmatch(report_id):
        raise GuardError("report id must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}")
    return report_id
