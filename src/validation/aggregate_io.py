"""Aggregate-only writer for anything that may be committed.

Refuses to write (a) an OASIS subject/session ID, (b) a case id of the batch runner, (c) a numeric sequence long
enough to be a per-subject vector, (d) a path outside docs/validation/ or docs/figures/validation/ when asked
to write inside the repository. Every file written carries the results marker, the acknowledgement and the DOI.
"""
import json
import pathlib
import re

from .ack import ACK_TEXT, DOI, MARKER, notice

ID_RE = re.compile(r"OAS[0-9]_[0-9]{4}", re.I)
CASE_ID_RE = re.compile(r"\bX[0-9]{4,}\b")
MAX_NUMERIC_SEQUENCE = 50                       # a histogram or CI is short; a per-subject vector is not
REPO = pathlib.Path(__file__).resolve().parents[2]
ALLOWED_DIRS = (REPO / "docs" / "validation", REPO / "docs" / "figures" / "validation")


class AggregateError(ValueError):
    """The content or destination is not aggregate-only. The message never contains the offending value."""


def _numeric_seq(o):
    return isinstance(o, (list, tuple)) and len(o) > MAX_NUMERIC_SEQUENCE and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in o)


def check_aggregate(o, case_ids=()):
    """Raise AggregateError if `o` (nested dict/list/str/number) is not aggregate-only."""
    ids = set(case_ids)
    if isinstance(o, str):
        if ID_RE.search(o) or CASE_ID_RE.search(o) or any(c and c in o for c in ids):
            raise AggregateError("string contains a subject or case identifier")
    elif _numeric_seq(o):
        raise AggregateError("numeric sequence is long enough to be per-subject values")
    elif isinstance(o, dict):
        for k, v in o.items():
            check_aggregate(str(k), ids)
            check_aggregate(v, ids)
    elif isinstance(o, (list, tuple)):
        for v in o:
            check_aggregate(v, ids)
    elif hasattr(o, "shape") and getattr(o, "ndim", 0) >= 1 and o.size > MAX_NUMERIC_SEQUENCE:
        raise AggregateError("array is large enough to be per-subject values")


def _check_dest(path):
    p = pathlib.Path(path).resolve()
    inside_repo = p == REPO or REPO in p.parents
    if inside_repo and not any(d == p or d in p.parents for d in ALLOWED_DIRS):
        raise AggregateError("inside the repository only docs/validation/ and docs/figures/validation/ are allowed")
    return p


def write_json(path, obj, case_ids=()):
    check_aggregate(obj, case_ids)
    p = _check_dest(path)
    body = {"_notice": notice(), **obj}
    text = json.dumps(body, indent=2, sort_keys=True, default=float) + "\n"
    check_aggregate(text, case_ids)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def write_markdown(path, text, case_ids=()):
    check_aggregate(text, case_ids)
    p = _check_dest(path)
    out = f"<!-- {MARKER} -->\n\n{ACK_TEXT}.\nCitation: {DOI}\n\n{text.rstrip()}\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(out)
    return p
