#!/usr/bin/env python3
"""Build the S6f manifest (pre-registered sensitivity, protocol section 7): `inputs.csv` restricted to every case with
CDR >= 1 plus the first 30 CDR 0 cases in sorted case-id order, with ONLY the `orientation` column set to
`oasis1_mirrored`. Reads exactly two columns of labels.csv (case_id and CDR); everything else in that file is ignored.
Writes inputs.csv only (never labels.csv). Prints counts and one sha256, never an id, a CDR or a path.

    make_s6f_manifest.py --inputs INPUTS.csv --labels LABELS.csv --out-dir DIR

The selection uses the same label map and rule as `analyze_oasis.s6f_subset` (positive class = CDR >= 1, reference class
= CDR 0), over all cohort cases regardless of any later QC verdict.
"""
import argparse
import csv
import hashlib
import io
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.assessment import labelmaps  # noqa: E402
from src.validation import stats  # noqa: E402

MIRRORED = "oasis1_mirrored"
N_REFERENCE = stats.S6F_N_CDR0   # protocol section 7, S6f: the first 30 CDR 0 cases by sorted case id (defined once, in stats.py)
LABEL_MAP = "cdr_3class"


class S6fError(RuntimeError):
    pass


def read_labels(path):
    """{case_id: CDR class} from the case_id and CDR columns only (looked up by header name, other columns never used)."""
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        if "case_id" not in header or "CDR" not in header:
            raise S6fError("labels file lacks case_id or CDR")
        ic, ik = header.index("case_id"), header.index("CDR")
        lm = labelmaps.get(LABEL_MAP)
        out = {}
        for row in rd:
            if row[ic] in out:
                raise S6fError("duplicate case id in labels")
            out[row[ic]] = lm(labelmaps.parse_cdr(row[ik]))
    return out


def select_ids(classes, n_reference=N_REFERENCE):
    """Every positive-class case plus the first `n_reference` reference-class cases in SORTED case-id order."""
    lm = labelmaps.get(LABEL_MAP)
    ordered = sorted(classes)
    pos = [c for c in ordered if classes[c] == lm.positive]
    ref = [c for c in ordered if classes[c] == lm.reference][:n_reference]
    if len(ref) != n_reference:
        raise S6fError("fewer reference-class cases than the rule needs")
    return sorted(pos + ref), len(pos), len(ref)


def build(inputs_path, labels_path, out_dir, n_reference=N_REFERENCE, expect_positive=None):
    classes = read_labels(labels_path)
    raw = pathlib.Path(inputs_path).read_bytes().decode()
    lines = raw.splitlines(keepends=True)
    header = next(csv.reader([lines[0]]))
    if "case_id" not in header or "orientation" not in header:
        raise S6fError("inputs lacks case_id or orientation")
    ic, io_ = header.index("case_id"), header.index("orientation")
    rows = {}
    for ln in lines[1:]:
        r = next(csv.reader([ln]))
        if r[ic] in rows:
            raise S6fError("duplicate case id in inputs")
        rows[r[ic]] = (r, ln)
    if set(rows) != set(classes):
        raise S6fError("case id sets of inputs and labels differ")
    ids, n_pos, n_ref = select_ids(classes, n_reference)
    if expect_positive is not None and n_pos != expect_positive:
        raise S6fError("positive-class count differs from the expected one")
    nl = "\r\n" if lines[0].endswith("\r\n") else "\n"
    out_lines = [lines[0] if lines[0].endswith(("\n", "\r")) else lines[0] + nl]
    for cid in ids:                                   # sorted case-id order
        r, ln = rows[cid]
        buf = io.StringIO()
        csv.writer(buf, lineterminator=ln[len(ln.rstrip("\r\n")):] or nl).writerow(r)
        if buf.getvalue() != (ln if ln.endswith(("\n", "\r")) else ln + nl):
            raise S6fError("a row does not round-trip through the csv writer; refusing to rewrite it")
        r2 = list(r)
        r2[io_] = MIRRORED
        buf = io.StringIO()
        csv.writer(buf, lineterminator=nl).writerow(r2)
        out_lines.append(buf.getvalue())
    text = "".join(out_lines)
    out = pathlib.Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise S6fError("output folder is not empty")
    out.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(out, 0o700)
    target = out / "inputs.csv"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", newline="") as f:
        f.write(text)
    return {"n_inputs": len(rows), "n_positive": n_pos, "n_reference": n_ref, "n_rows": len(ids),
            "sha256": hashlib.sha256(text.encode()).hexdigest()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    try:
        r = build(a.inputs, a.labels, a.out_dir)
    except S6fError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(f"inputs rows {r['n_inputs']}; selected CDR>=1: {r['n_positive']}; CDR 0: {r['n_reference']}; rows written {r['n_rows']}")
    print(f"s6f inputs.csv sha256 {r['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
