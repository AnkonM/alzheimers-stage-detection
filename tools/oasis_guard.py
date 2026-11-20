#!/usr/bin/env python3
"""OASIS data-use guard. Standard library only, so a git hook can run it anywhere.

Purpose: mechanically keep OASIS-1 images, derived volumes and subject-level rows
out of git (OASIS data-use terms: non-commercial, no redistribution; the protocol
allows aggregate metrics only). This is separate from tools/dua_scan.py (an ADNI
inventory tool whose baseline is non-zero by design): this guard must exit 0.

Rules
  R1 path     a path component named like the OASIS data area, or a path that
              contains an OASIS subject/session ID.                       [all modes]
  R2 content  an OASIS subject/session ID anywhere in file content.       [all modes]
  R3 mapping  a CSV whose header has a case_id column (the case_id <-> subject map,
              or any per-case table).                                     [all modes]
  R4 formats  a NEW/CHANGED raw neuro-image, array, pickle or raster file outside
              docs/figures/validation/ (only skull-stripped group figures there). [staged, diff]
  R5 ack      a file carrying the OASIS-results marker without the exact
              acknowledgement text and the DOI.                           [all modes]

Findings print rule, path and line only; matched IDs are never echoed.

Usage
  oasis_guard.py --staged            # pre-commit: files in the index
  oasis_guard.py --diff BASE [HEAD]  # CI: files changed between two commits (R4 applies)
  oasis_guard.py --git REF           # every tracked file at REF (R1-R3, R5)
  oasis_guard.py --tree DIR          # every file below DIR (R1-R3, R5); for tests
  oasis_guard.py --history           # pickaxe over all commits for R2 (counts commits)
Exit 0 = clean, 1 = findings, 2 = usage or git error.
"""
import argparse
import os
import re
import subprocess
import sys

# Built so that this file does not contain a match for its own patterns.
ID_PATTERN = r"OAS[0-9]_[0-9]{4}"
ID_RE = re.compile(ID_PATTERN.encode(), re.I)
ID_RE_STR = re.compile(ID_PATTERN, re.I)
REDACT_RE = re.compile(ID_PATTERN + "(_MR[0-9]+)?", re.I)
DATA_DIR_NAMES = {"oasis", "oasis_work", "oasis-data", "oasis_data", "oasis1", "oasis-1"}
RESULTS_MARKER = "OASIS" + "_RESULTS_MARKER"
ACK_TEXT = ("Data were provided by OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, "
            "R. Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, "
            "R01 AG021910, P20 MH071616, U24 RR021382")
DOI = "https://doi.org/10.1162/jocn.2007.19.9.1498"
FIGURE_DIR = "docs/figures/validation/"
FORMATS = (".nii", ".nii.gz", ".hdr", ".img", ".t4", ".mgz", ".mgh", ".npy", ".npz", ".pkl", ".pickle",
           ".h5", ".pt", ".png", ".gif", ".jpg", ".jpeg", ".tif", ".tiff")
MAX_READ = 64 * 1024 * 1024


def git(args, cwd, text=False):
    p = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=text)
    if p.returncode != 0:
        err = p.stderr if text else p.stderr.decode("utf8", "replace")
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {err.strip()[:200]}")
    return p.stdout


def norm(p):
    p = p.replace("\\", "/")
    return p[2:] if p.startswith("./") else p


def check_path(path):
    out = []
    parts = path.split("/")
    if any(c.lower() in DATA_DIR_NAMES for c in parts[:-1]) or parts[-1].lower() in DATA_DIR_NAMES:
        out.append(("R1", path, 0))
    elif ID_RE_STR.search(path):
        out.append(("R1", path, 0))
    return out


def check_content(path, data):
    out = []
    for i, line in enumerate(data.split(b"\n"), 1):
        if ID_RE.search(line):
            out.append(("R2", path, i))
    head = data.split(b"\n", 1)[0].decode("utf8", "ignore").lower()
    if path.lower().endswith(".csv") and "case_id" in [c.strip().strip('"') for c in head.split(",")]:
        out.append(("R3", path, 1))
    if RESULTS_MARKER.encode() in data:
        text = data.decode("utf8", "ignore")
        if ACK_TEXT not in text or DOI not in text:
            out.append(("R5", path, 0))
    return out


def check_format(path):
    low = path.lower()
    if low.endswith(FORMATS) and not path.startswith(FIGURE_DIR):
        return [("R4", path, 0)]
    if path.startswith(FIGURE_DIR) and not low.endswith(".png"):
        return [("R4", path, 0)]
    return []


def scan(items, formats):
    """items: iterable of (path, read_fn). Returns a list of (rule, path, line)."""
    findings = []
    for path, read in items:
        findings += check_path(path)
        if formats:
            findings += check_format(path)
        try:
            data = read()
        except (OSError, RuntimeError):
            continue
        findings += check_content(path, data[:MAX_READ])
    return findings


def staged_items(cwd):
    names = git(["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], cwd).split(b"\0")
    return [(n.decode(), (lambda n=n: git(["show", b":" + n], cwd))) for n in names if n]


def diff_items(base, head, cwd):
    names = git(["diff", "--name-only", "--diff-filter=ACMR", "-z", base, head], cwd).split(b"\0")
    return [(n.decode(), (lambda n=n: git(["show", head.encode() + b":" + n], cwd))) for n in names if n]


def ref_items(ref, cwd):
    names = git(["ls-tree", "-r", "-z", "--name-only", ref], cwd).split(b"\0")
    return [(n.decode(), (lambda n=n: git(["show", ref.encode() + b":" + n], cwd))) for n in names if n]


def tree_items(root):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x != ".git"]
        for f in files:
            full = os.path.join(d, f)
            rel = norm(os.path.relpath(full, root))
            yield rel, (lambda full=full: open(full, "rb").read(MAX_READ))


def ref_findings_fast(ref, cwd):
    """--git REF: path rules over the listing; content rules via git grep (fast), then CSV headers
    and marker files individually."""
    findings = []
    names = [n.decode() for n in git(["ls-tree", "-r", "-z", "--name-only", ref], cwd).split(b"\0") if n]
    for n in names:
        findings += check_path(n)
    p = subprocess.run(["git", "grep", "-a", "-n", "-i", "-E", "-e", ID_PATTERN, ref, "--"], cwd=cwd,
                       capture_output=True)
    if p.returncode not in (0, 1):
        raise RuntimeError("git grep failed")
    for ln in p.stdout.decode("utf8", "replace").splitlines():
        m = re.match(r"^[^:]*:(.+?):(\d+):", ln)
        if m:
            findings.append(("R2", m.group(1), int(m.group(2))))
    for n in names:
        if n.lower().endswith(".csv"):
            try:
                head = git(["show", f"{ref}:{n}"], cwd)[:4096]
            except RuntimeError:
                continue
            findings += [f for f in check_content(n, head.split(b"\n", 1)[0]) if f[0] == "R3"]
    q = subprocess.run(["git", "grep", "-a", "-l", "-F", "-e", RESULTS_MARKER, ref, "--"], cwd=cwd,
                       capture_output=True)
    for ln in q.stdout.decode("utf8", "replace").splitlines():
        path = ln.split(":", 1)[1] if ":" in ln else ln
        findings += [f for f in check_content(path, git(["show", f"{ref}:{path}"], cwd)) if f[0] == "R5"]
    return findings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--staged", action="store_true")
    g.add_argument("--diff", nargs="+", metavar=("BASE", "HEAD"))
    g.add_argument("--git", metavar="REF")
    g.add_argument("--tree", metavar="DIR")
    g.add_argument("--history", action="store_true")
    ap.add_argument("--repo", default=".", help="git work tree (default: cwd)")
    a = ap.parse_args(argv)
    cwd = a.repo
    try:
        if a.staged:
            findings = scan(staged_items(cwd), formats=True)
        elif a.diff:
            base, head = a.diff[0], (a.diff[1] if len(a.diff) > 1 else "HEAD")
            findings = scan(diff_items(base, head, cwd), formats=True)
        elif a.git:
            findings = ref_findings_fast(a.git, cwd)
        elif a.tree:
            findings = scan(tree_items(a.tree), formats=False)
        else:
            out = git(["log", "--all", "-i", "-G", ID_PATTERN, "--format=%h"], cwd, text=True)
            n = len([x for x in out.splitlines() if x.strip()])
            print(f"oasis_guard: history commits touching an OASIS ID pattern: {n}")
            return 1 if n else 0
    except RuntimeError as e:
        print(f"oasis_guard: error: {e}", file=sys.stderr)
        return 2
    findings = sorted(set(findings))
    for rule, path, line in findings:
        print(f"{rule} {REDACT_RE.sub('<id>', path)}" + (f":{line}" if line else ""))
    if findings:
        counts = {}
        for r, _, _ in findings:
            counts[r] = counts.get(r, 0) + 1
        print("oasis_guard: FAIL " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())) +
              "  (matched values are never printed)", file=sys.stderr)
        return 1
    print("oasis_guard: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
