#!/usr/bin/env python3
"""DUA scan: find files that must not reach a public repository.

The ADNI data use agreement forbids redistributing subject-level data. This
scanner flags anything that is, or may contain, subject-level data or images:

  EXT_IMAGE    neuroimaging files (.nii, .nii.gz, .img, .hdr, .dcm, .mgz)
  EXT_ARRAY    array / pickle containers (.npz, .npy, .pkl, .pickle)
  EXT_WEIGHTS  trained model weights (.pt, .pth, .ckpt, .h5, .cbm, .onnx); excluded by decision D1
  EXT_ARCHIVE  archives (.zip, .tar, .tar.gz, .tgz, .7z)
  PATH_ID      a subject ID (ADNI PTID ###_S_####) in the file path
  ADNI_TABLE   a file named like a known ADNI table (ADNIMERGE, UCSFFSX, ADSL, ...)
  CONTENT_ID   text content containing subject IDs (PTID pattern)
  CSV_SUBJECT  a CSV/TSV whose header has a subject or image key (RID, PTID, IMAGEUID, ...)
  NB_OUTPUT    a Jupyter notebook with stored outputs
  IMAGE_REVIEW a raster image (.png/.jpg/.svg) that may be derived from a scan; needs a human look

Only aggregate output is printed. Subject IDs are always redacted to ###_S_####,
so the report itself can be committed.

Usage:
  dua_scan.py --tree DIR [--allow FILE] [--json OUT]       scan a directory (e.g. a release export)
  dua_scan.py --git REF [REF ...] [--allow FILE]           scan git tree(s) at REF
  dua_scan.py --all-branches [--history]                   every local branch; --history also lists
                                                           flagged paths ever committed on any ref
Exit status: 0 = no findings (after the allowlist), 1 = findings, 2 = usage error.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

PTID = re.compile(r"\b\d{3}_S_\d{4}\b")
IMAGE_EXT = (".nii", ".nii.gz", ".img", ".hdr", ".dcm", ".mgz", ".mgh")
ARRAY_EXT = (".npz", ".npy", ".pkl", ".pickle")
WEIGHT_EXT = (".pt", ".pth", ".ckpt", ".h5", ".hdf5", ".cbm", ".onnx", ".keras")
ARCHIVE_EXT = (".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".gz")
RASTER_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".tif", ".tiff", ".webp")
TEXT_EXT = (".csv", ".tsv", ".txt", ".md", ".json", ".py", ".log", ".yaml", ".yml",
            ".html", ".ipynb", ".tex", ".sh", ".xml", ".cfg", ".ini", ".toml")
ADNI_TABLES = re.compile(
    r"(ADNIMERGE|UCSFFSX|ADSL|DXSUM|MMSE|ADAS|CDR|NEUROBAT|ECOG|MOCA|FAQ|APOERES|PTDEMOG|"
    r"REGISTRY|MRILIST|MRI3META|VISITS|ROSTER|UPENN|BAIPETNMRC|idaSearch|ida_search)",
    re.IGNORECASE)
SUBJECT_COLS = {"rid", "ptid", "imageuid", "image_id", "imageid", "subject", "subject_id",
                "subjectid", "scanid", "loni_image", "visit_id"}
MAX_TEXT_BYTES = 20_000_000


def redact(s: str) -> str:
    return PTID.sub("###_S_####", s)


def classify_path(path: str) -> list[str]:
    low = path.lower()
    base = os.path.basename(path)
    hits = []
    if low.endswith(IMAGE_EXT):
        hits.append("EXT_IMAGE")
    elif low.endswith(ARRAY_EXT):
        hits.append("EXT_ARRAY")
    elif low.endswith(WEIGHT_EXT):
        hits.append("EXT_WEIGHTS")
    elif low.endswith(ARCHIVE_EXT):
        hits.append("EXT_ARCHIVE")
    elif low.endswith(RASTER_EXT):
        hits.append("IMAGE_REVIEW")
    if PTID.search(path):
        hits.append("PATH_ID")
    if low.endswith((".csv", ".tsv", ".xlsx", ".xls", ".parquet")) and ADNI_TABLES.search(base):
        hits.append("ADNI_TABLE")
    return hits


def classify_content(path: str, data: bytes) -> list[str]:
    low = path.lower()
    if not low.endswith(TEXT_EXT) or len(data) > MAX_TEXT_BYTES:
        return []
    text = data.decode("utf-8", errors="replace")
    hits = []
    if PTID.search(text):
        hits.append("CONTENT_ID")
    if low.endswith((".csv", ".tsv")):
        header = text.split("\n", 1)[0]
        cols = {c.strip().strip('"').lower() for c in re.split(r"[,\t]", header)}
        if cols & SUBJECT_COLS:
            hits.append("CSV_SUBJECT")
    if low.endswith(".ipynb"):
        try:
            nb = json.loads(text)
            if any(c.get("outputs") for c in nb.get("cells", []) if c.get("cell_type") == "code"):
                hits.append("NB_OUTPUT")
        except json.JSONDecodeError:
            pass
    return hits


def load_allow(path: str | None) -> list[str]:
    if not path:
        return []
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def allowed(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def scan_tree(root: str):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d != ".git"]
        for fn in fns:
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root)
            try:
                with open(full, "rb") as f:
                    data = f.read(MAX_TEXT_BYTES + 1)
            except OSError:
                data = b""
            yield rel, data


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def scan_git(ref: str):
    """Yield (path, content) for every blob in REF, via one `git cat-file --batch`."""
    entries = []
    for line in git("ls-tree", "-r", "-z", "--full-tree", ref).split(b"\0"):
        if not line:
            continue
        meta, path = line.split(b"\t", 1)
        _, typ, sha = meta.split()
        if typ == b"blob":
            entries.append((path.decode(), sha.decode()))
    proc = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    for path, sha in entries:
        need_content = path.lower().endswith(TEXT_EXT)
        proc.stdin.write((sha + "\n").encode())
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        size = int(header[2])
        data = proc.stdout.read(size)
        proc.stdout.read(1)  # trailing newline
        yield path, (data if need_content else b"")
    proc.stdin.close()
    proc.wait()


def run_scan(items, allow):
    findings = defaultdict(list)  # rule -> [redacted path]
    n_files = 0
    for path, data in items:
        n_files += 1
        if allowed(path, allow):
            continue
        rules = classify_path(path) + classify_content(path, data)
        for r in rules:
            findings[r].append(redact(path))
    return n_files, findings


def history_paths(allow):
    """Flagged paths ever added on any ref (path rules only; no content)."""
    out = git("-c", "core.quotePath=false", "log", "--all", "--no-renames", "--diff-filter=A", "--name-only", "--format=")
    hits = defaultdict(list)
    # De-duplicate on the raw path, redact only for output: redacting first would
    # collapse every ###_S_#### file in a folder into one entry.
    for p in sorted({p for p in out.decode().splitlines() if p}):
        if allowed(p, allow):
            continue
        for r in classify_path(p):
            hits[r].append(redact(p))
    return hits


def summarise_dirs(paths, depth=2):
    c = Counter()
    for p in paths:
        parts = p.split("/")
        c["/".join(parts[:depth]) + ("/" if len(parts) > depth else "")] += 1
    return c


def report(label, n_files, findings, out):
    total = sum(len(v) for v in findings.values())
    out.append(f"### {label}: {n_files} files, {total} findings")
    if not total:
        out.append("")
        return
    out.append("")
    out.append("| Rule | Files | Top locations (file count) |")
    out.append("|---|---|---|")
    for rule in sorted(findings):
        locs = summarise_dirs(findings[rule]).most_common(6)
        loc_s = "; ".join(f"`{d}` ({n})" for d, n in locs)
        out.append(f"| {rule} | {len(findings[rule])} | {loc_s} |")
    out.append("")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tree")
    g.add_argument("--git", nargs="+")
    g.add_argument("--all-branches", action="store_true")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--allow")
    ap.add_argument("--json", help="write full redacted findings (paths) as JSON")
    ap.add_argument("--md", help="write the aggregate markdown report here")
    a = ap.parse_args(argv)
    allow = load_allow(a.allow)

    out, full, any_findings = [], {}, False
    if a.tree:
        n, f = run_scan(scan_tree(a.tree), allow)
        report(f"tree {a.tree}", n, f, out)
        full[a.tree] = f
        any_findings |= bool(f)
    else:
        refs = a.git or git("for-each-ref", "--format=%(refname:short)", "refs/heads").decode().split()
        for ref in refs:
            n, f = run_scan(scan_git(ref), allow)
            report(f"branch `{ref}`", n, f, out)
            full[ref] = f
            any_findings |= bool(f)
        if a.history:
            h = history_paths(allow)
            report("history (paths ever added, any ref)", sum(len(v) for v in h.values()),
                   {k: sorted(v) for k, v in h.items()}, out)
            full["history"] = {k: sorted(v) for k, v in h.items()}
            any_findings |= bool(h)

    text = "\n".join(out)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(text + "\n")
    print(text)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({k: {r: v for r, v in d.items()} for k, d in full.items()}, fh, indent=1)
    return 1 if any_findings else 0


if __name__ == "__main__":
    sys.exit(main())
