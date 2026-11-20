"""Build reviewer_response.zip (outside the repo) and provenance.json; final checks. Any env.

Checks (each must pass, else no zip is written):
  - frozen artefacts unchanged: sha256sum -c analysis/fusion_v3/frozen/MANIFEST.sha256 in this worktree
    (the 15 gitignored ALZ checkpoints are absent here and are checked in the main checkout instead,
    whose result is recorded too), and the manifest's own sha256 equals the freeze commit message's;
  - no tracked file that existed at 8f351b6 is modified (git diff 8f351b6 limited to pre-existing paths);
  - no subject identifiers: no ADNI PTID pattern (ddd_S_dddd), no RID/PTID column, no train/val/test
    RID value as a CSV cell, and no CSV with 123/569/567/691 data rows in the bundle.
"""
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

MANIFEST_SHA = "5e266705ca96290c4e6daaf1476317aeb0c71a07346ff16d02b5b360d7cbde92"
MAIN = os.path.join(os.path.dirname(C.ROOT), "ad-detection")
BUILD = C.OUT
ZIP = os.path.join(os.path.dirname(BUILD), "reviewer_response.zip")
PTID = re.compile(r"\b\d{3}_S_\d{4}\b")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def manifest_check(root):
    r = subprocess.run(["sha256sum", "-c", "analysis/fusion_v3/frozen/MANIFEST.sha256"], cwd=root,
                       capture_output=True, text=True)
    lines = (r.stdout + r.stderr).splitlines()
    ok = [l for l in lines if l.endswith(": OK")]
    bad = [l for l in lines if l.startswith("analysis/") and not l.endswith(": OK")]
    return {"root": root, "ok": len(ok), "not_ok": len(bad), "not_ok_files": sorted({l.split(":")[0] for l in bad}),
            "manifest_sha256": sha(os.path.join(root, "analysis/fusion_v3/frozen/MANIFEST.sha256"))}


def id_scan(files):
    rids = {str(r) for r in (C.TRAIN_RIDS | C.VAL_RIDS | C.TEST_RIDS)}
    problems = []
    for f in files:
        if f.endswith((".png", ".pdf")):
            continue
        txt = open(f, encoding="utf-8", errors="replace").read()
        if PTID.search(txt):
            problems.append(f"{f}: PTID pattern")
        if f.endswith(".csv"):
            rows = list(csv.reader(open(f)))
            hdr = [h.strip().upper() for h in rows[0]] if rows else []
            if {"RID", "PTID", "SUBJECT", "SUBJECT_ID"} & set(hdr):
                problems.append(f"{f}: identifier column")
            if len(rows) - 1 in (123, 567, 569, 691):
                problems.append(f"{f}: {len(rows) - 1} data rows (per-subject export?)")
            body = rows[1:]
            for j in range(len(hdr)):
                col = [r[j].strip() for r in body if j < len(r)]
                ints = [c for c in col if c.isdigit()]
                if len(ints) >= 10 and sum(c in rids for c in ints) > 0.5 * len(ints) and len(set(ints)) > 10:
                    problems.append(f"{f}: column '{rows[0][j]}' looks like subject RIDs")
    return problems


def main():
    t0 = time.strftime("%Y-%m-%d %H:%M:%S")
    git = lambda *a: subprocess.run(["git", *a], cwd=C.ROOT, capture_output=True, text=True).stdout.strip()  # noqa
    m_wt, m_main = manifest_check(C.ROOT), manifest_check(MAIN)
    ckpts = [f for f in m_wt["not_ok_files"] if f.startswith("analysis/fusion_feasibility/runs/")]
    checks = {
        "manifest sha256 == freeze commit message": m_wt["manifest_sha256"] == MANIFEST_SHA == m_main["manifest_sha256"],
        "worktree: every present frozen file OK (only the 15 gitignored ALZ checkpoints absent)":
            m_wt["ok"] == 94 and len(ckpts) == 15 and len(ckpts) == len(m_wt["not_ok_files"]),
        "main checkout: 109/109 OK (incl. the 15 ALZ checkpoints)": m_main["ok"] == 109 and m_main["not_ok"] == 0,
    }
    changed = [l for l in git("diff", "--name-status", "8f351b6", "HEAD").splitlines() + git("status", "--porcelain").splitlines()
               if l and not l.split()[-1].startswith(("analysis/reviewer_response/", "docs/reviewer_response_prereg.md"))
               and l.split()[-1] != ".gitignore"]
    checks["no pre-existing tracked file modified (except .gitignore append)"] = not changed
    gi = git("diff", "8f351b6", "HEAD", "--", ".gitignore")
    checks[".gitignore change is a pure append"] = all(not l.startswith("-") or l.startswith("---")
                                                         for l in gi.splitlines())
    files = sorted(f for f in glob.glob(os.path.join(BUILD, "**", "*"), recursive=True) if os.path.isfile(f))
    probs = id_scan(files)
    checks["no subject identifiers in the bundle"] = not probs
    prov = {"built": t0, "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "commit": git("rev-parse", "HEAD"),
            "base_commit": "8f351b6 (fusion-v3 freeze)", "prereg_commit": git("log", "--format=%H %ad", "--date=iso",
                                                                             "-1", "--", "docs/reviewer_response_prereg.md"),
            "worktree_clean": git("status", "--porcelain") == "",
            "checks": checks, "id_scan_problems": probs, "frozen_manifest": {"worktree": m_wt, "main_checkout": m_main},
            "steps": {os.path.basename(p)[:-5]: json.load(open(p))
                      for p in sorted(glob.glob(os.path.join(BUILD, "provenance", "*.json")))},
            "outputs": {os.path.relpath(f, BUILD): sha(f) for f in files if "/provenance/" not in f}}
    json.dump(prov, open(os.path.join(BUILD, "provenance.json"), "w"), indent=1)
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if probs:
        print("\n".join(probs[:20]))
    if not all(checks.values()):
        sys.exit("checks failed: zip not written")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(glob.glob(os.path.join(BUILD, "**", "*"), recursive=True)):
            if os.path.isfile(f):
                z.write(f, os.path.join("reviewer_response", os.path.relpath(f, BUILD)))
    print("wrote", ZIP)


if __name__ == "__main__":
    main()
