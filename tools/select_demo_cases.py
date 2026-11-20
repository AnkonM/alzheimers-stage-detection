#!/usr/bin/env python3
"""Apply docs/demo_case_selection.md (committed before any candidate was looked at). Run with LYP/.venv-app.

Reads only inputs and quality fields plus the true baseline stage that defines the CN/LMCI/AD slots; never a model
output. Prints the chosen RIDs (local use; do not commit the output).
"""
import csv
import glob
import json
import sys

sys.path.insert(0, ".")
from src.assessment import adni_demo, clinical  # noqa: E402
from src.assessment.paths import REPO  # noqa: E402

PROC, FV3 = REPO / "dataset" / "processed", REPO / "analysis" / "fusion_v3"
NAMES = ["CN", "EMCI", "LMCI", "AD"]


def rows(path):
    return list(csv.DictReader(open(path, newline="")))


def select() -> dict:
    val = {int(float(r["RID"])) for r in rows(PROC / "val_rids.csv")}
    step2 = {int(float(r["RID"])): r for r in rows(PROC / "step2_features.csv")}
    ba = {int(r["RID"]): r for r in rows(FV3 / "brainage_val.csv")}
    si = {int(r["RID"]): r for r in rows(FV3 / "scan_inputs_val.csv")}
    fields = clinical.input_fields()

    def eligible(rid):
        r = step2[rid]
        missing = [f for f in fields if r.get(f, "") == ""]
        return all(r.get(f, "") != "" for f in clinical.REQUIRED) and len(missing) <= 4

    def scan_ok(rid):
        b, s = ba.get(rid), si.get(rid)
        return bool(b and s and s["has_mri"] == "True" and float(b["corr_vs_mni"]) >= 0.30 and b["excluded"] != "True"
                    and s["unreliable"] != "True")

    pool = sorted(r for r in val if r in step2 and eligible(r))
    chosen = {}
    for name, k in (("cn", 0), ("lmci", 2), ("ad", 3)):
        chosen[name] = next(r for r in pool if scan_ok(r) and int(float(step2[r]["label"])) == k)
    noscan = [r for r in pool if not (si.get(r) and si[r]["has_mri"] == "True")]
    chosen["missing_modality"] = noscan[0] if noscan else next(
        r for r in pool if int(float(step2[r]["label"])) in (0, 1) and r not in chosen.values())
    sel = {r["RID"]: r["Image ID"] for r in rows(REPO / "dataset" / "mri_raw" / "selected_scans.csv")}
    on_disk = {}
    for p in glob.glob(__import__("os").environ.get("ADNI_NIFTI_GLOB", "adni_nifti/ADNI/*/*/*/I*/*.nii")):
        on_disk[p.split("/I")[-1].split("/")[0]] = p
    cand = [r for r in pool if scan_ok(r) and sel.get(str(r)) in on_disk]
    up = cand[0] if cand else None
    chosen["new_upload"] = up
    chosen["new_upload_native"] = on_disk[sel[str(up)]] if up else None
    chosen["missing_modality_has_scan"] = bool(si.get(chosen["missing_modality"]) and
                                               si[chosen["missing_modality"]]["has_mri"] == "True")
    return chosen


def main():
    print(json.dumps(select(), indent=1))


if __name__ == "__main__":
    main()
