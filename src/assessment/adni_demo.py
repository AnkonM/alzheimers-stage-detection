"""ADNI demo and parity cases: clinical values, FreeSurfer volumes and stored m1 outputs, read by RID from the local
(gitignored-by-policy) step-2 and scan tables. Never used for new uploads; nothing here is committed with data."""
import csv

from .clinical import input_fields
from .paths import REPO

PROC = REPO / "dataset" / "processed"
FV3 = REPO / "analysis" / "fusion_v3"
PREP = REPO / "dataset" / "mri_prep"


def split_of(rid: int) -> str:
    for s in ("train", "val", "test"):
        if any(int(float(r["RID"])) == rid for r in csv.DictReader(open(PROC / f"{s}_rids.csv"))):
            return s
    raise KeyError(f"RID {rid} is in no split")


def _row(path, rid):
    for r in csv.DictReader(open(path, newline="")):
        if int(float(r["RID"])) == rid:
            return r
    return None


def case(rid: int) -> dict:
    sp = split_of(rid)
    row = _row(PROC / "step2_features.csv", rid)
    clinical = {}
    for f in input_fields():
        v = row.get(f, "")
        if v != "":
            clinical[f] = v if f == "PTGENDER" else float(v)
    out = {"rid": rid, "split": sp, "clinical": clinical, "has_scan": False}
    ba = _row(FV3 / f"brainage_{sp}.csv", rid) if sp in ("val", "test") else None
    si = _row(FV3 / f"scan_inputs_{sp}.csv", rid) if sp in ("val", "test") else None
    if si and si.get("has_mri") == "True" and ba:
        ptid = si["PTID"]
        out.update(has_scan=True, ptid=ptid, mni=str(PREP / "mni" / f"{ptid}.nii.gz"),
                   vol96=str(PREP / "vol96" / f"{ptid}.nii.gz"), corr_vs_mni=float(ba["corr_vs_mni"]),
                   age_at_scan=float(ba["age_at_scan"]))
    return out
