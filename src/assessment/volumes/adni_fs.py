"""FreeSurfer 5.1 volumes for ADNI demo and parity cases (ADNIMERGE2 UCSFFSX51, joined in step 2).

Reads the local, gitignored-by-policy step-2 table by RID, so ADNI cases take the exact frozen path.
Never used for new uploads.
"""
import csv

from ..paths import REPO

TABLE = REPO / "dataset" / "processed" / "step2_features.csv"
VOLUMES = ["Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp"]


def lookup(rid: int) -> dict:
    """Return {Ventricles..MidTemp, ICV} in mm^3 for the subject; raises KeyError if absent or incomplete."""
    with open(TABLE, newline="") as f:
        for row in csv.DictReader(f):
            if int(float(row["RID"])) == int(rid):
                out = {k: float(row[k]) if row[k] != "" else None for k in VOLUMES + ["ICV"]}
                if out["ICV"] is None:
                    raise KeyError(f"RID {rid}: no ICV in the FreeSurfer table")
                return out
    raise KeyError(f"RID {rid} not in the step-2 table")
