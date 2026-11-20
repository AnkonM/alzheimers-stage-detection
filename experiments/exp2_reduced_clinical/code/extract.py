"""C1 extractor: raw step-1 values for the 691 train+val subjects -> trainval_extra.parquet.

dataset/processed/step1_labeled.csv also holds the 123 test subjects, so it is
never loaded as a table. It is streamed line by line with the csv module; a
line is kept only if its RID is a train or val RID, and nothing from a test
line is parsed beyond its RID field. The output is asserted to contain no test
RID and exactly the 691 train+val RIDs.

Check: the raw values, passed through the paper's step-4 transform with the
manifest's stored statistics, must reproduce fusion_trainval.csv for the 28
non-CogIndex clinical features (and CogIndex) on all 691 rows.
Run with LYP/.venv-fusion and PYTHONPATH=LYP/.pylib-exp2 (pyarrow).
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SRC = os.path.join(C.ROOT, "dataset", "processed", "step1_labeled.csv")
MAP = json.load(open(os.path.join(C.ROOT, "dataset", "processed", "step2_manifest.json")))["feature_source_map"]
MAP = {k: v for k, v in MAP.items() if k not in C.FS_RAW}                 # 28 clinical features
MAP.update({"Q4SCORE": "Q4SCORE", "TRABSCOR": "TRABSCOR", "CDGLOBAL": "CDGLOBAL__CDR", "CDMEMORY": "CDMEMORY",
            "LDELTOTL": "LDELTOTL", "LIMMTOTL": "LIMMTOTL", "MPACCTRAILSB": "MPACCTRAILSB", "DIGITSCR": "DIGITSCR"})
KEEP_IDS = {"RID": "RID", "label": "label", "DX_bl": "DX_bl"}


def main():
    want = C.TRAIN_RIDS | C.VAL_RIDS
    rows, n_lines, n_skipped = [], 0, 0
    with open(SRC, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        ix = {name: header.index(src) for name, src in {**KEEP_IDS, **MAP}.items()}
        rid_ix = ix["RID"]
        for line in r:
            n_lines += 1
            rid = int(line[rid_ix])
            if rid not in want:                       # test (or unsplit) subject: dropped unparsed
                n_skipped += 1
                continue
            rows.append({name: line[i] for name, i in ix.items()})
    df = pd.DataFrame(rows)
    df["RID"] = df.RID.astype(int)
    C.assert_no_test(df.RID, "extract output")
    assert set(df.RID) == want and len(df) == 691, (len(df), len(set(df.RID)))
    for c in df.columns:
        if c not in ("PTGENDER", "DX_bl"):
            df[c] = pd.to_numeric(df[c].replace("", np.nan))
    df["PTGENDER"] = df.PTGENDER.replace("", np.nan)
    df["split"] = np.where(df.RID.isin(C.TRAIN_RIDS), "train", "val")

    # check against the paper's step-4 values
    man = json.load(open(os.path.join(C.ROOT, "dataset", "processed", "step4_manifest.json")))
    ft = pd.read_csv(os.path.join(C.ROOT, "analysis", "kan_fusion", "fusion_trainval.csv")).set_index("RID")
    ft = ft.loc[df.RID]
    num = [c for c in MAP if c in man["normalization"]["mean"]]
    worst = 0.0
    for c in num:
        z = (df[c].fillna(man["imputation"]["median"][c]) - man["normalization"]["mean"][c]) / man["normalization"]["std"][c]
        worst = max(worst, float(np.abs(z.to_numpy() - ft[c].to_numpy()).max()))
    g = df.PTGENDER.fillna(man["imputation"]["mode"]["PTGENDER"]).to_numpy()
    assert (g == ft.PTGENDER.to_numpy()).all()
    assert (df.label.to_numpy() == ft.label.to_numpy()).all()
    assert worst < 1e-9, worst
    print(f"lines {n_lines}, kept {len(df)}, skipped {n_skipped}; {len(num)} features reproduce fusion_trainval "
          f"(max |diff| {worst:.1e}); PTGENDER and label identical")
    print("DX_bl x label:", pd.crosstab(df.DX_bl, df.label).to_dict())
    tr = df[df.split == "train"]
    print("train missingness:", {c: round(float(tr[c].isna().mean()), 4) for c in MAP if tr[c].isna().any()})
    df.to_parquet(C.EXTRA, index=False)
    print("wrote", C.EXTRA, df.shape)


if __name__ == "__main__":
    main()
