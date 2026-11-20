"""B2 fold-wise preprocessing: raw extractor + FoldPrep (the step-4 rule, fitted on any row set).

extract_raw(): dataset/processed/step2_features.csv holds all 814 subjects, so it is never loaded as a
  table. It is streamed with the csv module; a line is kept only if its RID is a train or val RID, and
  nothing from another line is parsed beyond its RID field. Output: private/b2_raw_trainval.csv (691).
FoldPrep: src/step4_processing.py functions (imported unchanged): median/mode imputation, > 10%
  missingness indicators on the raw fit rows, z-score (ddof 1) of the 35 numeric features, CogIndex from
  the z-scored MMSE, CDRSB, ADAS13; plus the 6 FreeSurfer ratios (volume / ICV; median imputation and
  z-score over the fit rows, as analysis/kan_fusion/build_fusion_dataset.py does over train).
frame(): a copy of the specialists.load_trainval() rows with the 29 clinical features, CogIndex, the 6
  ratios and the 7 raw FreeSurfer columns replaced by FoldPrep output (everything else unchanged).
"""
import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.ROOT, "src"))
import step4_processing as S4  # noqa: E402

SRC = os.path.join(C.ROOT, "dataset", "processed", "step2_features.csv")
RAW = os.path.join(C.PRIV, "b2_raw_trainval.csv")
FS = ["Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV"]
VOLS = FS[:-1]
RATIOS = [f"{v}_ICV" for v in VOLS]


def extract_raw():
    want = C.TRAIN_RIDS | C.VAL_RIDS
    rows, n_other = [], 0
    with open(SRC, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        ri = header.index("RID")
        for line in r:
            if int(line[ri]) in want:
                rows.append(line)
            else:
                n_other += 1
    os.makedirs(C.PRIV, exist_ok=True)
    with open(RAW, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    d = pd.read_csv(RAW)
    C.assert_no_test(d.RID, "b2_raw_trainval.csv")
    assert len(d) == 691 and set(d.RID) == want
    return d, n_other


def load_raw():
    d = pd.read_csv(RAW)
    C.assert_no_test(d.RID, "b2_raw_trainval.csv")
    return d


class FoldPrep:
    def fit(self, raw):
        feats = [c for c in raw.columns if c not in S4.ID_COLUMNS]
        self.cat = [c for c in S4.CATEGORICAL_FEATURES if c in feats]
        self.num = [c for c in feats if c not in self.cat]
        self.medians, self.modes = S4.fit_imputation(raw, self.num, self.cat)
        self.ind, _ = S4.fit_missingness_indicators(raw, self.num, S4.MISSINGNESS_THRESHOLD)
        imp = S4.add_missingness_indicators(S4.apply_imputation(raw, self.medians, self.modes), raw, self.ind)
        self.means, self.stds = S4.fit_normalization(imp, self.num)
        r = self._ratios(raw)
        self.r_med = {c: float(r[c].median()) for c in RATIOS}
        r = r.fillna(self.r_med)
        self.r_mean = {c: float(r[c].mean()) for c in RATIOS}
        self.r_std = {c: float(r[c].std()) for c in RATIOS}
        return self

    @staticmethod
    def _ratios(raw):
        return pd.DataFrame({f"{v}_ICV": raw[v].to_numpy() / raw["ICV"].to_numpy() for v in VOLS}, index=raw.index)

    def transform(self, raw):
        imp = S4.add_missingness_indicators(S4.apply_imputation(raw, self.medians, self.modes), raw, self.ind)
        out = S4.add_cogindex(S4.apply_normalization(imp, self.means, self.stds))
        r = self._ratios(raw).fillna(self.r_med)
        for c in RATIOS:
            out[c] = (r[c] - self.r_mean[c]) / self.r_std[c]
        return out


def frame(base, raw, prep):
    """base, raw: aligned row sets (same RID order). Returns base with processed columns replaced."""
    assert (base.RID.to_numpy() == raw.RID.to_numpy()).all()
    t = prep.transform(raw.reset_index(drop=True))
    out = base.reset_index(drop=True).copy()
    for c in t.columns:
        if c in ("RID", "label"):
            continue
        out[c] = t[c].to_numpy()
    return out
