"""EXP2 shared definitions: paths, test guard, tiers, in-fold preprocessing, data loading.

Imports numpy/pandas only, so it loads in both LYP/.venv-fusion (CatBoost, SHAP)
and LYP/.venv-kan (torch, pykan). Never reads a test row: the only test-derived
input is the RID list in dataset/processed/test_rids.csv, used to assert absence.
Writes only under experiments/exp2_reduced_clinical/.
"""
import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
V3 = os.path.join(ROOT, "analysis", "fusion_v3")
OOF, LOGS, TABLES, FIGS = (os.path.join(EXP, d) for d in ("oof", "logs", "tables", "figures"))
EXTRA = os.path.join(EXP, "trainval_extra.parquet")
SEEDS = (42, 43, 44, 45, 46)
NAMES = ["CN", "EMCI", "LMCI", "AD"]

# --- test guard ------------------------------------------------------------------------
TEST_RIDS = frozenset(pd.read_csv(os.path.join(ROOT, "dataset", "processed", "test_rids.csv")).RID.astype(int))
TRAIN_RIDS = frozenset(pd.read_csv(os.path.join(ROOT, "dataset", "processed", "train_rids.csv")).RID.astype(int))
VAL_RIDS = frozenset(pd.read_csv(os.path.join(ROOT, "dataset", "processed", "val_rids.csv")).RID.astype(int))
assert len(TEST_RIDS) == 123 and len(TRAIN_RIDS) == 569 and len(VAL_RIDS) == 122
assert not (TEST_RIDS & (TRAIN_RIDS | VAL_RIDS))


def assert_no_test(rids, where):
    """Abort if any test RID is present; every call is appended to logs/assertions.log."""
    rids = np.asarray(rids).astype(int)
    bad = set(rids.tolist()) & TEST_RIDS
    assert not bad, f"TEST RIDs present in {where}: {sorted(bad)[:5]}"
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "assertions.log"), "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')}\tno_test\t{where}\t{len(rids)}\n")


def assert_disjoint(a_rids, b_rids, where):
    assert not (set(np.asarray(a_rids).tolist()) & set(np.asarray(b_rids).tolist())), f"fold overlap in {where}"
    with open(os.path.join(LOGS, "assertions.log"), "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')}\tdisjoint\t{where}\t{len(a_rids)}+{len(b_rids)}\n")


# --- tiers -----------------------------------------------------------------------------
ECOG = [f"Ecog{s}{d}" for s in ("Pt", "SP") for d in ("Mem", "Lang", "Visspat", "Plan", "Organ", "Divatt", "Total")]
T0 = ["AGE", "PTGENDER", "PTEDUCAT"]
T1 = T0 + ["FAQ"] + ECOG
T2 = T1 + ["MOCA"]
T3 = T2 + ["ADAS11", "ADAS13", "Q4SCORE", "RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting",
           "RAVLT_perc_forgetting", "TRABSCOR"]
PAPER29 = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4", "CDRSB", "ADAS11", "ADAS13", "MMSE", "RAVLT_immediate",
           "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting", "FAQ", "MOCA"] + ECOG + ["CogIndex"]
TIERS = {"T0": T0, "T1": T1, "T2": T2, "T3": T3, "Full-spec": PAPER29, "S-T2+APOE4": T2 + ["APOE4"],
         "Full-paper": PAPER29}
TREND = ["T0", "T1", "T2", "T3", "Full-spec"]
REDUCED = ["T0", "T1", "T2", "T3", "S-T2+APOE4"]
BLACKLIST = {"ACTARM", "ACTARMCD", "ARM", "ARMCD", "ARM_CAT", "DIAGNOSIS", "DX", "DXCHANGE", "DXCONTYP", "DXCONV",
             "DXCURREN", "DXREV", "DXSUM_CAT", "DX_bl", "TRTA", "TRTP", "label"}
FS_RAW = {"Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV"}
EXCLUDED = {"MMSE", "CDRSB", "CDGLOBAL", "CDMEMORY", "CogIndex", "LDELTOTL", "LIMMTOTL", "MPACCTRAILSB"} | BLACKLIST | FS_RAW
PAPER_EXTRAS = ["AGE", "CDRSB", "CogIndex", "MMSE"]
RATIOS = ["Ventricles_ICV", "Hippocampus_ICV", "WholeBrain_ICV", "Entorhinal_ICV", "Fusiform_ICV", "MidTemp_ICV"]
for _t in REDUCED:
    assert not set(TIERS[_t]) & EXCLUDED, _t


def assert_allowed(tier, cols, where):
    if tier in REDUCED:
        bad = set(cols) & EXCLUDED
        assert not bad, f"excluded columns {bad} in {tier} {where}"
        with open(os.path.join(LOGS, "assertions.log"), "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')}\texcluded\t{tier} {where}\t{len(cols)}\n")


# --- in-fold preprocessing (the paper's step-4 rule) ------------------------------------
class Prep:
    """Median (numeric) / mode (PTGENDER) imputation, >10% missingness indicators,
    z-score (ddof 1) and, when CogIndex is requested, z(MMSE) - z(CDRSB) - z(ADAS13).
    Fitted on the rows passed to fit(); identity when paper=True (Full-paper uses
    the already-processed step-4 table)."""

    def __init__(self, cols, paper=False):
        self.cols, self.paper = list(cols), paper
        self.raw = [c for c in self.cols if c != "CogIndex"]
        if "CogIndex" in self.cols and not paper:
            self.raw = list(dict.fromkeys(self.raw + ["MMSE", "CDRSB", "ADAS13"]))

    def fit(self, df):
        if self.paper:
            return self
        num = [c for c in self.raw if c != "PTGENDER"]
        self.med = {c: float(df[c].median()) for c in num}
        self.mode = df.PTGENDER.mode(dropna=True).iloc[0] if "PTGENDER" in self.raw else None
        self.ind = [c for c in num if df[c].isna().mean() > 0.10]
        imp = df[num].fillna(self.med)
        self.mu, self.sd = imp.mean(), imp.std()
        return self

    def transform(self, df):
        if self.paper:
            return df[self.cols].copy()
        num = [c for c in self.raw if c != "PTGENDER"]
        out = (df[num].fillna(self.med) - self.mu) / self.sd
        if "PTGENDER" in self.raw:
            out["PTGENDER"] = df.PTGENDER.fillna(self.mode)
        for c in self.ind:
            out[f"{c}_missing"] = df[c].isna().astype(int)
        if "CogIndex" in self.cols:
            out["CogIndex"] = out.MMSE - out.CDRSB - out.ADAS13
        return out[self.cols + [f"{c}_missing" for c in self.ind]]


# --- data --------------------------------------------------------------------------------
def load():
    """Train (569) and val (122) frames: fusion_trainval.csv (paper step-4 values, MRI inputs)
    + raw clinical values from trainval_extra.parquet (suffix _raw is not used: raw columns
    replace nothing; they live in a separate frame) + brain age + reliability flag."""
    d = pd.read_csv(os.path.join(ROOT, "analysis", "kan_fusion", "fusion_trainval.csv"))
    assert_no_test(d.RID, "fusion_trainval.csv")
    ba = pd.read_csv(os.path.join(ROOT, "analysis", "brain_age", "brainage_trainval.csv"),
                     usecols=["RID", "pred_age", "age_at_scan"])
    rel = pd.read_csv(os.path.join(ROOT, "analysis", "kan_fusion", "brainage_reliability.csv"),
                      usecols=["RID", "unreliable"])
    assert_no_test(ba.RID, "brainage_trainval.csv")
    assert_no_test(rel.RID, "brainage_reliability.csv")
    d = d.merge(ba, on="RID", how="left").merge(rel, on="RID", how="left")
    d["ba_ok"] = d.pred_age.notna() & (d.unreliable != True)              # noqa: E712
    d["SEX"] = (d.PTGENDER == "Male").astype(float)
    raw = pd.read_parquet(EXTRA)
    assert_no_test(raw.RID, "trainval_extra.parquet")
    assert set(raw.RID) == set(d.RID) and len(raw) == len(d)
    raw = raw.set_index("RID").loc[d.RID].reset_index()
    assert (raw.label.to_numpy() == d.label.to_numpy()).all()
    out = {}
    for s in ("train", "val"):
        m = (d.split == s).to_numpy()
        out[s] = (d[m].reset_index(drop=True), raw[m].reset_index(drop=True))
        assert set(out[s][0].RID) == (TRAIN_RIDS if s == "train" else VAL_RIDS)
    return out


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
