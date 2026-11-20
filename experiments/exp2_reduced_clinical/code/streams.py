"""EXP2 stream scores per tier (LYP/.venv-fusion + PYTHONPATH=LYP/.pylib-exp2). Train/val only.

For tier T and seed s (outer and inner folds as analysis/fusion_v3/step2_streams.py):
  C     clinical tier specialist: in-fold preprocessing (Prep), CatBoost on all tier
        columns -> mean |SHAP| -> top K = min(12, n) -> refit (T0: all 3, no selection).
        Outer OOF over the 569, inner OOF per outer fold (the KAN's training scores),
        and val scores from the 569 fit.
  F-CB  frozen CatBoost on C's top-K ∪ {AGE} + 6 ratios, ALZ, BAG, SEX (no imputation;
        NaN for subjects without a scan; BAG with in-fold alpha/beta), fitted on the same
        rows as C. Outer OOF and val.
  extras  KAN extra inputs per unit (T0: AGE, SEX, PTEDUCAT; Full-paper: AGE, CDRSB,
        CogIndex, MMSE from fusion_trainval; otherwise AGE + first 3 non-AGE columns of
        C's SHAP ranking on the KAN's training rows), preprocessed on those rows.
--tier MRI computes the MRI specialist (tier-independent) and compares it with
analysis/fusion_v3/streams.npz (read-only).
Output: oof/streams_<tier>.npz (or oof/dry_streams_<tier>.npz with --dry).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

sys.path.insert(0, C.V3)
import specialists as S  # noqa: E402  (pure functions/classes; nothing is written)

MRI_KIND, MRI_HP = S.MRI_CANDIDATES["logreg_C0.1"]


class ClinTier:
    def __init__(self, tier, seed):
        self.tier, self.seed = tier, seed
        self.cols = C.TIERS[tier]
        self.paper = tier == "Full-paper"
        self.k = min(12, len(self.cols))
        self.select = tier != "T0"

    def X(self, d, r):
        X = self.prep.transform(d if self.paper else r)
        C.assert_allowed(self.tier, X.columns, "C design matrix")
        return X

    def fit(self, d, r):
        self.prep = C.Prep(self.cols, self.paper).fit(d if self.paper else r)
        X, y = self.X(d, r), d.label.to_numpy()
        cols = list(X.columns)
        if self.select:
            m = S._cb(X, y, self.seed)
            sv = np.abs(np.asarray(shap.TreeExplainer(m).shap_values(X)))
            imp = sv.mean(axis=(0, 2)) if sv.ndim == 3 else sv.mean(0)
            self.rank = [cols[i] for i in np.argsort(-imp, kind="stable")]
            self.top = self.rank[:self.k]
        else:
            self.rank, self.top = cols, cols
        self.model = S._cb(X[self.top], y, self.seed)
        return self

    def predict_proba(self, d, r):
        return self.model.predict_proba(self.X(d, r)[self.top])


def bag_coef(d):
    cn = d[(d.label == 0) & d.ba_ok & d.has_mri]
    return np.polyfit(cn.age_at_scan, cn.pred_age, 1)


def fcb_X(clin, d, r, coef):
    X = clin.X(d, r)
    cols = list(dict.fromkeys(clin.top + ["AGE"]))
    X = X[cols].reset_index(drop=True)
    alpha, beta = coef
    M = pd.DataFrame({c: d[c].to_numpy(dtype=float) for c in C.RATIOS + ["ALZ"]})
    M["BAG"] = np.where(d.ba_ok, (d.pred_age - beta) / alpha - d.age_at_scan, np.nan)
    M["SEX"] = d.SEX.to_numpy()
    M.loc[~d.has_mri.to_numpy(), :] = np.nan
    X = pd.concat([X, M], axis=1)
    C.assert_allowed(clin.tier, X.columns, "F-CB design matrix")
    return X


class FCB:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, clin, d, r):
        self.clin, self.coef = clin, bag_coef(d)
        self.model = S._cb(fcb_X(clin, d, r, self.coef), d.label.to_numpy(), self.seed)
        return self

    def predict_proba(self, d, r):
        return self.model.predict_proba(fcb_X(self.clin, d, r, self.coef))


def extras(tier, clin, d_fit, r_fit, d_ev, r_ev):
    """Names and values (fit rows, eval rows) of the KAN extra inputs."""
    if tier == "Full-paper":
        names = C.PAPER_EXTRAS
        return names, d_fit[names].to_numpy(float), d_ev[names].to_numpy(float)
    if tier == "T0":
        names = ["AGE", "PTGENDER", "PTEDUCAT"]
    else:
        names = ["AGE"] + [c for c in clin.rank if c != "AGE" and not c.endswith("_missing")][:3]
    C.assert_allowed(tier, names, "KAN extras")
    p = C.Prep(names, False).fit(r_fit)              # Prep adds MMSE/CDRSB/ADAS13 itself for CogIndex

    def val(r):
        X = p.transform(r)[names].copy()
        if "PTGENDER" in names:
            X["PTGENDER"] = (X.PTGENDER == "Male").astype(float)
        return X.to_numpy(float)
    return names, val(r_fit), val(r_ev)


def run_tier(tier, data, seeds, folds, do_val, log):
    (tr, rtr), (va, rva) = data["train"], data["val"]
    y = tr.label.to_numpy()
    n_s = len(C.SEEDS)
    out = {"clin_oof": np.full((n_s, len(tr), 4), np.nan), "clin_val": np.full((n_s, len(va), 4), np.nan),
           "fcb_oof": np.full((n_s, len(tr), 4), np.nan), "fcb_val": np.full((n_s, len(va), 4), np.nan)}
    meta = {}
    for seed in seeds:
        si = C.SEEDS.index(seed)
        for fi, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y)):
            if fi not in folds:
                continue
            t0 = time.time()
            otr, ortr = tr.iloc[a].reset_index(drop=True), rtr.iloc[a].reset_index(drop=True)
            ote, orte = tr.iloc[b].reset_index(drop=True), rtr.iloc[b].reset_index(drop=True)
            C.assert_disjoint(otr.RID, ote.RID, f"{tier} s{seed} f{fi} outer")
            assert (ortr.RID.to_numpy() == otr.RID.to_numpy()).all()
            clin = ClinTier(tier, seed).fit(otr, ortr)
            out["clin_oof"][si, b] = clin.predict_proba(ote, orte)
            t_c = time.time() - t0
            ci = np.zeros((len(a), 4))
            for ii, (ia, ib) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(otr, otr.label)):
                C.assert_disjoint(otr.RID.iloc[ia], otr.RID.iloc[ib], f"{tier} s{seed} f{fi} inner{ii}")
                ci[ib] = ClinTier(tier, seed).fit(otr.iloc[ia], ortr.iloc[ia]).predict_proba(otr.iloc[ib], ortr.iloc[ib])
            t_in = time.time() - t0 - t_c
            t1 = time.time()
            out["fcb_oof"][si, b] = FCB(seed).fit(clin, otr, ortr).predict_proba(ote, orte)
            t_f = time.time() - t1
            names, etr, ete = extras(tier, clin, otr, ortr, ote, orte)
            out[f"clin_inner_{seed}_{fi}"] = ci
            out[f"idx_train_{seed}_{fi}"], out[f"idx_test_{seed}_{fi}"] = a, b
            out[f"extras_tr_{seed}_{fi}"], out[f"extras_te_{seed}_{fi}"] = etr, ete
            meta[f"{seed}_{fi}"] = {"top": clin.top, "rank": clin.rank, "extras": names,
                                    "indicators": getattr(clin.prep, "ind", [])}
            log(f"{tier} seed {seed} fold {fi}: C outer {t_c:.1f}s, C inner x5 {t_in:.1f}s, F-CB {t_f:.1f}s; "
                f"top {clin.top}; extras {names}")
        if do_val:
            t0 = time.time()
            C.assert_disjoint(tr.RID, va.RID, f"{tier} s{seed} val")
            clin = ClinTier(tier, seed).fit(tr, rtr)
            out["clin_val"][si] = clin.predict_proba(va, rva)
            out["fcb_val"][si] = FCB(seed).fit(clin, tr, rtr).predict_proba(va, rva)
            names, etr, ete = extras(tier, clin, tr, rtr, va, rva)
            out[f"extras_tr_{seed}_val"], out[f"extras_te_{seed}_val"] = etr, ete
            meta[f"{seed}_val"] = {"top": clin.top, "rank": clin.rank, "extras": names,
                                   "indicators": getattr(clin.prep, "ind", [])}
            log(f"{tier} seed {seed} val: {time.time() - t0:.1f}s; extras {names}")
    return out, meta


def run_mri(data, seeds, folds, do_val, log):
    (tr, _), (va, _) = data["train"], data["val"]
    y = tr.label.to_numpy()
    n_s = len(C.SEEDS)
    out = {"mri_oof": np.full((n_s, len(tr), 4), np.nan), "mri_val": np.full((n_s, len(va), 4), np.nan)}
    for seed in seeds:
        si = C.SEEDS.index(seed)
        for fi, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y)):
            if fi not in folds:
                continue
            otr = tr.iloc[a].reset_index(drop=True)
            C.assert_disjoint(otr.RID, tr.RID.iloc[b], f"MRI s{seed} f{fi} outer")
            out["mri_oof"][si, b] = S.MRI(MRI_KIND, seed, **MRI_HP).fit(otr).predict_proba(tr.iloc[b])
            mi = np.full((len(a), 4), np.nan)
            for ii, (ia, ib) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(otr, otr.label)):
                C.assert_disjoint(otr.RID.iloc[ia], otr.RID.iloc[ib], f"MRI s{seed} f{fi} inner{ii}")
                mi[ib] = S.MRI(MRI_KIND, seed, **MRI_HP).fit(otr.iloc[ia]).predict_proba(otr.iloc[ib])
            out[f"mri_inner_{seed}_{fi}"] = mi
        if do_val:
            out["mri_val"][si] = S.MRI(MRI_KIND, seed, **MRI_HP).fit(tr).predict_proba(va)
    # compare with the paper's saved streams (read-only)
    old = np.load(os.path.join(C.V3, "streams.npz"))
    assert (old["train_rids"] == tr.RID.to_numpy()).all() and (old["val_rids"] == va.RID.to_numpy()).all()
    diffs = {}
    for k, v in out.items():
        if k in old:
            m = ~np.isnan(v)
            diffs[k] = float(np.nanmax(np.abs(v[m] - old[k][m]))) if m.any() else None
    log(f"MRI vs analysis/fusion_v3/streams.npz: max |diff| {diffs}")
    return out, {"max_abs_diff_vs_paper": diffs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=list(C.TIERS) + ["MRI"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS))
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.makedirs(C.OOF, exist_ok=True)
    pre = "dry_" if a.dry else ""
    logf = open(os.path.join(C.LOGS, f"{pre}streams_{a.tier}.log"), "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    data = C.load()
    t0 = time.time()
    if a.tier == "MRI":
        out, meta = run_mri(data, a.seeds, a.folds, not a.no_val, log)
    else:
        out, meta = run_tier(a.tier, data, a.seeds, a.folds, not a.no_val, log)
    meta["runtime_s"] = time.time() - t0
    np.savez(os.path.join(C.OOF, f"{pre}streams_{a.tier}.npz"), train_rids=data["train"][0].RID.to_numpy(),
             val_rids=data["val"][0].RID.to_numpy(), **out)
    C.write_json(meta, os.path.join(C.OOF, f"{pre}streams_{a.tier}_meta.json"))
    log(f"{a.tier} done in {meta['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
