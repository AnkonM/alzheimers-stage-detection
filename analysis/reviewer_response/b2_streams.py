"""B2 step 1 (LYP/.venv-fusion): fold-wise preprocessing; stream scores and CatBoost F/R on train CV.

1. extract_raw() (train+val lines only) and gate B2-a: FoldPrep fitted on all 569 train reproduces
   fusion_trainval.csv (29 clinical features incl. CogIndex, 6 ratios; train+val rows) and
   step4_{train,val}.csv (36 features) within 1e-9. Failure stops B2.
2. Harness check: with the global (569) FoldPrep, seed 42 fold 0 reproduces the stored clinical and
   MRI out-of-fold scores (streams.npz) and CatBoost F/R (catboost_preds.npz) on that fold.
3. Fold-wise run, every seed 42-46 and outer fold (the paper's folds):
   outer models (clinical specialist, MRI specialist, CatBoost F = fold top-12 + 6 ratios + ALZ,
   CatBoost R = 36 step-4 features) use FoldPrep fitted on the outer-train rows; the inner OOF stream
   scores use FoldPrep fitted on each inner-train part; KAN extras (AGE, CDRSB, CogIndex, MMSE) for the
   unit come from the outer-train FoldPrep (= the KAN's training rows) and are stored per unit.
Val and test are not scored. Output: private/b2_streams_foldwise.npz.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2_lib as L  # noqa: E402
import common as C  # noqa: E402

sys.path.insert(0, C.V3)
import specialists as S  # noqa: E402

sys.path.insert(0, C.KF)
import catboost_stage as CB  # noqa: E402  (fit(): harness CatBoost; MRI_INPUTS)

MRI_KIND, MRI_HP = S.MRI_CANDIDATES["logreg_C0.1"]
EXTRAS = ["AGE", "CDRSB", "CogIndex", "MMSE"]


def gate_b2a(pv, base_all, raw_all):
    prep = L.FoldPrep().fit(raw_all[base_all.split.to_numpy() == "train"].reset_index(drop=True))
    fr = L.frame(base_all, raw_all, prep)
    ft = pd.read_csv(pv.file(os.path.join(C.KF, "fusion_trainval.csv")))
    ft = ft.set_index("RID").loc[base_all.RID].reset_index()
    clin = S.clinical_features()
    num = [c for c in clin + L.RATIOS if c != "PTGENDER"]
    d1 = float(np.abs(fr[num].to_numpy(float) - ft[num].to_numpy(float)).max())
    g1 = (fr.PTGENDER.to_numpy() == ft.PTGENDER.to_numpy()).all()
    d2 = 0.0
    for s in ("train", "val"):
        s4 = pd.read_csv(pv.file(os.path.join(C.ROOT, "dataset", "processed", f"step4_{s}.csv")))
        C.assert_no_test(s4.RID, f"step4_{s}.csv")
        m = fr.set_index("RID").loc[s4.RID]
        cols = [c for c in s4.columns if c not in ("RID", "label", "PTGENDER")]
        d2 = max(d2, float(np.abs(m[cols].to_numpy(float) - s4[cols].to_numpy(float)).max()))
        g1 = g1 and (m.PTGENDER.to_numpy() == s4.PTGENDER.to_numpy()).all()
    pv.check("gate B2-a: global step-4 rule reproduces fusion_trainval (clinical + ratios) within 1e-9", d1 <= 1e-9,
             1e-9, d1)
    pv.check("gate B2-a: global step-4 rule reproduces step4_{train,val}.csv (36 features) within 1e-9", d2 <= 1e-9,
             1e-9, d2)
    pv.check("gate B2-a: PTGENDER identical", bool(g1))
    pv.check("gate B2-a: no missingness indicator created (as in the paper)", prep.ind == [], [], prep.ind)
    return prep


def fit_unit(seed, base, raw, a, b, prep_outer, inner=True):
    otr, ote = L.frame(base.iloc[a], raw.iloc[a], prep_outer), L.frame(base.iloc[b], raw.iloc[b], prep_outer)
    out = {}
    clin = S.Clinical(seed).fit(otr)
    out["clin"] = clin.predict_proba(ote)
    out["mri"] = S.MRI(MRI_KIND, seed, **MRI_HP).fit(otr).predict_proba(ote)
    fcols = clin.top + CB.MRI_INPUTS
    out["cbF"] = CB.fit(otr[fcols], otr.label.to_numpy(), seed).predict_proba(ote[fcols])
    rcols = [c for c in RCOLS]
    out["cbR"] = CB.fit(otr[rcols], otr.label.to_numpy(), seed).predict_proba(ote[rcols])
    out["extras_tr"], out["extras_te"] = otr[EXTRAS].to_numpy(float), ote[EXTRAS].to_numpy(float)
    out["top"] = clin.top
    if inner:
        ci, mi = np.zeros((len(a), 4)), np.full((len(a), 4), np.nan)
        ba, ra = base.iloc[a].reset_index(drop=True), raw.iloc[a].reset_index(drop=True)
        for ia, ib in StratifiedKFold(5, shuffle=True, random_state=seed).split(ba, ba.label):
            pi = L.FoldPrep().fit(ra.iloc[ia].reset_index(drop=True))
            itr, ite = L.frame(ba.iloc[ia], ra.iloc[ia], pi), L.frame(ba.iloc[ib], ra.iloc[ib], pi)
            ci[ib] = S.Clinical(seed).fit(itr).predict_proba(ite)
            mi[ib] = S.MRI(MRI_KIND, seed, **MRI_HP).fit(itr).predict_proba(ite)
        out["clin_inner"], out["mri_inner"] = ci, mi
    return out


RCOLS = None


def main():
    global RCOLS
    pv = C.Prov("B2_streams", __file__)
    raw_all, n_other = L.extract_raw()
    pv.file(L.SRC)
    pv.note(f"extractor: kept 691 train+val lines; {n_other} other lines skipped after reading their RID field only")
    base_all = S.load_trainval()
    C.assert_no_test(base_all.RID, "specialists.load_trainval")
    raw_all = raw_all.set_index("RID").loc[base_all.RID].reset_index()
    pv.check("raw labels == fusion_trainval labels", (raw_all.label.to_numpy() == base_all.label.to_numpy()).all())
    s4 = pd.read_csv(os.path.join(C.ROOT, "dataset", "processed", "step4_train.csv"), nrows=1)
    RCOLS = [c for c in s4.columns if c not in ("RID", "label")]
    pv.check("R has 36 features", len(RCOLS) == 36, 36, len(RCOLS))
    prep_global = gate_b2a(pv, base_all, raw_all)

    tr_m = (base_all.split == "train").to_numpy()
    base, raw = base_all[tr_m].reset_index(drop=True), raw_all[tr_m].reset_index(drop=True)
    y = base.label.to_numpy()
    st = np.load(pv.file(os.path.join(C.V3, "streams.npz")))
    cbp = np.load(pv.file(os.path.join(C.KF, "catboost_preds.npz")))
    pv.check("train order == streams.npz", (st["train_rids"] == base.RID.to_numpy()).all())
    pv.check("train order == catboost_preds.npz", (cbp["train_rids"] == base.RID.to_numpy()).all())

    # harness check: global prep reproduces the stored predictions on seed 42 fold 0
    a, b = st["idx_train_42_0"], st["idx_test_42_0"]
    h = fit_unit(42, base, raw, a, b, prep_global, inner=False)
    for k, ref in (("clin", st["clin_oof"][0, b]), ("mri", st["mri_oof"][0, b]), ("cbF", cbp["oof_F"][0, b]),
                   ("cbR", cbp["oof_R"][0, b])):
        m = np.isfinite(ref)
        dd = float(np.abs(h[k][m] - ref[m]).max())
        pv.check(f"harness check (global prep, seed 42 fold 0): {k} reproduces stored OOF (< 1e-9)", dd < 1e-9,
                 "< 1e-9", dd)

    n_s = len(C.SEEDS)
    out = {k: np.full((n_s, len(base), 4), np.nan) for k in ("clin_oof", "mri_oof", "cbF_oof", "cbR_oof")}
    tops, t0 = {}, time.time()
    for si, seed in enumerate(C.SEEDS):
        for fi, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(base, y)):
            assert (a == st[f"idx_train_{seed}_{fi}"]).all() and (b == st[f"idx_test_{seed}_{fi}"]).all()
            assert not set(base.RID.iloc[a]) & set(base.RID.iloc[b])
            prep = L.FoldPrep().fit(raw.iloc[a].reset_index(drop=True))
            u = fit_unit(seed, base, raw, a, b, prep)
            out["clin_oof"][si, b], out["mri_oof"][si, b] = u["clin"], u["mri"]
            out["cbF_oof"][si, b], out["cbR_oof"][si, b] = u["cbF"], u["cbR"]
            out[f"clin_inner_{seed}_{fi}"], out[f"mri_inner_{seed}_{fi}"] = u["clin_inner"], u["mri_inner"]
            out[f"extras_tr_{seed}_{fi}"], out[f"extras_te_{seed}_{fi}"] = u["extras_tr"], u["extras_te"]
            out[f"idx_train_{seed}_{fi}"], out[f"idx_test_{seed}_{fi}"] = a, b
            tops[f"{seed}_{fi}"] = u["top"]
            print(f"seed {seed} fold {fi} done ({time.time() - t0:.0f}s); top12 {u['top']}", flush=True)
    np.savez(os.path.join(C.PRIV, "b2_streams_foldwise.npz"), train_rids=base.RID.to_numpy(), y=y,
             has=base.has_mri.to_numpy(), **out)
    C.write_json({"fold_top12": tops}, os.path.join(C.OUT, "json", "B2_fold_top12.json"))
    pv.note("CatBoost F uses the clinical specialist's fold top-12 (the same CatBoost/SHAP ranking as "
            "catboost_stage.py on the same rows; confirmed by the harness check) + 6 ratios + ALZ.")
    pv.write([os.path.join(C.OUT, "json", "B2_fold_top12.json")])


if __name__ == "__main__":
    main()
