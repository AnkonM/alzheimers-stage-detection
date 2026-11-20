"""B1: logistic-regression and MLP stackers on the residual KAN's exact inputs (train CV only).

Run with LYP/.venv-kan. Val and test are never read: only the 569 train rows of
fusion_trainval.csv and the stream scores in analysis/fusion_v3/streams.npz.
For outer seed s and fold f (prereg B1):
  inputs = kan_fusion.Inputs (imported, unchanged) with the config 32 settings, fitted on the
  outer-train rows' inner OOF stream scores, applied to the outer-test fold's OOF scores
  (kan_fusion.unit_data, as the KAN).
  LR  : multinomial LogisticRegression, L2, C = 1, lbfgs, class_weight balanced.
  MLP : one hidden ReLU layer, h in {4, 8, 16, 32} chosen by inner 5-fold CV on the outer-train rows
        (mean inner macro-F1 of the 5-seed ensemble; tie -> smaller h); class-balanced CE + 0.5 BCE(ADD),
        Adam 0.01, full batch, <= 400 steps, stratified 15% holdout early stopping (patience 40),
        seeds 42-46 averaged.
KAN reference: kan_runs/32.npz (stored, no refit).
"""
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

sys.path.insert(0, C.V3)
import kan_fusion as K  # noqa: E402

HS = (4, 8, 16, 32)
MSEEDS = (42, 43, 44, 45, 46)
MARGIN = {"macro_f1": 0.02, "ad_auc": 0.01}
CFG = None


def init():
    global CFG
    torch.set_num_threads(1)
    K.init_worker()
    C.assert_no_test(K.DATA["train"].RID, "fusion_trainval train")
    # K.DATA["val"] is loaded by kan_fusion.init_worker but never used: only CV units 0-4 are run
    CFG = {**K.DEFAULT, **json.loads(str(np.load(os.path.join(C.V3, "kan_runs", "32.npz"))["cfg"]))}


def sub(d, rows):
    return {k: (v[rows] if isinstance(v, np.ndarray) and len(v) == len(d["y"]) else v) for k, v in d.items()}


def design(trd, ted):
    inp = K.Inputs(CFG, trd["pc"], trd["pm"], trd["has"], trd["y"], trd["extras"])
    return (inp(trd["pc"], trd["pm"], trd["has"], trd["extras"]).numpy().astype(np.float64),
            inp(ted["pc"], ted["pm"], ted["has"], ted["extras"]).numpy().astype(np.float64))


def fit_lr(X, y, Xe):
    m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, class_weight="balanced")
    return m.fit(X, y).predict_proba(Xe)


def fit_mlp(X, y, Xe, h, seed):
    X, Xe = torch.tensor(X, dtype=torch.float32), torch.tensor(Xe, dtype=torch.float32)
    i_fit, i_ho = train_test_split(np.arange(len(y)), test_size=K.HOLDOUT, stratify=y, random_state=seed)
    cnt = np.bincount(y[i_fit], minlength=4)
    w = torch.tensor(len(i_fit) / (4 * cnt), dtype=torch.float32)
    yt = torch.tensor(y)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], h), torch.nn.ReLU(), torch.nn.Linear(h, 5))
    opt = torch.optim.Adam(net.parameters(), lr=K.LR)

    def loss(out, yy):
        return Fn.cross_entropy(out[:, :4], yy, weight=w) + K.LAM_ADD * Fn.binary_cross_entropy_with_logits(
            out[:, 4], (yy == 3).float())
    best, best_step, state = np.inf, -1, None
    for step in range(K.MAX_STEPS):
        opt.zero_grad()
        l = loss(net(X[i_fit]), yt[i_fit])
        l.backward()
        opt.step()
        with torch.no_grad():
            hl = float(loss(net(X[i_ho]), yt[i_ho]))
        if hl < best:
            best, best_step = hl, step
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        elif step - best_step >= K.PATIENCE:
            break
    net.load_state_dict(state)
    with torch.no_grad():
        return torch.softmax(net(Xe)[:, :4], 1).numpy().astype(np.float64), best_step


def mlp_ens(X, y, Xe, h):
    ps = [fit_mlp(X, y, Xe, h, s) for s in MSEEDS]
    return np.mean([p for p, _ in ps], 0), [b for _, b in ps]


def job(arg):
    seed, fold = arg
    trd, ted, trdf, _ = K.unit_data(seed, fold)
    for d in (trd, ted):
        d["extras"] = trdf.iloc[d["rows"]][CFG["extras"]].to_numpy(dtype=float)
    assert not set(trd["rows"]) & set(ted["rows"])
    y = trd["y"]
    # inner CV for h (outer-train rows only)
    inner = {h: [] for h in HS}
    for ia, ib in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        Xa, Xb = design(sub(trd, ia), sub(trd, ib))
        for h in HS:
            p, _ = mlp_ens(Xa, y[ia], Xb, h)
            inner[h].append(C.M.metrics(y[ib], p)["macro_f1"])
    score = {h: float(np.mean(v)) for h, v in inner.items()}
    h_star = max(HS, key=lambda h: (round(score[h], 12), -h))
    X, Xe = design(trd, ted)
    p_lr = fit_lr(X, y, Xe)
    p_mlp, steps = mlp_ens(X, y, Xe, h_star)
    return seed, fold, ted["rows"], p_lr, p_mlp, h_star, score, steps


def label(lo, hi, m):
    out = []
    if lo > -m:
        out.append("stacker non-inferior to KAN")
    if lo > 0:
        out.append("stacker better")
    if hi < 0:
        out.append("KAN better")
    if hi < -m:
        out.append("stacker inferior beyond margin")
    return out or ["no difference resolved"]


def main():
    pv = C.Prov("B1_stackers", __file__)
    for p in ("streams.npz", "kan_runs/32.npz", "kan_fusion.py", "metrics.py"):
        pv.file(os.path.join(C.V3, p))
    pv.file(os.path.join(C.KF, "fusion_trainval.csv"))
    init()
    tr = K.DATA["train"]
    y = tr.label.to_numpy()
    pv.check("streams train order", (K.ST["train_rids"] == tr.RID.to_numpy()).all())
    kan = np.load(os.path.join(C.V3, "kan_runs", "32.npz"))["cv"]
    kf1 = float(C.f1_seeds(y, kan).mean())
    pv.check("KAN config 32 CV macro-F1 == 0.7562", abs(kf1 - 0.7562) < 5e-5, 0.7562, round(kf1, 5))
    # folds in streams.npz are the paper's StratifiedKFold(5, shuffle, random_state=s)
    for s in C.SEEDS:
        for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(tr, y)):
            assert (K.ST[f"idx_test_{s}_{f}"] == b).all()
    pv.check("streams.npz fold indices == StratifiedKFold(5, shuffle, seed)", True)

    P = {"LR": np.full((5, len(y), 4), np.nan), "MLP": np.full((5, len(y), 4), np.nan)}
    hsel, t0 = {}, time.time()
    with Pool(15, initializer=init) as pool:
        for seed, fold, rows, plr, pmlp, h, score, steps in pool.imap_unordered(
                job, [(s, f) for s in C.SEEDS for f in range(5)]):
            si = C.SEEDS.index(seed)
            P["LR"][si, rows], P["MLP"][si, rows] = plr, pmlp
            hsel[f"{seed}_{fold}"] = {"h": h, "inner_macro_f1": score, "best_steps": steps}
            print(f"seed {seed} fold {fold}: h* {h} inner {score} ({time.time() - t0:.0f}s)", flush=True)
    for k in P:
        pv.check(f"{k}: every subject predicted in every seed", np.isfinite(P[k]).all())
    os.makedirs(C.PRIV, exist_ok=True)
    np.savez(os.path.join(C.PRIV, "b1_oof.npz"), lr=P["LR"], mlp=P["MLP"])

    idx = C.boot_idx(len(y))
    sysP = {"KAN K2 (config 32, stored)": kan, "LR stacker": P["LR"], "MLP stacker": P["MLP"]}
    res = {"systems": {}, "comparisons": {}, "h_selected": hsel}
    bf = {k: C.f1_boot(y, v, idx) for k, v in sysP.items()}
    ba = {k: C.auc_boot(y, v, idx) for k, v in sysP.items()}
    for k, v in sysP.items():
        f1s, aucs = C.f1_seeds(y, v), C.auc_seeds(y, v)
        res["systems"][k] = {"macro_f1": float(f1s.mean()), "macro_f1_sd": float(f1s.std(ddof=1)),
                             "macro_f1_ci": C.ci(bf[k]), "ad_auc": float(aucs.mean()), "ad_auc_ci": C.ci(ba[k]),
                             "per_seed_macro_f1": f1s.tolist(), **{x: float(np.mean([C.M.metrics(y, p)[x] for p in v]))
                                                                  for x in ("bal_acc", "acc", "f1_LMCI", "f1_AD", "logloss")}}
    kref = "KAN K2 (config 32, stored)"
    for k in ("LR stacker", "MLP stacker"):
        res["comparisons"][k] = {}
        for met, b in (("macro_f1", bf), ("ad_auc", ba)):
            d = res["systems"][k][met] - res["systems"][kref][met]
            c = C.ci(b[k] - b[kref])
            res["comparisons"][k][met] = {"delta": d, "ci": c, "margin": MARGIN[met], "labels": label(*c, MARGIN[met])}
    outs = [C.write_json(res, os.path.join(C.OUT, "json", "B1.json"))]
    f = os.path.join(C.OUT, "csv", "B1_fusion_baselines.csv")
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "cv_macro_f1", "ci_lo", "ci_hi", "sd_over_seeds", "cv_ad_auc", "auc_ci_lo", "auc_ci_hi",
                    "delta_f1_vs_kan", "d_f1_ci_lo", "d_f1_ci_hi", "f1_rule (margin 0.02)", "delta_auc_vs_kan",
                    "d_auc_ci_lo", "d_auc_ci_hi", "auc_rule (margin 0.01)"])
        for k, s in res["systems"].items():
            row = [k, f"{s['macro_f1']:.4f}", f"{s['macro_f1_ci'][0]:.4f}", f"{s['macro_f1_ci'][1]:.4f}",
                   f"{s['macro_f1_sd']:.4f}", f"{s['ad_auc']:.4f}", f"{s['ad_auc_ci'][0]:.4f}", f"{s['ad_auc_ci'][1]:.4f}"]
            c = res["comparisons"].get(k)
            if c:
                for met in ("macro_f1", "ad_auc"):
                    row += [f"{c[met]['delta']:+.4f}", f"{c[met]['ci'][0]:+.4f}", f"{c[met]['ci'][1]:+.4f}",
                            "; ".join(c[met]["labels"])]
            else:
                row += ["reference"] + [""] * 7
            w.writerow(row)
    outs.append(f)
    f = os.path.join(C.OUT, "csv", "B1_mlp_hidden_size.csv")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["outer_seed", "outer_fold", "h_selected"] + [f"inner_macro_f1_h{h}" for h in HS])
        for k in sorted(hsel):
            s, fo = k.split("_")
            w.writerow([s, fo, hsel[k]["h"]] + [f"{hsel[k]['inner_macro_f1'][h]:.4f}" for h in HS])
    outs.append(f)
    pv.note("MLP loss/training per prereg B1: no EMD ordinal term and no modality dropout (not pre-registered for "
            "the MLP); KAN keeps its own (config 32).")
    pv.write(outs)
    print(json.dumps({k: {m: round(v[m], 4) for m in ("macro_f1", "ad_auc")} for k, v in res["systems"].items()}))
    print(json.dumps(res["comparisons"], indent=1))


if __name__ == "__main__":
    main()
