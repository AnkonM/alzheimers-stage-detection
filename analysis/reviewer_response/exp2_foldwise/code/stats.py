"""[reviewer-response B2 copy] Source: 880f4a7:experiments/exp2_reduced_clinical/code/stats.py (unchanged; paths via common.py).

EXP2 metrics and paired bootstrap (numpy/sklearn only).

Four-stage: macro-F1, balanced accuracy, accuracy, per-class F1, mean stage-distance
error, log-loss. Binary: AD-vs-rest AUC (p_AD); CN-vs-MCI AUC on y in {CN, EMCI, LMCI}
with (p_E + p_L) / (p_C + p_E + p_L); EMCI-vs-LMCI AUC on true MCI with p_L / (p_E + p_L).
The confusion / F1 formulas are those of analysis/fusion_v3/metrics.py.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

NAMES = ["CN", "EMCI", "LMCI", "AD"]
N_BOOT = 2000


def confusion(y, p, k=4):
    return np.bincount(k * np.asarray(y) + np.asarray(p), minlength=k * k).reshape(k, k)


def f1s(cm):
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(len(tp)), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(len(tp)), where=cm.sum(1) > 0)
    return np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(tp)), where=(prec + rec) > 0), rec


def metrics_labels(y, p):
    f1, rec = f1s(confusion(y, p))
    return {"macro_f1": float(f1.mean()), "bal_acc": float(rec.mean()), "acc": float(np.mean(p == y)),
            **{f"f1_{n}": float(f1[i]) for i, n in enumerate(NAMES)}}


def ratio(num, den):
    """num / den; 0.5 (a tie) where den == 0 (e.g. a tree giving p_E = p_L = 0)."""
    return np.divide(num, den, out=np.full(len(num), 0.5), where=den > 0)


def binary_aucs(y, P):
    out = {}
    for k, fn in AUC_FN.items():
        t, s, m = fn(y, P)
        out[k] = roc_auc_score(t[m], s[m])
    return {k: float(v) for k, v in out.items()}


def metrics(y, P):
    p = P.argmax(1)
    out = metrics_labels(y, p)
    out["stage_dist"] = float(np.abs(p - y).mean())
    out["logloss"] = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))
    out.update(binary_aucs(y, P))
    return out


def boot_idx(n, seed=0):
    return np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))


def _macro_f1_boot(y, p, idx, present=False):
    codes = 4 * y[idx] + p[idx]
    cms = np.stack([np.bincount(c, minlength=16) for c in codes]).reshape(-1, 4, 4)
    tp = np.diagonal(cms, axis1=1, axis2=2).astype(float)
    ps, rs = cms.sum(1), cms.sum(2)
    prec = np.divide(tp, ps, out=np.zeros_like(tp), where=ps > 0)
    rec = np.divide(tp, rs, out=np.zeros_like(tp), where=rs > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    if present:                                   # E1: mean over true classes present in each resample
        return (f1 * (rs > 0)).sum(1) / (rs > 0).sum(1)
    return f1.mean(1)


def macro_f1_present(y, p):
    """E1: macro-F1 averaged over the true classes present in y."""
    f1, _ = f1s(confusion(y, p))
    return float(f1[np.bincount(y, minlength=4) > 0].mean())


AUC_FN = {
    "ad_auc": lambda y, P: (y == 3, P[:, 3], np.ones(len(y), bool)),
    "cn_mci_auc": lambda y, P: (y > 0, ratio(P[:, 1] + P[:, 2], P[:, :3].sum(1)), y < 3),
    "emci_lmci_auc": lambda y, P: (y == 2, ratio(P[:, 2], P[:, 1] + P[:, 2]), (y == 1) | (y == 2)),
}


_W_CACHE = {}


def resample_weights(idx):
    """(B, n) multiplicity of each subject in each resample."""
    key = (idx.shape, idx[:2, :5].tobytes())
    if key not in _W_CACHE:
        _W_CACHE[key] = np.stack([np.bincount(r, minlength=idx.shape[1]) for r in idx]).astype(float)
    return _W_CACHE[key]


def boot_auc(t, s, m, W):
    """AUC of score s for target t on rows m, in every resample given by weights W (B, n).
    Weighted Mann-Whitney with ties = 0.5: identical to roc_auc_score on the resampled rows.
    Returns (auc for valid resamples, valid mask); a resample is invalid without both classes."""
    t, s, Wm = t[m], s[m], W[:, m]
    _, inv = np.unique(s, return_inverse=True)
    G = np.zeros((len(s), inv.max() + 1))
    G[np.arange(len(s)), inv] = 1
    Gp, Gn = (Wm * t) @ G, (Wm * ~t) @ G
    below = np.cumsum(Gn, 1) - Gn
    num, den = (Gp * (below + 0.5 * Gn)).sum(1), Gp.sum(1) * Gn.sum(1)
    valid = den > 0
    return num[valid] / den[valid], valid


def paired_delta_loop(y, Pa, Pb, idx, metric):
    """Reference implementation (per-resample roc_auc_score); used only to test boot_auc."""
    t, sa, m = AUC_FN[metric](y, Pa)
    _, sb, _ = AUC_FN[metric](y, Pb)
    d = []
    for r in idx:
        rm = r[m[r]]
        tt = t[rm]
        if tt.all() or not tt.any():
            continue
        d.append(roc_auc_score(tt, sa[rm]) - roc_auc_score(tt, sb[rm]))
    return np.asarray(d)


def paired_delta(y, Pa, Pb, idx, metric, present=False):
    """Seed-averaged probabilities Pa, Pb (n, 4). Returns point Δ (a − b), 95% percentile CI,
    number of resamples dropped (undefined AUC)."""
    if metric == "macro_f1":
        pa, pb = Pa.argmax(1), Pb.argmax(1)
        if present:
            point = macro_f1_present(y, pa) - macro_f1_present(y, pb)
        else:
            point = metrics_labels(y, pa)["macro_f1"] - metrics_labels(y, pb)["macro_f1"]
        d = _macro_f1_boot(y, pa, idx, present) - _macro_f1_boot(y, pb, idx, present)
        dropped = 0
    else:
        t, sa, m = AUC_FN[metric](y, Pa)
        _, sb, _ = AUC_FN[metric](y, Pb)
        if t[m].all() or not t[m].any():
            return {"delta": None, "lo": None, "hi": None, "dropped": len(idx)}
        point = roc_auc_score(t[m], sa[m]) - roc_auc_score(t[m], sb[m])
        W = resample_weights(idx)
        a, va = boot_auc(t, sa, m, W)
        b, _ = boot_auc(t, sb, m, W)
        d, dropped = a - b, int((~va).sum())
    return {"delta": float(point), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "dropped": int(dropped)}


def paper_style_delta(y, Pa_seeds, Pb_seeds, idx):
    """analysis/fusion_v3/evaluate.py: mean over seeds of per-seed macro-F1 Δ; CI from the
    seed-mean of per-seed bootstrap Δ."""
    pa, pb = Pa_seeds.argmax(-1), Pb_seeds.argmax(-1)
    point = np.mean([metrics_labels(y, pa[s])["macro_f1"] - metrics_labels(y, pb[s])["macro_f1"] for s in range(len(pa))])
    d = np.mean([_macro_f1_boot(y, pa[s], idx) - _macro_f1_boot(y, pb[s], idx) for s in range(len(pa))], 0)
    return {"delta": float(point), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
