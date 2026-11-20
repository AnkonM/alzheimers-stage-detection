"""[reviewer-response B2 copy] Source: 880f4a7:experiments/exp2_reduced_clinical/code/phase1_label_audit.py (unchanged; paths via common.py).

Phase 1 — label-definition audit (exploratory; LYP/.venv-fusion). Train/val only.

1. Univariate: each candidate column, 1-D logistic per one-vs-rest class, in-fold
   median imputation + z-score, repeated stratified 5-fold CV (seeds 42-46, the
   paper folds), AUC of seed-averaged OOF probabilities; Spearman rho with stage
   on raw train values.
2. Reconstructability: depth-3 tree and multinomial logistic on {MMSE, CDGLOBAL,
   LDELTOTL, PTEDUCAT}; same CV; seed-averaged OOF -> macro-F1, confusion matrix.
3. Completeness: train subjects with every label-rule variable.
4. ADNI-GO/2 enrolment cut-offs as a rule-based classifier on train+val (691).
Outputs: tables/phase1_*.csv, logs/phase1.json.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import stats as ST  # noqa: E402

RULE_VARS = ["MMSE", "CDGLOBAL", "LDELTOTL", "PTEDUCAT"]
CANDIDATES = (C.T3 + ["APOE4", "MMSE", "CDRSB", "CDGLOBAL", "CDMEMORY", "LDELTOTL", "LIMMTOTL", "MPACCTRAILSB",
                      "CogIndex"])


def zfit(X):
    med = np.nanmedian(X, 0)
    Xi = np.where(np.isnan(X), med, X)
    mu, sd = Xi.mean(0), Xi.std(0, ddof=1)
    sd[sd == 0] = 1
    return lambda Z: (np.where(np.isnan(Z), med, Z) - mu) / sd


def cv_proba(X, y, make):
    P = np.zeros((len(C.SEEDS), len(y), 4))
    for si, s in enumerate(C.SEEDS):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            f = zfit(X[a])
            m = make(s).fit(f(X[a]), y[a])
            P[si, b] = m.predict_proba(f(X[b]))
    return P.mean(0)


def univariate(raw, y):
    rows = []
    for c in CANDIDATES:
        col = (raw[c] == "Male").astype(float).where(raw[c].notna()) if c == "PTGENDER" else raw[c]
        x = col.to_numpy(dtype=float)[:, None]
        r = {"column": c, "tag": "excluded" if c in C.EXCLUDED else "eligible", "train_missing": float(np.isnan(x).mean())}
        ok = ~np.isnan(x[:, 0])
        r["spearman_stage"] = float(spearmanr(x[ok, 0], y[ok]).statistic)
        for k, n in enumerate(C.NAMES):
            yk = (y == k).astype(int)
            P = np.zeros(len(y))
            for s in C.SEEDS:
                for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(x, y):
                    f = zfit(x[a])
                    P[b] += LogisticRegression(max_iter=5000).fit(f(x[a]), yk[a]).predict_proba(f(x[b]))[:, 1] / len(C.SEEDS)
            r[f"auc_{n}_vs_rest"] = float(roc_auc_score(yk, P))
        rows.append(r)
    return pd.DataFrame(rows)


def band(edu):
    return np.where(edu >= 16, 2, np.where(edu >= 8, 1, 0))


def rule_classifier(df):
    """ADNI-GO/2 enrolment cut-offs (user-supplied; ADNI-2 protocol ADC-039). Returns predictions
    with -1 = ambiguous (several classes match), -2 = unmatched."""
    mm, cdr, lm, b = df.MMSE.to_numpy(), df.CDGLOBAL.to_numpy(), df.LDELTOTL.to_numpy(), band(df.PTEDUCAT.to_numpy())
    mci_mmse = (mm >= 24) & (mm <= 30)
    cn_lo = np.array([3, 5, 9])[b]
    em_lo, em_hi = np.array([3, 5, 9])[b], np.array([6, 9, 11])[b]
    lm_hi = np.array([2, 4, 8])[b]
    match = np.stack([
        mci_mmse & (cdr == 0) & (lm >= cn_lo),
        mci_mmse & (cdr == 0.5) & (lm >= em_lo) & (lm <= em_hi),
        mci_mmse & (cdr == 0.5) & (lm <= lm_hi),
        (mm >= 20) & (mm <= 26) & np.isin(cdr, [0.5, 1.0]),
    ], 1)
    n = match.sum(1)
    pred = np.where(n == 1, match.argmax(1), np.where(n > 1, -1, -2))
    return pred, match


def f1_with_other(y, p):
    """Macro-F1 over the 4 classes; predictions outside 0-3 count as errors (FN only)."""
    f = []
    for k in range(4):
        tp = np.sum((p == k) & (y == k))
        fp, fn = np.sum((p == k) & (y != k)), np.sum((p != k) & (y == k))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)), f


def main():
    data = C.load()
    (tr, rtr), (va, rva) = data["train"], data["val"]
    y = tr.label.to_numpy()
    raw = rtr.copy()
    raw["CogIndex"] = tr.CogIndex.to_numpy()          # paper step-4 CogIndex (train-fitted), audit only
    out = {}

    uni = univariate(raw, y)
    uni.to_csv(os.path.join(C.TABLES, "phase1_univariate.csv"), index=False)

    X = rtr[RULE_VARS].to_numpy(dtype=float)
    rec = {}
    for name, make in (("tree_depth3", lambda s: DecisionTreeClassifier(max_depth=3, random_state=s)),
                       ("logistic", lambda s: LogisticRegression(C=1.0, max_iter=5000))):
        P = cv_proba(X, y, make)
        m = ST.metrics(y, P)
        cm = ST.confusion(y, P.argmax(1))
        rec[name] = {"macro_f1": m["macro_f1"], "acc": m["acc"], "bal_acc": m["bal_acc"],
                     **{f"f1_{n}": m[f"f1_{n}"] for n in C.NAMES}, "confusion": cm.tolist()}
        pd.DataFrame(cm, index=[f"true_{n}" for n in C.NAMES], columns=[f"pred_{n}" for n in C.NAMES]).to_csv(
            os.path.join(C.TABLES, f"phase1_reconstruct_{name}_confusion.csv"))
    out["reconstruct"] = {"inputs": RULE_VARS, "n": int(len(y)), **rec}
    out["complete_label_vars_train"] = int(rtr[RULE_VARS].notna().all(1).sum())
    out["complete_label_vars_train_per_class"] = rtr[rtr[RULE_VARS].notna().all(1)].label.value_counts().sort_index().tolist()

    allr = pd.concat([rtr, rva], ignore_index=True)
    ya = allr.label.to_numpy()
    assert allr[RULE_VARS].notna().all().all()
    pred, match = rule_classifier(allr)
    uniq = pred >= 0
    cm = ST.confusion(ya[uniq], pred[uniq])
    mf_all, f1s = f1_with_other(ya, pred)
    mu = ST.metrics_labels(ya[uniq], pred[uniq])
    per = pd.DataFrame({"true": [C.NAMES[k] for k in ya], "pred": pred})
    extra = pd.crosstab(per.true, per.pred.map(lambda v: "ambiguous" if v == -1 else ("unmatched" if v == -2 else "matched")))
    out["rule_based"] = {
        "n": int(len(ya)), "uniquely_matched": int(uniq.sum()), "ambiguous": int((pred == -1).sum()),
        "unmatched": int((pred == -2).sum()),
        "acc_all": float(np.mean(pred == ya)), "macro_f1_all": mf_all, "f1_all": f1s,
        "acc_matched": mu["acc"], "macro_f1_matched": mu["macro_f1"],
        "confusion_matched": cm.tolist(), "status_by_true_class": extra.reindex(C.NAMES).fillna(0).astype(int).to_dict("index"),
        "ambiguous_patterns": pd.Series([tuple(np.flatnonzero(m)) for m in match[pred == -1]]).value_counts()
                                .rename(lambda t: "+".join(C.NAMES[i] for i in t)).to_dict(),
        "cdmemory_rule_check": {  # protocol also requires CDR memory box 0 (CN) / >= 0.5 (MCI); not in the user's rule
            "CN_label_with_memory_box_gt0": int(((ya == 0) & (allr.CDMEMORY > 0)).sum()),
            "MCI_label_with_memory_box_0": int((np.isin(ya, [1, 2]) & (allr.CDMEMORY == 0)).sum())},
    }
    pd.DataFrame(cm, index=[f"true_{n}" for n in C.NAMES], columns=[f"pred_{n}" for n in C.NAMES]).to_csv(
        os.path.join(C.TABLES, "phase1_rule_based_confusion.csv"))
    C.write_json(out, os.path.join(C.LOGS, "phase1.json"))
    pd.set_option("display.width", 220)
    print(uni.round(3).to_string(index=False))
    import json
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    os.makedirs(C.TABLES, exist_ok=True)
    main()
