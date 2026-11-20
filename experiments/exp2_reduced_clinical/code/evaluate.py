"""Phase 4 — metrics, paired Δ, rules R1–R5, E1, tables, figures (LYP/.venv-fusion +
PYTHONPATH=LYP/.pylib-exp2). Train CV and val only. Reads oof/ and logs/; writes
results.json, tables/*.csv, figures/*.png. Implements PREREGISTRATION.md §3 and the addendum.
"""
import collections
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import stats as ST  # noqa: E402
from phase1_label_audit import rule_classifier  # noqa: E402

TIERS = ["T0", "T1", "T2", "T3", "Full-spec", "S-T2+APOE4", "Full-paper"]
SYSTEMS = ["C", "M", "F-KAN", "F-CB", "F-KAN-noMRI", "F-KAN-mask-inf"]
METRICS4 = ["macro_f1", "bal_acc", "acc", "f1_CN", "f1_EMCI", "f1_LMCI", "f1_AD", "stage_dist", "logloss",
            "ad_auc", "cn_mci_auc", "emci_lmci_auc"]
AUCS = ["ad_auc", "emci_lmci_auc", "cn_mci_auc"]
CV_COMPS = [("F-KAN", "C"), ("F-CB", "C"), ("F-KAN", "F-KAN-noMRI"), ("F-KAN", "F-KAN-mask-inf"), ("F-KAN-noMRI", "C")]
VAL_COMPS = [("F-KAN", "C"), ("F-CB", "C")]


def load():
    data = C.load()
    (tr, rtr), (va, rva) = data["train"], data["val"]
    mri = np.load(os.path.join(C.OOF, "streams_MRI.npz"))
    P = {"cv": {}, "val": {}}
    for t in TIERS:
        st = np.load(os.path.join(C.OOF, f"streams_{t}.npz"))
        kn = np.load(os.path.join(C.OOF, f"kan_{t}.npz"))
        for k in ("train_rids", "val_rids"):
            assert (st[k] == mri[k]).all()
        assert (st["train_rids"] == tr.RID.to_numpy()).all() and (st["val_rids"] == va.RID.to_numpy()).all()
        P["cv"][t] = {"C": st["clin_oof"], "M": mri["mri_oof"], "F-CB": st["fcb_oof"],
                      **{s: kn[f"{s}__cv"] for s in ("F-KAN", "F-KAN-noMRI", "F-KAN-mask-inf")}}
        P["val"][t] = {"C": st["clin_val"], "M": mri["mri_val"], "F-CB": st["fcb_val"],
                       **{s: kn[f"{s}__val"] for s in ("F-KAN", "F-KAN-noMRI", "F-KAN-mask-inf")}}
    return tr, rtr, va, rva, P


def r4(x):
    return None if x is None else round(float(x), 4)


def main():
    tr, rtr, va, rva, P = load()
    y = {"cv": tr.label.to_numpy(), "val": va.label.to_numpy()}
    has = {"cv": tr.has_mri.to_numpy(), "val": va.has_mri.to_numpy()}
    anomalies = []

    # NaN checks
    for ev in P:
        for t in TIERS:
            for s, A in P[ev][t].items():
                rows = has[ev] if s == "M" else np.ones(len(y[ev]), bool)
                assert not np.isnan(A[:, rows]).any(), f"NaN in {ev} {t} {s}"
                assert np.allclose(A[:, rows].sum(-1), 1, atol=1e-5), f"rows not summing to 1: {ev} {t} {s}"

    # point metrics (seed-averaged) and per-seed macro-F1
    met, seeds, rows_csv = {"cv": {}, "val": {}}, {}, []
    for ev in ("cv", "val"):
        for t in TIERS:
            met[ev][t] = {}
            for s in SYSTEMS:
                mk = has[ev] if s == "M" else np.ones(len(y[ev]), bool)
                A = P[ev][t][s][:, mk]
                m = ST.metrics(y[ev][mk], A.mean(0))
                ps = [ST.metrics_labels(y[ev][mk], a.argmax(1))["macro_f1"] for a in A]
                m.update({"n": int(mk.sum()), "paper_style_macro_f1": float(np.mean(ps)),
                          "seed_macro_f1_min": float(min(ps)), "seed_macro_f1_max": float(max(ps))})
                met[ev][t][s] = m
                rows_csv.append({"eval": ev, "tier": t, "system": s, **{k: r4(v) if isinstance(v, float) else v
                                                                         for k, v in m.items()}})
    pd.DataFrame(rows_csv).to_csv(os.path.join(C.TABLES, "metrics_all.csv"), index=False)

    # paired deltas
    idx = {"cv": ST.boot_idx(len(y["cv"])), "val": ST.boot_idx(len(y["val"])),
           "cv_scan": ST.boot_idx(int(has["cv"].sum())), "val_scan": ST.boot_idx(int(has["val"].sum()))}
    deltas, drows = {"cv": {}, "val": {}}, []
    for ev, comps in (("cv", CV_COMPS), ("val", VAL_COMPS)):
        for t in TIERS:
            deltas[ev][t] = {}
            for a, b in comps:
                key = f"{a} - {b}"
                deltas[ev][t][key] = {}
                for metric in ["macro_f1"] + AUCS:
                    d = ST.paired_delta(y[ev], P[ev][t][a].mean(0), P[ev][t][b].mean(0), idx[ev], metric)
                    deltas[ev][t][key][metric] = d
                    drows.append({"eval": ev, "tier": t, "comparison": key, "metric": metric,
                                  **{k: r4(v) if k != "dropped" else v for k, v in d.items()}})
                    if d["dropped"]:
                        anomalies.append(f"{ev} {t} {key} {metric}: {d['dropped']} bootstrap resamples dropped")
    # M − C on scan subjects (R4 and descriptive), all tiers
    for ev in ("cv", "val"):
        mk = has[ev]
        for t in TIERS:
            deltas[ev][t]["M - C (scan subjects)"] = {}
            for metric in ["macro_f1"] + AUCS:
                d = ST.paired_delta(y[ev][mk], P[ev][t]["M"][:, mk].mean(0), P[ev][t]["C"][:, mk].mean(0),
                                    idx[f"{ev}_scan"], metric)
                deltas[ev][t]["M - C (scan subjects)"][metric] = d
                drows.append({"eval": ev, "tier": t, "comparison": "M - C (scan subjects)", "metric": metric,
                              **{k: r4(v) if k != "dropped" else v for k, v in d.items()}})
    # paper-style Δ (C5) at Full-paper and Full-spec
    paper_style = {}
    for t in ("Full-paper", "Full-spec"):
        paper_style[t] = {"C": met["cv"][t]["C"]["paper_style_macro_f1"],
                          "F-KAN": met["cv"][t]["F-KAN"]["paper_style_macro_f1"],
                          "delta": ST.paper_style_delta(y["cv"], P["cv"][t]["F-KAN"], P["cv"][t]["C"], idx["cv"])}
    pd.DataFrame(drows).to_csv(os.path.join(C.TABLES, "deltas_all.csv"), index=False)

    # rules
    def r12(t, metric):
        cv, vd = deltas["cv"][t]["F-KAN - C"][metric], deltas["val"][t]["F-KAN - C"][metric]
        return {"cv_delta": cv["delta"], "cv_lo": cv["lo"], "cv_hi": cv["hi"], "val_delta": vd["delta"],
                "met": bool(cv["lo"] > 0 and vd["delta"] > 0)}
    rules = {"R1": {t: r12(t, "macro_f1") for t in C.TREND + ["S-T2+APOE4"]},
             "R2": {t: r12(t, "ad_auc") for t in C.TREND + ["S-T2+APOE4"]}}
    dvec = [deltas["cv"][t]["F-KAN - C"]["macro_f1"]["delta"] for t in C.TREND]
    rho = float(spearmanr(range(5), dvec).statistic)
    rules["R3"] = {"tiers": C.TREND, "delta_macro_f1": dvec, "spearman": rho, "met": bool(rho <= -0.8),
                   "note": "5 points; descriptive"}
    d4 = deltas["cv"]["T0"]["M - C (scan subjects)"]["ad_auc"]
    rules["R4"] = {"M_ad_auc": met["cv"]["T0"]["M"]["ad_auc"],
                   "C_ad_auc_scan": ST.metrics(y["cv"][has["cv"]], P["cv"]["T0"]["C"][:, has["cv"]].mean(0))["ad_auc"],
                   **d4, "met": bool(d4["lo"] > 0)}
    r5 = {}
    for rule, metric in (("R1", "macro_f1"), ("R2", "ad_auc")):
        for t, r in rules[rule].items():
            if r["met"]:
                fcb = deltas["cv"][t]["F-CB - C"][metric]["delta"]
                same = np.sign(fcb) == np.sign(r["cv_delta"])
                r5[f"{rule} {t}"] = {"fkan_delta": r["cv_delta"], "fcb_delta": fcb,
                                     "verdict": "MET" if same else "KAN-specific"}
    rules["R5"] = r5 if r5 else {"note": "not applicable: no tier met R1 or R2"}

    # E1 — rule-ambiguity subsets
    e1 = {"composition": {}, "cv": {}, "val": {}}
    subsets = {}
    for ev, raw, yy in (("cv", rtr, y["cv"]), ("val", rva, y["val"])):
        pred, match = rule_classifier(raw)
        amb = pred == -1
        mm, cdr = raw.MMSE.to_numpy(), raw.CDGLOBAL.to_numpy()
        assert ((cdr[amb] == 0.5) & (mm[amb] >= 24) & (mm[amb] <= 26)).all()
        assert (match[amb, 3] & (match[amb, 1] | match[amb, 2])).all() and (match[amb].sum(1) == 2).all()
        subsets[ev] = {"AMBIGUOUS": amb, "UNAMBIGUOUS": pred >= 0, "UNMATCHED": pred == -2}
        e1["composition"][ev] = {k: {"n": int(v.sum()), **{C.NAMES[c]: int(((yy == c) & v).sum()) for c in range(4)},
                                     "n_with_scan": int((v & has[ev]).sum()),
                                     "rule_agrees_with_label": (int((pred[v] == yy[v]).sum()) if k == "UNAMBIGUOUS" else None)}
                                 for k, v in subsets[ev].items()}
    e1rows = []
    for ev in ("cv", "val"):
        for sub in ("AMBIGUOUS", "UNAMBIGUOUS"):
            mask = subsets[ev][sub]
            n = int(mask.sum())
            with_ci = ev == "cv" or n >= 30
            bidx = ST.boot_idx(n)
            e1[ev][sub] = {}
            for t in TIERS:
                o = {}
                for s in ("C", "M", "F-KAN", "F-CB"):
                    mk = mask & has[ev] if s == "M" else mask
                    yy, PP = y[ev][mk], P[ev][t][s][:, mk].mean(0)
                    pr = PP.argmax(1)
                    auc = (float(ST.roc_auc_score(yy == 3, PP[:, 3])) if 0 < (yy == 3).sum() < len(yy) else None)
                    o[s] = {"n": int(mk.sum()), "acc": float((pr == yy).mean()),
                            "macro_f1_present": ST.macro_f1_present(yy, pr), "ad_auc": auc}
                yy = y[ev][mask]
                A, B = P[ev][t]["F-KAN"][:, mask].mean(0), P[ev][t]["C"][:, mask].mean(0)
                for metric in ("macro_f1", "ad_auc"):
                    d = ST.paired_delta(yy, A, B, bidx, metric, present=True)
                    if not with_ci:
                        d = {"delta": d["delta"], "lo": None, "hi": None, "dropped": None}
                    o[f"delta_FKAN_C_{metric}"] = d
                e1[ev][sub][t] = o
                e1rows.append({"eval": ev, "subset": sub, "tier": t, "n": n,
                               **{f"{s}_{k}": r4(v) if isinstance(v, float) else v for s in ("C", "M", "F-KAN", "F-CB")
                                  for k, v in o[s].items()},
                               **{f"dFKAN_C_{m}_{k}": r4(v) if isinstance(v, float) else v
                                  for m in ("macro_f1", "ad_auc") for k, v in o[f"delta_FKAN_C_{m}"].items()}})
    e1["amb_minus_unamb_cv"] = {t: {m: (None if e1["cv"]["AMBIGUOUS"][t][f"delta_FKAN_C_{m}"]["delta"] is None else
                                        e1["cv"]["AMBIGUOUS"][t][f"delta_FKAN_C_{m}"]["delta"]
                                        - e1["cv"]["UNAMBIGUOUS"][t][f"delta_FKAN_C_{m}"]["delta"])
                                    for m in ("macro_f1", "ad_auc")} for t in TIERS}
    pd.DataFrame(e1rows).to_csv(os.path.join(C.TABLES, "e1_ambiguity_subsets.csv"), index=False)

    # confusion matrices
    cms = {}
    for t in ("T1", "T2", "Full-spec", "Full-paper"):
        cms[t] = {s: ST.confusion(y["cv"], P["cv"][t][s].mean(0).argmax(1)).tolist() for s in ("C", "F-KAN", "F-CB")}
        for s in ("C", "F-KAN"):
            pd.DataFrame(cms[t][s], index=[f"true_{n}" for n in C.NAMES], columns=[f"pred_{n}" for n in C.NAMES]).to_csv(
                os.path.join(C.TABLES, f"confusion_cv_{t}_{s}.csv"))

    # run diagnostics: feature selection, KAN fits, warnings, runtime
    diag = {"selection": {}, "kan": {}, "runtime_s": {}, "warnings": {}}
    for t in TIERS:
        sm = json.load(open(os.path.join(C.OOF, f"streams_{t}_meta.json")))
        km = json.load(open(os.path.join(C.OOF, f"kan_{t}_meta.json")))
        units = {k: v for k, v in sm.items() if isinstance(v, dict)}
        if t in C.REDUCED:
            for u, v in units.items():
                bad = set(v["top"] + v["extras"]) & C.EXCLUDED
                assert not bad, f"excluded {bad} in {t} {u}"
        ext = collections.Counter(c for v in units.values() for c in v["extras"])
        top = collections.Counter(c for v in units.values() for c in v["top"])
        ind = sorted({c for v in units.values() for c in v["indicators"]})
        diag["selection"][t] = {"n_units": len(units), "extras_freq": dict(ext.most_common()),
                                "top_freq": dict(top.most_common()), "missingness_indicators": ind}
        fails = sum(d["fails"] for d in km["diag"])
        steps = [s for d in km["diag"] for s in d["best_steps"]]
        diag["kan"][t] = {"jobs": len(km["diag"]), "kan_fits": 5 * len(km["diag"]), "failed_fits": fails,
                          "best_step_min": min(steps), "best_step_max": max(steps),
                          "n_in": sorted({d["n_in"] for d in km["diag"]})}
        if fails:
            anomalies.append(f"{t}: {fails} failed KAN fits")
        if ind:
            anomalies.append(f"{t}: missingness indicators created: {ind}")
        diag["runtime_s"][t] = {"streams": sm["runtime_s"], "kan": km["runtime_s"]}
    for f in sorted(glob.glob(os.path.join(C.LOGS, "phase3_*.out"))):
        txt = open(f).read()
        w = collections.Counter(m.group(1) for m in re.finditer(r"(\w+Warning)", txt))
        if w:
            diag["warnings"][os.path.basename(f)] = dict(w)
    cnt = collections.Counter(l.split("\t")[1] for l in open(os.path.join(C.LOGS, "assertions.log")))
    diag["assertions"] = dict(cnt)

    res = {"metrics": met, "deltas": deltas, "paper_style": paper_style, "rules": rules, "e1": e1,
           "confusion_cv": cms, "diagnostics": diag, "anomalies": anomalies,
           "gate": json.load(open(os.path.join(C.LOGS, "gate.json"))),
           "phase1": json.load(open(os.path.join(C.LOGS, "phase1.json"))), "n_boot": ST.N_BOOT}
    C.write_json(res, os.path.join(C.EXP, "results.json"))
    figures(res)
    print(json.dumps(rules, indent=1, default=float))
    print("anomalies:", anomalies)
    print("assertions:", diag["assertions"])


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    col = {"F-KAN − C": "#2a78d6", "F-CB − C": "#eb6834"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "text.color": ink, "figure.facecolor": surf, "axes.facecolor": surf})
    tiers = C.TREND
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in ((axes[0], "macro_f1", "Δ macro-F1 (CV, seed-averaged OOF)"),
                              (axes[1], "ad_auc", "Δ AD-vs-rest AUC (CV, seed-averaged OOF)")):
        ax.axhline(0, color=ink2, lw=1)
        for j, (lab, key) in enumerate((("F-KAN − C", "F-KAN - C"), ("F-CB − C", "F-CB - C"))):
            x = np.arange(len(tiers)) + (j - 0.5) * 0.18
            d = [res["deltas"]["cv"][t][key][metric] for t in tiers]
            v = np.array([e["delta"] for e in d])
            lo, hi = v - np.array([e["lo"] for e in d]), np.array([e["hi"] for e in d]) - v
            ax.errorbar(x, v, yerr=[lo, hi], fmt="o", ms=7, color=col[lab], ecolor=col[lab], elinewidth=2, capsize=0,
                        label=lab, mec=surf, mew=2)
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels(tiers)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    fig.text(0.01, 0.01, "Points: paired Δ; bars: 95% subject bootstrap CI (2,000 resamples). n = 569.",
             color=ink2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(C.FIGS, "delta_by_tier.png"), dpi=150)
    plt.close(fig)
    for t in ("T1", "T2", "Full-spec"):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, s in zip(axes, ("C", "F-KAN")):
            cm = np.array(res["confusion_cv"][t][s])
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                            color="#ffffff" if cm[i, j] > 0.55 * cm.max() else ink, fontsize=10)
            ax.set_xticks(range(4))
            ax.set_xticklabels(C.NAMES)
            ax.set_yticks(range(4))
            ax.set_yticklabels(C.NAMES)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{t} — {s} (CV, seed-averaged)", fontsize=10, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(C.FIGS, f"confusion_{t}.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    os.makedirs(C.TABLES, exist_ok=True)
    os.makedirs(C.FIGS, exist_ok=True)
    main()
