"""B4: operating characteristics and reliability of model probabilities, from stored predictions only.

Run with LYP/.venv-fusion. No model is fitted and no inference is run.
- CV: analysis/fusion_v3/kan_runs/32.npz (config 32 = the frozen system's configuration, trained in
  the paper's train CV; cv: (5, 569, 4) stage probabilities, cv_add: (5, 569) ADD head).
- Test: adc4ccb:analysis/fusion_v3/frozen_run/test_kan.npz (frozen 25-KAN ensemble, stored at the single
  test run) and the labels y from adc4ccb:.../test_streams.npz. Read from git objects; nothing else
  from test is loaded. Labelled "post-hoc metric on already-read test predictions; no model change".
Threshold (prereg B4): the app defines none for AD vs rest (stage = argmax, report_builder.py:120;
ADD probability shown without a cut-off, report_builder.py:278 at ffbabe6), so primary = ADD >= 0.5;
secondary = the app's stage call (argmax == AD).
Reliability: Brier (ADD head; multiclass stage), ECE 10 equal-width bins (ADD head; top-label stage),
per seed then mean, bootstrap CIs; reliability diagram from the 5 seeds pooled.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

TEST_LABEL = "post-hoc metric on already-read test predictions; no model change"
THR = 0.5
BINS = np.linspace(0, 1, 11)


def sens_spec(pos_true, pos_pred):
    tp = np.sum(pos_true & pos_pred)
    tn = np.sum(~pos_true & ~pos_pred)
    return tp / pos_true.sum(), tn / (~pos_true).sum()


def bin_of(p):
    return np.minimum((p * 10).astype(int), 9)


def ece(conf, hit):
    b = bin_of(conf)
    e = 0.0
    for k in range(10):
        m = b == k
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return e


def rel_table(conf, hit):
    b = bin_of(conf)
    rows = []
    for k in range(10):
        m = b == k
        rows.append({"bin_lo": BINS[k], "bin_hi": BINS[k + 1], "n": int(m.sum()),
                     "mean_prob": float(conf[m].mean()) if m.any() else None,
                     "observed": float(hit[m].mean()) if m.any() else None})
    return rows


def stats_cv(y, P, A):
    """Per-seed statistics (dict of name -> (n_seeds,) array)."""
    ad = y == 3
    out = {k: [] for k in ("sens_add", "spec_add", "sens_stage", "spec_stage", "brier_add", "brier_stage",
                           "ece_add", "ece_stage")}
    onehot = np.eye(4)[y]
    for p, a in zip(P, A):
        s, sp = sens_spec(ad, a >= THR)
        out["sens_add"].append(s), out["spec_add"].append(sp)
        s, sp = sens_spec(ad, p.argmax(1) == 3)
        out["sens_stage"].append(s), out["spec_stage"].append(sp)
        out["brier_add"].append(np.mean((a - ad) ** 2))
        out["brier_stage"].append(np.mean(((p - onehot) ** 2).sum(1)))
        out["ece_add"].append(ece(a, ad.astype(float)))
        out["ece_stage"].append(ece(p.max(1), (p.argmax(1) == y).astype(float)))
    return {k: np.array(v) for k, v in out.items()}


def main():
    pv = C.Prov("B4_operating", __file__)
    tr, _ = C.load_trainval()
    st = np.load(pv.file(os.path.join(C.V3, "streams.npz")))
    pv.check("streams.npz train order == fusion_trainval train order", (st["train_rids"] == tr.RID.to_numpy()).all())
    z = np.load(pv.file(os.path.join(C.V3, "kan_runs", "32.npz")))
    y = tr.label.to_numpy()
    P, A = z["cv"], z["cv_add"]
    cvf1 = float(np.mean(C.f1_seeds(y, P)))
    pv.check("config 32 CV macro-F1 == 0.7562 (config_log row 32)", abs(cvf1 - 0.7562) < 5e-5, 0.7562, round(cvf1, 5))
    pv.file(os.path.join(C.ROOT, "analysis", "fusion_v3", "config_log.csv"))

    # app threshold source (read-only, recorded)
    rb = pv.git_blob("ffbabe6", "src/assessment/report_builder.py").decode().splitlines()
    pv.check("report_builder.py:120 is the argmax stage call", "np.argmax(p)" in rb[119], observed=rb[119].strip())
    pv.check("report_builder.py:278 is the ADD-head probability", "KAN_ADD_HEAD" in rb[277], observed=rb[277].strip())

    # CV
    point = {k: float(v.mean()) for k, v in stats_cv(y, P, A).items()}
    per_seed = {k: v.tolist() for k, v in stats_cv(y, P, A).items()}
    idx = C.boot_idx(len(y))
    boots = {k: [] for k in point}
    for ii in idx:
        s = stats_cv(y[ii], P[:, ii], A[:, ii])
        for k in point:
            boots[k].append(s[k].mean())
    cv = {k: {"value": point[k], "ci": C.ci(boots[k]), "per_seed": per_seed[k]} for k in point}

    # Test (already-read stored predictions)
    tk = pv.npz_blob("adc4ccb", "analysis/fusion_v3/frozen_run/test_kan.npz")
    ts = pv.npz_blob("adc4ccb", "analysis/fusion_v3/frozen_run/test_streams.npz")
    yt, pt, at = ts["y"], tk["p"], tk["add"]
    pv.check("test arrays aligned (123)", len(yt) == len(pt) == len(at) == 123, 123, len(yt))
    # stored test macro-F1 must match the published 0.764 (sanity that the right file was read)
    f1t = C.M.metrics(yt, pt)["macro_f1"]
    pv.check("stored test KAN macro-F1 == 0.764 (final_results §2.1)", abs(f1t - 0.764) < 5e-4, 0.764, round(f1t, 4))
    adt = yt == 3
    tests = {}
    for name, pred in (("add", at >= THR), ("stage", pt.argmax(1) == 3)):
        s, sp = sens_spec(adt, pred)
        bs = []
        for ii in C.boot_idx(len(yt)):
            bs.append(sens_spec(adt[ii], pred[ii]))
        bs = np.array(bs)
        tests[name] = {"sens": float(s), "sens_ci": C.ci(bs[:, 0]), "spec": float(sp), "spec_ci": C.ci(bs[:, 1]),
                       "n_AD": int(adt.sum()), "n_rest": int((~adt).sum()),
                       "TP": int((adt & pred).sum()), "FN": int((adt & ~pred).sum()),
                       "TN": int((~adt & ~pred).sum()), "FP": int((~adt & pred).sum()), "label": TEST_LABEL}

    # reliability diagram tables (pooled seeds)
    Ap, Pp, yp = A.reshape(-1), P.reshape(-1, 4), np.tile(y, len(C.SEEDS))
    rel_add = rel_table(Ap, (yp == 3).astype(float))
    rel_stage = rel_table(Pp.max(1), (Pp.argmax(1) == yp).astype(float))

    os.makedirs(os.path.join(C.OUT, "csv"), exist_ok=True)
    outs = []
    f = os.path.join(C.OUT, "csv", "B4_operating_characteristics.csv")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "rule", "metric", "value", "ci_lo", "ci_hi", "n", "note"])
        for rule, tag in (("ADD head >= 0.5 (primary)", "add"), ("argmax stage == AD (app stage call)", "stage")):
            for m in ("sens", "spec"):
                c = cv[f"{m}_{tag}"]
                w.writerow(["CV OOF (569, mean of 5 seeds)", rule, m, f"{c['value']:.4f}", f"{c['ci'][0]:.4f}",
                            f"{c['ci'][1]:.4f}", 569, ""])
            for m in ("sens", "spec"):
                t = tests[tag]
                w.writerow(["test (123, frozen ensemble)", rule, m, f"{t[m]:.4f}", f"{t[m + '_ci'][0]:.4f}",
                            f"{t[m + '_ci'][1]:.4f}", 123, TEST_LABEL])
    outs.append(f)
    f = os.path.join(C.OUT, "csv", "B4_reliability_summary.csv")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["output", "metric", "value", "ci_lo", "ci_hi", "per_seed", "note"])
        for k, out, met in (("brier_add", "ADD head", "Brier"), ("ece_add", "ADD head", "ECE (10 bins)"),
                            ("brier_stage", "stage (4-class)", "multiclass Brier"),
                            ("ece_stage", "stage (4-class)", "top-label ECE (10 bins)")):
            c = cv[k]
            w.writerow([out, met, f"{c['value']:.4f}", f"{c['ci'][0]:.4f}", f"{c['ci'][1]:.4f}",
                        " ".join(f"{v:.4f}" for v in c["per_seed"]),
                        "reliability of model probabilities; CV out-of-fold, 569 train subjects"])
    outs.append(f)
    f = os.path.join(C.OUT, "csv", "B4_reliability_bins.csv")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["output", "bin_lo", "bin_hi", "n_pooled_5_seeds", "mean_probability", "observed_frequency"])
        for name, tab in (("ADD head: P(AD) vs fraction AD", rel_add), ("stage: top-label confidence vs accuracy", rel_stage)):
            for r in tab:
                w.writerow([name, f"{r['bin_lo']:.1f}", f"{r['bin_hi']:.1f}", r["n"],
                            "" if r["mean_prob"] is None else f"{r['mean_prob']:.4f}",
                            "" if r["observed"] is None else f"{r['observed']:.4f}"])
    outs.append(f)
    res = {"threshold": THR, "threshold_source": "none defined in the app; 0.5 used as instructed", "cv": cv,
           "test": tests, "rel_add": rel_add, "rel_stage": rel_stage}
    outs.append(C.write_json(res, os.path.join(C.OUT, "json", "B4.json")))
    outs += figure(rel_add, rel_stage, cv)
    pv.write(outs)
    print({k: round(v["value"], 4) for k, v in cv.items()})
    print({k: (round(v["sens"], 3), round(v["spec"], 3)) for k, v in tests.items()})


def figure(rel_add, rel_stage, cv):
    sys.path.insert(0, C.HERE)
    import paper_style as PS
    plt = PS.plt
    fig, axes = plt.subplots(2, 2, figsize=(PS.DOUBLE, 3.6), gridspec_kw={"height_ratios": [3, 1.1]}, sharex="col")
    for j, (tab, lab, xl, yl, key) in enumerate((
            (rel_add, "a", "Mean ADD-head P(AD) in bin", "Fraction AD", "add"),
            (rel_stage, "b", "Mean top-label stage probability in bin", "Fraction correct", "stage"))):
        ax, axn = axes[0, j], axes[1, j]
        ax.plot([0, 1], [0, 1], color=PS.OTHER_GREY, lw=0.8, ls="--", zorder=1)
        xs = [r["mean_prob"] for r in tab if r["n"] > 0]
        ys = [r["observed"] for r in tab if r["n"] > 0]
        ax.plot(xs, ys, marker="o", ms=3.5, color=PS.OKABE_ITO["blue"], zorder=2)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.set_ylabel(yl)
        b, e = cv[f"brier_{key}"], cv[f"ece_{key}"]
        ax.text(0.03, 0.97, f"{'Brier' if key == 'add' else 'Multiclass Brier'} {b['value']:.3f} "
                            f"[{b['ci'][0]:.3f}, {b['ci'][1]:.3f}]\nECE {e['value']:.3f} [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]",
                transform=ax.transAxes, va="top", ha="left")
        PS.panel_label(ax, lab, x=-0.13, y=1.06)
        centers = [(r["bin_lo"] + r["bin_hi"]) / 2 for r in tab]
        axn.bar(centers, [r["n"] for r in tab], width=0.09, color=PS.OTHER_GREY)
        axn.set_yscale("log")
        axn.set_ylabel("n (5 seeds)")
        axn.set_xlabel(xl)
        axn.set_xlim(0, 1)
    fig.tight_layout()
    return list(PS.save(fig, os.path.join(C.OUT, "figures"), "B4_reliability_diagram"))


if __name__ == "__main__":
    main()
