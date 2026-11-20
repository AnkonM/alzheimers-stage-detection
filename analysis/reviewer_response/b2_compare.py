"""B2 step 3 (LYP/.venv-fusion): original vs fold-wise numbers, prereg B2 rules. Train CV (+ EXP2 val).

Table II CV column: clinical specialist, MRI specialist (567 with a scan), KAN 32, CatBoost F, CatBoost R,
KAN 34 (MRI masked); comparisons KAN − clinical, KAN − CatBoost F, KAN − KAN 34 (macro-F1, paired
bootstrap as evaluate.py). Check: the original comparison CIs reproduce final_results.md §1.
Rules: a reading changes if a CI moves between "excludes 0" and "includes 0"; |shift| > 0.010 in a
system's CV macro-F1 is "noteworthy".
EXP2: rules R1–R5 and the reproduction gate from 880f4a7 results.json vs the fold-wise results.json.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

EXP2_NEW = os.path.join(C.PRIV, "exp2_foldwise", "results.json")
SYS = ["Clinical specialist", "MRI specialist", "KAN K2 (final, config 32)", "CatBoost F", "CatBoost R",
       "KAN K2, MRI masked (config 34)"]
COMPS = [("KAN K2 (final, config 32)", "Clinical specialist"), ("KAN K2 (final, config 32)", "CatBoost F"),
         ("KAN K2 (final, config 32)", "KAN K2, MRI masked (config 34)")]
PUBLISHED = {COMPS[0]: (0.013, -0.006, 0.034), COMPS[1]: (-0.002, -0.021, 0.018), COMPS[2]: (0.007, -0.006, 0.019)}


def excl0(c):
    return c[0] > 0 or c[1] < 0


def table2(pv):
    tr, _ = C.load_trainval()
    y, has = tr.label.to_numpy(), tr.has_mri.to_numpy().astype(bool)
    st = np.load(pv.file(os.path.join(C.V3, "streams.npz")))
    cb = np.load(pv.file(os.path.join(C.KF, "catboost_preds.npz")))
    k32 = np.load(pv.file(os.path.join(C.V3, "kan_runs", "32.npz")))["cv"]
    k34 = np.load(pv.file(os.path.join(C.V3, "kan_runs", "34.npz")))["cv"]
    fw = np.load(pv.file(os.path.join(C.PRIV, "b2_streams_foldwise.npz")))
    fk = np.load(pv.file(os.path.join(C.PRIV, "b2_kan_foldwise.npz")))
    pv.check("fold-wise rows == train order", (fw["train_rids"] == tr.RID.to_numpy()).all())
    P = {"original": dict(zip(SYS, [st["clin_oof"], st["mri_oof"], k32, cb["oof_F"], cb["oof_R"], k34])),
         "fold-wise": dict(zip(SYS, [fw["clin_oof"], fw["mri_oof"], fk["cv32"], fw["cbF_oof"], fw["cbR_oof"],
                                     fk["cv34"]]))}
    idx = C.boot_idx(len(y))
    idx_s = C.boot_idx(int(has.sum()))
    res = {"systems": {}, "comparisons": {}}
    for v, Pv in P.items():
        for s, A in Pv.items():
            mk = has if s == "MRI specialist" else np.ones(len(y), bool)
            f1 = C.f1_seeds(y[mk], A[:, mk])
            auc = C.auc_seeds(y[mk], A[:, mk])
            r = res["systems"].setdefault(s, {"n": int(mk.sum())})
            r[v] = {"macro_f1": float(f1.mean()), "macro_f1_sd": float(f1.std(ddof=1)), "ad_auc": float(auc.mean()),
                    "macro_f1_ci": C.ci(C.f1_boot(y[mk], A[:, mk], idx_s if s == "MRI specialist" else idx))}
        for a, b in COMPS:
            d = float((C.f1_seeds(y, Pv[a]) - C.f1_seeds(y, Pv[b])).mean())
            c = C.ci(C.f1_boot(y, Pv[a], idx) - C.f1_boot(y, Pv[b], idx))
            res["comparisons"].setdefault(f"{a} − {b}", {})[v] = {"delta": d, "ci": c, "excludes_0": excl0(c)}
    for a, b in COMPS:
        o = res["comparisons"][f"{a} − {b}"]["original"]
        pub = PUBLISHED[(a, b)]
        ok = abs(o["delta"] - pub[0]) < 6e-4 and abs(o["ci"][0] - pub[1]) < 6e-4 and abs(o["ci"][1] - pub[2]) < 6e-4
        pv.check(f"original CI reproduces final_results §1: {a} − {b}", ok, pub,
                 [round(o["delta"], 4)] + [round(x, 4) for x in o["ci"]])
    for k, c in res["comparisons"].items():
        c["reading_changes"] = c["original"]["excludes_0"] != c["fold-wise"]["excludes_0"]
    for s, r in res["systems"].items():
        r["shift"] = r["fold-wise"]["macro_f1"] - r["original"]["macro_f1"]
        r["noteworthy"] = abs(r["shift"]) > 0.010
        r["shift_ad_auc"] = r["fold-wise"]["ad_auc"] - r["original"]["ad_auc"]
    return res


def exp2(pv):
    old = json.loads(pv.git_blob("880f4a7", "experiments/exp2_reduced_clinical/results.json"))
    new = json.load(open(pv.file(EXP2_NEW)))
    out = {"rules": {}, "gate": {}, "tiers": {}}
    for R in ("R1", "R2"):
        for t, o in old["rules"][R].items():
            n = new["rules"][R][t]
            out["rules"][f"{R} {t}"] = {"original": o, "fold-wise": n, "changed": o["met"] != n["met"]}
    for R in ("R3", "R4"):
        out["rules"][R] = {"original": old["rules"][R], "fold-wise": new["rules"][R],
                           "changed": old["rules"][R]["met"] != new["rules"][R]["met"]}
    out["rules"]["R5"] = {"original": old["rules"]["R5"], "fold-wise": new["rules"]["R5"],
                          "changed": json.dumps(old["rules"]["R5"], sort_keys=True) !=
                          json.dumps(new["rules"]["R5"], sort_keys=True)}
    out["gate"] = {"original": old["gate"]["pass"], "fold-wise": new["gate"]["pass"],
                   "rows_fold_wise": new["gate"]["rows"]}
    for t in old["metrics"]["cv"]:
        out["tiers"][t] = {}
        for ev in ("cv", "val"):
            for s in ("C", "M", "F-KAN", "F-CB", "F-KAN-noMRI"):
                for m in ("macro_f1", "ad_auc"):
                    out["tiers"][t][f"{ev} {s} {m}"] = [old["metrics"][ev][t][s][m], new["metrics"][ev][t][s][m]]
            for comp in ("F-KAN - C", "F-CB - C"):
                for m in ("macro_f1", "ad_auc"):
                    od, nd = old["deltas"][ev][t][comp][m], new["deltas"][ev][t][comp][m]
                    out["tiers"][t][f"{ev} Δ {comp} {m}"] = [[od["delta"], od.get("lo"), od.get("hi")],
                                                             [nd["delta"], nd.get("lo"), nd.get("hi")]]
    return out


def main():
    pv = C.Prov("B2_compare", __file__)
    t2 = table2(pv)
    outs = [C.write_json(t2, os.path.join(C.OUT, "json", "B2_table2.json"))]
    f = os.path.join(C.OUT, "csv", "B2_table2_cv_original_vs_foldwise.csv")
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "n", "original", "original_ci_lo", "original_ci_hi", "fold_wise", "fold_wise_ci_lo",
                    "fold_wise_ci_hi", "shift", "flag"])
        for s, r in t2["systems"].items():
            o, n = r["original"], r["fold-wise"]
            w.writerow([f"{s}: CV macro-F1", r["n"], f"{o['macro_f1']:.4f}", f"{o['macro_f1_ci'][0]:.4f}",
                        f"{o['macro_f1_ci'][1]:.4f}", f"{n['macro_f1']:.4f}", f"{n['macro_f1_ci'][0]:.4f}",
                        f"{n['macro_f1_ci'][1]:.4f}", f"{r['shift']:+.4f}", "noteworthy (>0.010)" if r["noteworthy"] else ""])
            w.writerow([f"{s}: CV AD-vs-rest AUC", r["n"], f"{o['ad_auc']:.4f}", "", "", f"{n['ad_auc']:.4f}", "", "",
                        f"{r['shift_ad_auc']:+.4f}", ""])
        for k, c in t2["comparisons"].items():
            o, n = c["original"], c["fold-wise"]
            w.writerow([f"{k}: Δ CV macro-F1", 569, f"{o['delta']:+.4f}", f"{o['ci'][0]:+.4f}", f"{o['ci'][1]:+.4f}",
                        f"{n['delta']:+.4f}", f"{n['ci'][0]:+.4f}", f"{n['ci'][1]:+.4f}",
                        f"{n['delta'] - o['delta']:+.4f}", "READING CHANGES" if c["reading_changes"] else "reading unchanged"])
    outs.append(f)
    if os.path.exists(EXP2_NEW):
        e2 = exp2(pv)
        outs.append(C.write_json(e2, os.path.join(C.OUT, "json", "B2_exp2.json")))
        f = os.path.join(C.OUT, "csv", "B2_exp2_rules_original_vs_foldwise.csv")
        with open(f, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rule", "original_met", "fold_wise_met", "changed", "original_cv_delta [CI]; val_delta",
                        "fold_wise_cv_delta [CI]; val_delta"])
            for k, r in e2["rules"].items():
                o, n = r["original"], r["fold-wise"]
                if k.startswith(("R1", "R2")):
                    od = f"{o['cv_delta']:+.4f} [{o['cv_lo']:+.4f}, {o['cv_hi']:+.4f}]; {o['val_delta']:+.4f}"
                    nd = f"{n['cv_delta']:+.4f} [{n['cv_lo']:+.4f}, {n['cv_hi']:+.4f}]; {n['val_delta']:+.4f}"
                    w.writerow([k, o["met"], n["met"], r["changed"], od, nd])
                elif k == "R3":
                    w.writerow([k, o["met"], n["met"], r["changed"], f"rho {o['spearman']:+.3f}", f"rho {n['spearman']:+.3f}"])
                elif k == "R4":
                    w.writerow([k, o["met"], n["met"], r["changed"], f"{o['delta']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}]",
                                f"{n['delta']:+.4f} [{n['lo']:+.4f}, {n['hi']:+.4f}]"])
                else:
                    w.writerow([k, "", "", r["changed"], json.dumps(o), json.dumps(n)])
            w.writerow(["reproduction gate", e2["gate"]["original"], e2["gate"]["fold-wise"],
                        e2["gate"]["original"] != e2["gate"]["fold-wise"], "", "reported, not a stop condition (prereg B2)"])
        outs.append(f)
        f = os.path.join(C.OUT, "csv", "B2_exp2_metrics_original_vs_foldwise.csv")
        with open(f, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tier", "quantity", "original", "original_ci", "fold_wise", "fold_wise_ci", "shift"])
            for t, q in e2["tiers"].items():
                for k, (o, n) in q.items():
                    if isinstance(o, list):
                        oc = "" if o[1] is None else f"[{o[1]:+.4f}, {o[2]:+.4f}]"
                        nc = "" if n[1] is None else f"[{n[1]:+.4f}, {n[2]:+.4f}]"
                        w.writerow([t, k, f"{o[0]:+.4f}", oc, f"{n[0]:+.4f}", nc, f"{n[0] - o[0]:+.4f}"])
                    else:
                        w.writerow([t, k, f"{o:.4f}", "", f"{n:.4f}", "", f"{n - o:+.4f}"])
        outs.append(f)
    else:
        pv.note("EXP2 fold-wise results.json not found; EXP2 part not compared")
    pv.write(outs)
    print(json.dumps({s: {"orig": round(r["original"]["macro_f1"], 4), "fw": round(r["fold-wise"]["macro_f1"], 4)}
                      for s, r in t2["systems"].items()}))
    print(json.dumps({k: {v: [round(c[v]["delta"], 4)] + [round(x, 4) for x in c[v]["ci"]] for v in ("original", "fold-wise")}
                      | {"changes": c["reading_changes"]} for k, c in t2["comparisons"].items()}, indent=0))


if __name__ == "__main__":
    main()
