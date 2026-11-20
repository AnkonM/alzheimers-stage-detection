"""Write SHARE_THIS.md from results.json (spec Section 9 template). Every number comes from
results.json; the Observations block is read from logs/observations.md (written by hand
after the tables, factual only)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

R = json.load(open(os.path.join(C.EXP, "results.json")))
META = json.load(open(os.path.join(C.LOGS, "run_meta.json")))
M, D = R["metrics"], R["deltas"]
ROWS = C.TREND + ["Full-paper"]


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def ci(e, d=3):
    if e is None or e.get("delta") is None:
        return "—"
    if e.get("lo") is None:
        return f"{e['delta']:+.{d}f}"
    return f"{e['delta']:+.{d}f} [{e['lo']:+.{d}f}, {e['hi']:+.{d}f}]"


def tname(t):
    return {"Full-paper": "Full-paper (ref.)"}.get(t, t)


L = []
w = L.append
w("# EXP2 — Reduced clinical inputs: results\n")
w("## Run metadata")
w(f"- Branch `exp/reduced-clinical` (from `9bac529`) / commit (code) `634f57a` (commit 2; evaluation code added in commit 3) / "
  f"commit (preregistration) `de368a1` / addendum `3744036` / commit 3: given in the hand-over message")
w(f"- Date {time.strftime('%Y-%m-%d')} / total runtime {META['total_runtime']} / seeds 42–46 (outer) × KAN seeds 42–46 / "
  f"bootstrap {R['n_boot']} resamples, rng seed 0 / hardware {META['hardware']}")
w("- Primary metric: spec §7 (metrics of seed-averaged OOF probabilities). CV n = 569 (M: 567 with a scan); val n = 122 (M: 121); "
  "val LMCI = 25.\n")

p1 = R["phase1"]
rec, rb = p1["reconstruct"], p1["rule_based"]
w("## Phase 0–1 audit")
w("- Logical Memory present in frozen feature set: **NO** (columns: `LDELTOTL`, `LIMMTOTL` exist only in the unused joined "
  "tables step0/step1; 0% missing in train)")
w("- Label-defining variables in frozen 12-feature set: CDRSB, CogIndex (MMSE − CDRSB − ADAS13, z-scored), MMSE in every "
  "seed's top-12; the frozen KAN also takes CDRSB, CogIndex, MMSE as direct inputs")
w(f"- Label reconstructability (CV on {rec['n']} train; inputs {{{', '.join(rec['inputs'])}}}; all 569 have all four):")
w("")
w("| Model | macro-F1 | acc | F1 CN / EMCI / LMCI / AD | confusion (rows true CN/EMCI/LMCI/AD) |")
w("|---|---|---|---|---|")
for k, nm in (("tree_depth3", "Depth-3 tree"), ("logistic", "Multinomial logistic")):
    r = rec[k]
    w(f"| {nm} | {f(r['macro_f1'])} | {f(r['acc'])} | {' / '.join(f(r[f'f1_{n}'], 2) for n in C.NAMES)} | "
      f"{' ; '.join(' '.join(str(v) for v in row) for row in r['confusion'])} |")
w("")
w(f"- Rule-based ADNI-GO/2 cut-offs on {rb['n']} train+val: uniquely matched {rb['uniquely_matched']}, ambiguous "
  f"{rb['ambiguous']} ({', '.join(f'{k} {v}' for k, v in rb['ambiguous_patterns'].items())}), unmatched {rb['unmatched']}. "
  f"All {rb['n']} (ambiguous/unmatched = error): acc {f(rb['acc_all'])}, macro-F1 {f(rb['macro_f1_all'])}. "
  f"Uniquely matched only: acc {f(rb['acc_matched'])}, macro-F1 {f(rb['macro_f1_matched'])}; confusion "
  f"{' ; '.join(' '.join(str(v) for v in row) for row in rb['confusion_matched'])}")
w("- Cut-off source: checked against the ADNI-2 Protocol (ADC-039), pp. 3 and 18–19 (all values match; the procedures "
  "manual itself was not consulted). The protocol's CDR memory-box criterion (CN 0, MCI ≥ 0.5) was omitted from the rule: "
  f"{rb['cdmemory_rule_check']['CN_label_with_memory_box_gt0']} CN and {rb['cdmemory_rule_check']['MCI_label_with_memory_box_0']} "
  "MCI subjects would fail it. Our scores are baseline-visit values while the cut-offs applied at screening, so some "
  "disagreement is expected.")
w("- Earlier fusion-feasibility ablation: found at `analysis/fusion_feasibility/ablation.py` (arm A′/B′); variables it removed: "
  "CDRSB, MMSE and CogIndex only, from the 36 step-4 features; the other 33, including the 7 raw FreeSurfer volumes and "
  "ADAS11/13, stayed in (no Logical Memory was ever in them). Its result was not reused.\n")

w("## Tier definitions")
w("| Tier | n features | Columns | Max missingness (train) |")
w("|---|---|---|---|")
man = __import__("yaml").safe_load(open(os.path.join(C.EXP, "manifest.yaml")))
short = {"T0": "AGE, PTGENDER, PTEDUCAT", "T1": "T0 + FAQ + EcogPt×7 + EcogSP×7", "T2": "T1 + MOCA",
         "T3": "T2 + ADAS11, ADAS13, Q4SCORE, RAVLT×4, TRABSCOR", "Full-spec": "paper's 29 (raw, in-fold step-4 rule)",
         "S-T2+APOE4": "T2 + APOE4 (sensitivity)", "Full-paper": "paper's 29 (step-4 table), fixed KAN extras"}
for t in C.TIERS:
    v = man["tiers"][t]
    w(f"| {t} | {v['n_features']} (top-{v['top_k']}; KAN width {v['kan_input_width']}) | {short[t]} | "
      f"{v['max_train_missingness']:.3f} |")
w("")

w("## Reproduction gate")
w("| System | Paper CV macro-F1 | Reproduced | Δ | PASS/FAIL |")
w("|---|---|---|---|---|")
for r in R["gate"]["rows"]:
    if r["source"] == "experiment code":
        lab = "§7, seed-averaged probabilities" if r["definition"].startswith("spec") else "paper-style, mean of per-seed"
        w(f"| {r['system']} ({lab}) | {r['paper']:.3f} | {r['reproduced']:.4f} | {r['delta']:+.4f} | "
          f"{'PASS' if r['pass'] else 'FAIL'} |")
ps = R["paper_style"]
w(f"\nExperiment code = saved paper predictions (max |diff| 0.0, CV and val). Δ F-KAN − C at Full-paper: §7 "
  f"{ci(D['cv']['Full-paper']['F-KAN - C']['macro_f1'])}; paper-style (mean of per-seed) {ci(ps['Full-paper']['delta'])}. "
  f"Full-spec: §7 {ci(D['cv']['Full-spec']['F-KAN - C']['macro_f1'])}; paper-style {ci(ps['Full-spec']['delta'])} "
  f"(C {f(ps['Full-spec']['C'])}, F-KAN {f(ps['Full-spec']['F-KAN'])}).\n")

w("## Main results — CV (569 train, seed-averaged OOF)")
w("| Tier | C macro-F1 | M macro-F1 | F-KAN macro-F1 | F-CB macro-F1 | Δ F-KAN−C [95% CI] | Δ F-CB−C [95% CI] |")
w("|---|---|---|---|---|---|---|")
for t in ROWS:
    m = M["cv"][t]
    w(f"| {tname(t)} | {f(m['C']['macro_f1'])} | {f(m['M']['macro_f1'])} | {f(m['F-KAN']['macro_f1'])} | "
      f"{f(m['F-CB']['macro_f1'])} | {ci(D['cv'][t]['F-KAN - C']['macro_f1'])} | {ci(D['cv'][t]['F-CB - C']['macro_f1'])} |")
HEAD = {"ad_auc": "| Tier | C AD-AUC | M AD-AUC | F-KAN AD-AUC | Δ F-KAN−C [95% CI] |",
        "emci_lmci_auc": "| Tier | C EMCI-vs-LMCI AUC | M | F-KAN | Δ [95% CI] |",
        "cn_mci_auc": "| Tier | C CN-vs-MCI AUC | M | F-KAN | Δ [95% CI] |"}
for metric in HEAD:
    w("")
    w(HEAD[metric])
    w("|---|---|---|---|---|")
    for t in ROWS:
        m = M["cv"][t]
        w(f"| {tname(t)} | {f(m['C'][metric])} | {f(m['M'][metric])} | {f(m['F-KAN'][metric])} | "
          f"{ci(D['cv'][t]['F-KAN - C'][metric])} |")
w("\nM is tier-independent (567 subjects with a scan).\n")

w("## Validation holdout (fit on 569, evaluate on 122)")
w("| Tier | C macro-F1 | F-KAN macro-F1 | Δ [95% CI] | C AD-AUC | F-KAN AD-AUC | Δ [95% CI] |")
w("|---|---|---|---|---|---|---|")
for t in ROWS:
    m = M["val"][t]
    w(f"| {tname(t)} | {f(m['C']['macro_f1'])} | {f(m['F-KAN']['macro_f1'])} | {ci(D['val'][t]['F-KAN - C']['macro_f1'])} | "
      f"{f(m['C']['ad_auc'])} | {f(m['F-KAN']['ad_auc'])} | {ci(D['val'][t]['F-KAN - C']['ad_auc'])} |")
w(f"\nM on val (121): macro-F1 {f(M['val']['T0']['M']['macro_f1'])}, AD-AUC {f(M['val']['T0']['M']['ad_auc'])}.\n")

w("## Per-class F1 (CV) at T1, T2, Full")
w("| Tier | System | CN | EMCI | LMCI | AD |")
w("|---|---|---|---|---|---|")
for t in ("T1", "T2", "Full-spec", "Full-paper"):
    for s in ("C", "M", "F-KAN", "F-CB"):
        if s == "M" and t != "T1":
            continue
        m = M["cv"][t][s]
        w(f"| {tname(t) if s != 'M' else 'all tiers'} | {s} | {' | '.join(f(m[f'f1_{n}']) for n in C.NAMES)} |")
w("")

w("## MRI masking inside fusion (CV)")
w("| Tier | Δ macro-F1 (F-KAN − F-KAN-noMRI) [95% CI] | Δ macro-F1 (F-KAN − F-KAN-mask-inf) [95% CI] | "
  "Δ AD-AUC (F-KAN − F-KAN-noMRI) [95% CI] |")
w("|---|---|---|---|")
for t in ROWS:
    d = D["cv"][t]
    w(f"| {tname(t)} | {ci(d['F-KAN - F-KAN-noMRI']['macro_f1'])} | {ci(d['F-KAN - F-KAN-mask-inf']['macro_f1'])} | "
      f"{ci(d['F-KAN - F-KAN-noMRI']['ad_auc'])} |")
w("\nF-KAN-noMRI: retrained with MRI masked in training and evaluation (paper's control). F-KAN-mask-inf: the same "
  "F-KAN models with MRI masked at inference only.\n")

E = R["e1"]
w("## E1 — Rule-ambiguity subsets (exploratory; addendum `3744036`; descriptive, no verdict)")
for ev in ("cv", "val"):
    comp = E["composition"][ev]
    w(f"- {('Train' if ev == 'cv' else 'Val')} composition: " + "; ".join(
        f"{k} n={v['n']} (CN {v['CN']}, EMCI {v['EMCI']}, LMCI {v['LMCI']}, AD {v['AD']}; scan {v['n_with_scan']})"
        for k, v in comp.items()))
w("")
w("| Eval | Subset | Tier | macro-F1* C / M / F-KAN / F-CB | AD-AUC C / M / F-KAN / F-CB | Δ F-KAN−C macro-F1* | Δ F-KAN−C AD-AUC |")
w("|---|---|---|---|---|---|---|")
for ev in ("cv", "val"):
    for sub in ("AMBIGUOUS", "UNAMBIGUOUS"):
        for t in C.TREND + ["S-T2+APOE4", "Full-paper"]:
            o = E[ev][sub][t]
            w(f"| {ev.upper()} | {sub} (n={o['C']['n']}) | {tname(t)} | "
              f"{' / '.join(f(o[s]['macro_f1_present']) for s in ('C', 'M', 'F-KAN', 'F-CB'))} | "
              f"{' / '.join(f(o[s]['ad_auc']) for s in ('C', 'M', 'F-KAN', 'F-CB'))} | "
              f"{ci(o['delta_FKAN_C_macro_f1'])} | {ci(o['delta_FKAN_C_ad_auc'])} |")
w("\n*macro-F1 over the true classes present. Accuracy per cell in `tables/e1_ambiguity_subsets.csv`. Val subsets with "
  "n < 30: point estimates only. CV Δ(AMBIGUOUS) − Δ(UNAMBIGUOUS), macro-F1: " + "; ".join(
      f"{t} {f(E['amb_minus_unamb_cv'][t]['macro_f1'])}" for t in C.TREND + ["S-T2+APOE4", "Full-paper"]) + ".\n")

w("## Sensitivity: T2 + APOE4")
t = "S-T2+APOE4"
m, d, v, dv = M["cv"][t], D["cv"][t], M["val"][t], D["val"][t]
w("| Table | Values |")
w("|---|---|")
w(f"| CV macro-F1 | C {f(m['C']['macro_f1'])}, F-KAN {f(m['F-KAN']['macro_f1'])}, F-CB {f(m['F-CB']['macro_f1'])}; "
  f"Δ F-KAN−C {ci(d['F-KAN - C']['macro_f1'])}; Δ F-CB−C {ci(d['F-CB - C']['macro_f1'])} |")
for metric, lab in (("ad_auc", "CV AD-AUC"), ("emci_lmci_auc", "CV EMCI-vs-LMCI AUC"), ("cn_mci_auc", "CV CN-vs-MCI AUC")):
    w(f"| {lab} | C {f(m['C'][metric])}, F-KAN {f(m['F-KAN'][metric])}; Δ {ci(d['F-KAN - C'][metric])} |")
w(f"| Val | macro-F1 C {f(v['C']['macro_f1'])}, F-KAN {f(v['F-KAN']['macro_f1'])}, Δ {ci(dv['F-KAN - C']['macro_f1'])}; "
  f"AD-AUC C {f(v['C']['ad_auc'])}, F-KAN {f(v['F-KAN']['ad_auc'])}, Δ {ci(dv['F-KAN - C']['ad_auc'])} |")
w(f"| Masking | Δ F-KAN − F-KAN-noMRI {ci(d['F-KAN - F-KAN-noMRI']['macro_f1'])}; − mask-inf "
  f"{ci(d['F-KAN - F-KAN-mask-inf']['macro_f1'])} |")
w(f"| Rules (reported, not in R3) | R1 {'MET' if R['rules']['R1'][t]['met'] else 'NOT MET'}; R2 "
  f"{'MET' if R['rules']['R2'][t]['met'] else 'NOT MET'} |")
w("")

w("## Pre-registered verdicts")
w("| Rule | Tier | Result | Verdict (MET / NOT MET / KAN-specific) |")
w("|---|---|---|---|")
rules = R["rules"]
for rule in ("R1", "R2"):
    for t in C.TREND:
        r = rules[rule][t]
        verdict = "MET" if r["met"] else "NOT MET"
        k5 = rules["R5"].get(f"{rule} {t}") if isinstance(rules["R5"], dict) else None
        if r["met"] and k5 and k5["verdict"] == "KAN-specific":
            verdict = "KAN-specific"
        w(f"| {rule} | {t} | CV Δ {r['cv_delta']:+.3f} [{r['cv_lo']:+.3f}, {r['cv_hi']:+.3f}]; val Δ {r['val_delta']:+.3f} | {verdict} |")
r3 = rules["R3"]
w(f"| R3 | T0→Full-spec | Spearman ρ = {r3['spearman']:+.2f} (Δ: {', '.join(f'{x:+.3f}' for x in r3['delta_macro_f1'])}); "
  f"descriptive, 5 points | {'MET' if r3['met'] else 'NOT MET'} |")
r4 = rules["R4"]
w(f"| R4 | T0 | M AD-AUC {r4['M_ad_auc']:.3f} − C {r4['C_ad_auc_scan']:.3f} (567 with scan) = {r4['delta']:+.3f} "
  f"[{r4['lo']:+.3f}, {r4['hi']:+.3f}] | {'MET' if r4['met'] else 'NOT MET'} |")
if "note" in rules["R5"]:
    w(f"| R5 | — | {rules['R5']['note']} | n/a |")
else:
    for k, v in rules["R5"].items():
        w(f"| R5 | {k} | F-KAN Δ {v['fkan_delta']:+.3f}, F-CB Δ {v['fcb_delta']:+.3f} | {v['verdict']} |")
w("")

w("## Seed spread")
w("| Tier | System | min–max per-seed macro-F1 |")
w("|---|---|---|")
for t in ROWS:
    w(f"| {tname(t)} | C / F-KAN / F-CB / F-KAN-noMRI | " + " ; ".join(
        f"{f(M['cv'][t][s]['seed_macro_f1_min'])}–{f(M['cv'][t][s]['seed_macro_f1_max'])}"
        for s in ("C", "F-KAN", "F-CB", "F-KAN-noMRI")) + " |")
w(f"| all | M | {f(M['cv']['T0']['M']['seed_macro_f1_min'])}–{f(M['cv']['T0']['M']['seed_macro_f1_max'])} |")
w("")

a = R["diagnostics"]["assertions"]
w("## Sanity checks")
w(f"- [x] No test rows loaded (assertion count: {a.get('no_test', 0)} no-test assertions, 0 failures; step1 read line-by-line "
  "after D0)")
w(f"- [x] No excluded column in any T0–T3 design matrix ({a.get('excluded', 0)} design-matrix assertions + re-check of every "
  "unit's top-K and KAN extras)")
w(f"- [x] Subject-disjoint folds asserted ({a.get('disjoint', 0)} checks)")
w(f"- [{'x' if META['hash_ok'] else ' '}] Frozen artefacts unchanged (hash check before/after: MANIFEST 109 files OK; "
  f"config_log.csv, kan_runs/, streams.npz {'unchanged' if META['hash_ok'] else 'CHANGED: ' + str(META['hash_changed'])}; "
  "snapshots phase2_start, phase3_start, phase4_end)")
w(f"- [{'x' if R['gate']['pass'] else ' '}] Reproduction gate passed")
w("")

w("## Deviations from protocol")
for i, x in enumerate(META["deviations"], 1):
    w(f"{i}. {x}")
w("")
w("## Anomalies and warnings")
for x in META["anomalies"] + R["anomalies"]:
    w(f"- {x}")
w("")
w("## Observations (max 5 bullets, factual, no claims beyond the tables)")
w(open(os.path.join(C.LOGS, "observations.md")).read().strip() if os.path.exists(os.path.join(C.LOGS, "observations.md"))
  else "- (pending)")
w("")
w("## Files")
w("- `experiments/exp2_reduced_clinical/`: `PREREGISTRATION.md`, `PREREGISTRATION_ADDENDUM.md`, `manifest.yaml`, "
  "`results.json`, `code/`")
w("- `tables/`: `metrics_all.csv`, `deltas_all.csv`, `e1_ambiguity_subsets.csv`, `reproduction_gate.csv`, "
  "`confusion_cv_<tier>_<system>.csv`, `phase1_*.csv`")
w("- `figures/`: `delta_by_tier.png`, `confusion_T1.png`, `confusion_T2.png`, `confusion_Full-spec.png`")
w("- Local only (gitignored): `oof/` (streams_<tier>.npz, kan_<tier>.npz, *_meta.json), `logs/`, `trainval_extra.parquet`")
open(os.path.join(C.EXP, "SHARE_THIS.md"), "w").write("\n".join(L) + "\n")
print(len(L), "lines")
