"""Stage D: the analysis of docs/external_validation_protocol.md section 7, as a script that is tested on synthetic data
only and run once, after the full run is complete.

Blindness: `assemble` is the only entry that reads labels.csv, and it first requires a RUN_COMPLETE marker of a `full`
run whose pinned manifest hash equals the manifest on disk and whose state file shows every case finished. Otherwise it
raises before the labels file is opened. Scores come from `report_reader.score_view`; the specialist's argmax-stage
field that MRI_ONLY reports carry in their drivers block is never read.

Everything returned is aggregate-only and is written by `aggregate_io.write_json`, which refuses case ids, OASIS ids and
per-subject vectors and allows only docs/validation/ and docs/figures/validation/ inside the repository.

Constants (thresholds, seed, resamples, age edges, subset size) live in `stats.py`, once; a test compares them with the
protocol text. Probabilities are model probabilities; nothing here claims or tests that they are reliable as
probabilities.
"""
import csv
import json
import pathlib

import numpy as np
import pandas as pd

from . import stats as S
from ..assessment import batch, labelmaps
from ..assessment.report_reader import score_view

KEEP_QC = ("PASS", "WARN")
KNOWN_QC = ("PASS", "WARN", "FAIL", "failed")
REPO = pathlib.Path(__file__).resolve().parents[2]
ADNI_REFERENCE = REPO / "analysis" / "external_validation" / "adni_reference.json"
STAGE_A_JSON = REPO / "docs" / "validation" / "stage_a_counts.json"


class BlindnessError(RuntimeError):
    """The run is not complete, or its pinned manifest hash does not match: labels were NOT opened."""


# ----------------------------------------------------------------------------------------- blindness and loading
def check_run_complete(run_root, manifest_dir):
    """Return the state header of a finished `full` run, or raise BlindnessError. Reads no labels."""
    root, man = pathlib.Path(run_root), pathlib.Path(manifest_dir)
    marker = root / "RUN_COMPLETE"
    if not marker.exists():
        raise BlindnessError("no RUN_COMPLETE marker in the run root (unfinished, halted or plumbing run): labels not opened")
    try:
        mk = json.loads(marker.read_text())
    except ValueError:
        raise BlindnessError("RUN_COMPLETE marker is unreadable: labels not opened") from None
    recs = batch.State(root / "state.jsonl").load()
    header = next((r for r in recs if r.get("type") == "header"), None)
    footers = [r for r in recs if r.get("type") == "footer"]
    if header is None or not footers:
        raise BlindnessError("state file has no header or no footer: labels not opened")
    if header.get("stage") != "full" or header.get("mode") != "MRI_ONLY":
        raise BlindnessError("the run is not a full MRI_ONLY run: labels not opened")
    if header.get("inputs_sha256") != batch.sha256_file(man / "inputs.csv"):
        raise BlindnessError("pinned manifest hash differs from the manifest on disk: labels not opened")
    f = footers[-1]
    if (f["n_ok"], f["n_failed"], f["n_cases"]) != (mk.get("n_ok"), mk.get("n_failed"), mk.get("n_cases")):
        raise BlindnessError("marker and state footer disagree: labels not opened")
    st = batch.case_status(recs)
    ids = [r["case_id"] for r in batch.read_inputs(man)]
    if f["n_cases"] != len(ids) or any(st.get(c, {}).get("status") not in ("ok", "fail") for c in ids):
        raise BlindnessError("not every case has a finished status: labels not opened")
    return header


def load_demographics(manifest_dir):
    rows = batch.read_inputs(manifest_dir)                       # inputs.csv only
    return pd.DataFrame({"case_id": [r["case_id"] for r in rows], "age": [r["age"] for r in rows],
                         "sex": [r["sex"][0] for r in rows]})   # "F" / "M"


def load_scores(run_root, case_ids):
    """One row per case. Reports are read through score_view only. A case that failed with a quality-control error is
    qc FAIL; any other failure, an unreadable report or a missing verdict is qc failed. Scores of those are NaN."""
    root = pathlib.Path(run_root)
    st = batch.case_status(batch.State(root / "state.jsonl").load())
    rows = []
    for cid in case_ids:
        rec = {"case_id": cid, "alz": np.nan, "p_ad": np.nan, "bag": np.nan, "corr_vs_mni": np.nan, "qc": "failed",
               "volume_provider": None}
        s = st.get(cid, {})
        if s.get("status") == "ok":
            try:
                v = score_view(root / "cases" / cid / "report.json")
                rec.update(alz=_num(v["alz"]), p_ad=_num(v["p_ad"]), bag=_num(v["bag"]), corr_vs_mni=_num(v["corr_vs_mni"]),
                           qc=v["qc"] if v["qc"] in KNOWN_QC else "failed", volume_provider=v["volume_provider"])
            except (OSError, ValueError, KeyError):
                pass
        elif s.get("status") == "fail" and s.get("error_class") == "QCFail":
            rec["qc"] = "FAIL"
        rows.append(rec)
    return pd.DataFrame(rows)


def _num(x):
    return np.nan if x is None else float(x)


def load_labels(manifest_dir, expected_sha256=None):
    """labels.csv -> DataFrame (cdr, mmse, etiv, nwbv). Only called through `assemble`, after the completion check."""
    path = pathlib.Path(manifest_dir) / "labels.csv"
    if expected_sha256 and batch.sha256_file(path) != expected_sha256:
        raise BlindnessError("labels.csv hash differs from the one recorded at Stage A")
    with open(path, newline="") as f:
        rd = list(csv.DictReader(f))
    return pd.DataFrame({"case_id": [r["case_id"] for r in rd], "cdr": [labelmaps.parse_cdr(r["CDR"]) for r in rd],
                         "mmse": [float(r["MMSE"]) if r["MMSE"].strip() else np.nan for r in rd],
                         "etiv": [float(r["eTIV"]) if r["eTIV"].strip() else np.nan for r in rd],
                         "nwbv": [float(r["nWBV"]) if r["nWBV"].strip() else np.nan for r in rd]})


def stage_a_labels_sha256():
    try:
        return json.loads(STAGE_A_JSON.read_text())["manifest_sha256"]["labels.csv"]
    except (OSError, KeyError, ValueError):
        return None


def assemble(run_root, manifest_dir, expected_labels_sha256="stage_a"):
    """The only entry that reads labels. Completion check first, then scores, labels and demographics joined on case_id."""
    check_run_complete(run_root, manifest_dir)
    expected = stage_a_labels_sha256() if expected_labels_sha256 == "stage_a" else expected_labels_sha256
    labels = load_labels(manifest_dir, expected)
    demo = load_demographics(manifest_dir)
    scores = load_scores(run_root, list(demo["case_id"]))
    df = align_cases(scores, labels)
    return df.merge(demo, on="case_id", how="inner", validate="one_to_one").sort_values("case_id").reset_index(drop=True)


def load_scores_only(run_root, manifest_dir):
    """Scores of a second run (S6f), after the same completion check; labels are not read."""
    check_run_complete(run_root, manifest_dir)
    return load_scores(run_root, [r["case_id"] for r in batch.read_inputs(manifest_dir)])


# ----------------------------------------------------------------------------------------- table preparation
def align_cases(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join on case_id, sorted by case_id. Duplicate ids, missing ids and unmatched ids raise, so a shuffled or
    mis-sorted file can never silently pair the wrong score with the wrong label."""
    for name, df in (("scores", scores), ("labels", labels)):
        if "case_id" not in df.columns:
            raise ValueError(f"{name} has no case_id column")
        if df["case_id"].duplicated().any():
            raise ValueError(f"{name} has duplicate case ids")
    a, b = set(scores["case_id"]), set(labels["case_id"])
    if a != b:
        raise ValueError(f"case id sets differ: {len(a - b)} only in scores, {len(b - a)} only in labels")
    out = scores.merge(labels, on="case_id", how="inner", validate="one_to_one")
    return out.sort_values("case_id").reset_index(drop=True)


def normalise_qc(df: pd.DataFrame) -> pd.DataFrame:
    """A missing or unknown QC verdict counts as failed (excluded from the primary analysis and counted)."""
    df = df.copy()
    df["qc"] = df["qc"].where(df["qc"].isin(KNOWN_QC), "failed")
    return df


def add_classes(df: pd.DataFrame, label_map: str, raw_col: str = "cdr") -> pd.DataFrame:
    lm = labelmaps.get(label_map)
    df = df.copy()
    df["cls"] = [lm(v) for v in df[raw_col]]
    df["rank"] = [lm.rank(v) for v in df[raw_col]]
    return df


def qc_policy(df: pd.DataFrame, score_col: str):
    """Primary-analysis rows (QC PASS or WARN, a score present) plus exclusion counts by class."""
    ok = df["qc"].isin(KEEP_QC) & df[score_col].notna()
    excl = df[~ok].groupby(["cls", "qc"], dropna=False).size()
    counts = {f"{c}|{q}": int(n) for (c, q), n in excl.items()}
    return df[ok], {"n_total": int(len(df)), "n_kept": int(ok.sum()), "excluded_by_class_and_qc": counts}


def _contrast(df, score_col, lm):
    pos = df.loc[df["cls"] == lm.positive, score_col].to_numpy(float)
    neg = df.loc[df["cls"] == lm.reference, score_col].to_numpy(float)
    return pos, neg


def _auc_ci(pos, neg, min_each=2):
    return S.bootstrap_auc(pos, neg) if len(pos) >= min_each and len(neg) >= min_each else \
        {"auc": None, "ci95": None, "n_pos": int(len(pos)), "n_neg": int(len(neg)), "note": "too few cases"}


# ----------------------------------------------------------------------------------------- H1 and S1-S3
def h1(df, score_col="alz", label_map="cdr_3class"):
    """Primary: reference class vs positive class, with direction check and verdict. Intermediate classes unused."""
    lm = labelmaps.get(label_map)
    kept, qc = qc_policy(df, score_col)
    pos, neg = _contrast(kept, score_col, lm)
    res = S.bootstrap_auc(pos, neg)
    res.update({"verdict": S.h1_verdict(res["auc"], res["ci95"][0]), "inverted": res["auc"] < 0.5, "qc": qc,
                "contrast": f"{lm.reference} vs {lm.positive}", "score": score_col,
                "direction": "higher score = more impaired is the hypothesis"})
    return res


def h1_worst_case(df, score_col="alz", label_map="cdr_3class"):
    """Excluded or failed cases get the median of the kept scores (uninformative, tied with each other)."""
    lm = labelmaps.get(label_map)
    kept, _ = qc_policy(df, score_col)
    filled = df.copy()
    bad = ~(df["qc"].isin(KEEP_QC) & df[score_col].notna())
    filled.loc[bad, score_col] = float(np.median(kept[score_col]))
    pos, neg = _contrast(filled, score_col, lm)
    return {**S.bootstrap_auc(pos, neg), "n_filled": int(bad.sum()), "rule": "failed or excluded cases tied at the median"}


def s1(df, score_col="alz", label_map="cdr_3class"):
    """Reference class vs the intermediate class (CDR 0.5 for cdr_3class): AUC with CI, direction only."""
    lm = labelmaps.get(label_map)
    mid = [c for c in lm.classes if c not in (lm.reference, lm.positive)]
    kept, _ = qc_policy(df, score_col)
    out = {}
    for m in mid:
        r = S.bootstrap_auc(kept.loc[kept["cls"] == m, score_col], kept.loc[kept["cls"] == lm.reference, score_col])
        r["direction_higher_in_intermediate"] = r["auc"] > 0.5
        out[m] = r
    return out


def s2_trend(df, score_col, label_map="cdr_3class"):
    """One-sided Jonckheere-Terpstra (increasing with class rank) and Spearman against the ordinal rank. Rank-based
    statistics only. Direction matches when the group medians do not decrease with severity and rho > 0; the outcome
    is PASS when it matches and the one-sided p is below the protocol alpha."""
    lm = labelmaps.get(label_map)
    kept, _ = qc_policy(df, score_col)
    groups = [kept.loc[kept["rank"] == i, score_col].to_numpy(float) for i in range(len(lm.classes))]
    jt = S.jonckheere_terpstra(groups, "increasing")
    rho = S.bootstrap_spearman(kept["rank"].to_numpy(float), kept[score_col].to_numpy(float))
    med = [float(np.median(g)) for g in groups]
    monotone = all(med[i] <= med[i + 1] for i in range(len(med) - 1))
    direction_ok = bool(monotone and rho["rho"] > 0)
    return {"jt": jt, "spearman_vs_rank": rho, "group_n": [int(len(g)) for g in groups], "group_median": med,
            "monotone_medians": monotone, "direction_matches": direction_ok,
            "outcome": "PASS" if direction_ok and jt["p"] < S.S2_ALPHA else "NOT MET", "alpha": S.S2_ALPHA, "score": score_col}


def s3_mmse(df, score_col, mmse_col="mmse"):
    """Spearman correlation with MMSE and bootstrap CI. For ALZ the outcome is PASS when rho <= the protocol limit."""
    kept, _ = qc_policy(df, score_col)
    kept = kept[kept[mmse_col].notna()]
    r = S.bootstrap_spearman(kept[score_col].to_numpy(float), kept[mmse_col].to_numpy(float))
    r.update({"score": score_col, "n_missing_mmse_dropped": int(len(qc_policy(df, score_col)[0]) - len(kept))})
    if score_col == "alz":
        r.update({"limit": S.S3_RHO_MAX, "outcome": "PASS" if r["rho"] <= S.S3_RHO_MAX else "NOT MET"})
    return r


# ----------------------------------------------------------------------------------------- S4
def age_tertile(age):
    """0 / 1 / 2 for age <= e1, e1 < age <= e2, age > e2 (protocol; right-closed)."""
    return np.digitize(np.asarray(age, float), S.S4_AGE_EDGES, right=True)


def s4(df, score_col="alz", label_map="cdr_3class"):
    lm = labelmaps.get(label_map)
    kept, _ = qc_policy(df, score_col)
    out = {"age_edges": list(S.S4_AGE_EDGES), "groups": "age <= e1; e1 < age <= e2; age > e2"}
    out["partial_spearman_vs_rank_age_adjusted"] = S.bootstrap_partial_spearman(
        kept["rank"], kept[score_col], kept["age"])
    k2 = kept[kept["mmse"].notna()]
    out["partial_spearman_vs_mmse_age_adjusted"] = S.bootstrap_partial_spearman(k2[score_col], k2["mmse"], k2["age"])
    sub = kept[kept["cls"].isin([lm.reference, lm.positive])]
    strata = []
    for b in range(len(S.S4_AGE_EDGES) + 1):
        m = sub[age_tertile(sub["age"]) == b]
        pos, neg = _contrast(m, score_col, lm)
        r = _auc_ci(pos, neg, min_each=3)
        strata.append({"tertile": b, **{k: r[k] for k in ("auc", "ci95", "n_pos", "n_neg")}})
    out["auc_by_age_tertile"] = strata
    out["benchmarks"] = benchmarks(df, label_map)
    return out


def benchmarks(df, label_map="cdr_3class"):
    """Demographics and labels only (no model output, no QC filter): an age + sex logistic model scored by repeated
    stratified cross-validation, and the AUC of nWBV alone (lower nWBV = more impaired, a sign fixed in advance)."""
    lm = labelmaps.get(label_map)
    sub = df[df["cls"].isin([lm.reference, lm.positive])]
    y = (sub["cls"] == lm.positive).to_numpy(int)
    X = np.column_stack([sub["age"].to_numpy(float), (sub["sex"] == "M").to_numpy(float)])
    out = {"age_sex_logistic_repeated_cv": S.repeated_cv_auc(X, y), "n_cases": int(len(sub))}
    ok = sub["nwbv"].notna().to_numpy()
    out["nwbv_alone"] = {**S.bootstrap_auc((-sub["nwbv"][ok & (y == 1)]).to_numpy(float), (-sub["nwbv"][ok & (y == 0)]).to_numpy(float)),
                         "n_missing_dropped": int((~ok).sum()), "sign": "lower nWBV scores as more impaired"}
    return out


# ----------------------------------------------------------------------------------------- S5 and S6
def s5(df, label_map="cdr_3class", adni_ref=None):
    """MRI specialist P(AD): reference vs positive class, volume source stated, shift against the ADNI references."""
    lm = labelmaps.get(label_map)
    kept, _ = qc_policy(df, "p_ad")
    pos, neg = _contrast(kept, "p_ad", lm)
    res = S.bootstrap_auc(pos, neg)
    res["volume_source"] = sorted({str(v) for v in kept["volume_provider"].dropna().unique()}) or ["not stated"]
    res["note"] = "model probability; volumes from this source, not the FreeSurfer 5.1 volumes the specialist was trained on"
    if adni_ref:
        ref = adni_ref["MRI_specialist_Pad"]
        res["adni_reference"] = {"cn_vs_ad_auc": ref["CN_vs_AD"]["auc"], "ad_vs_rest_auc": ref["AD_vs_rest"]["auc"],
                                 "shift_vs_cn_vs_ad": res["auc"] - ref["CN_vs_AD"]["auc"]}
    return res


def s6c_include(df, score_col="alz", label_map="cdr_3class"):
    """Include every case the engine could score, whatever its verdict, plus the worst-case bound."""
    lm = labelmaps.get(label_map)
    scorable = df[df[score_col].notna()]
    pos, neg = _contrast(scorable, score_col, lm)
    return {"include_all_scorable": {**S.bootstrap_auc(pos, neg), "n_scored": int(len(scorable)),
                                     "n_not_scored": int(len(df) - len(scorable))},
            "worst_case_bound": h1_worst_case(df, score_col, label_map)}


def s6d_sex(df, score_col="alz", label_map="cdr_3class"):
    lm = labelmaps.get(label_map)
    kept, _ = qc_policy(df, score_col)
    out = {}
    for sex in ("F", "M"):
        pos, neg = _contrast(kept[kept["sex"] == sex], score_col, lm)
        out[sex] = _auc_ci(pos, neg)
    return out


def s6e_age(df, score_col="alz", label_map="cdr_3class"):
    lo, hi = S.AGE_RESTRICT
    lm = labelmaps.get(label_map)
    kept, _ = qc_policy(df, score_col)
    m = kept[(kept["age"] >= lo) & (kept["age"] <= hi)]
    pos, neg = _contrast(m, score_col, lm)
    return {**_auc_ci(pos, neg), "age_range": [lo, hi], "n_excluded_by_age": int(len(kept) - len(m))}


def s6f_subset(df, label_map="cdr_3class"):
    """Case ids of the S6f subset: every positive-class case plus the first S6F_N_CDR0 reference-class cases in sorted
    case-id order. The rule uses labels, so it is applied only after the main run is complete."""
    lm = labelmaps.get(label_map)
    d = df.sort_values("case_id")
    ids = list(d.loc[d["cls"] == lm.positive, "case_id"]) + list(d.loc[d["cls"] == lm.reference, "case_id"][:S.S6F_N_CDR0])
    return sorted(ids)


def s6f(df, mirrored, label_map="cdr_3class", score_col="alz"):
    """Paired comparison of the primary and the mirrored-convention scores on the S6f subset. Descriptive only."""
    lm = labelmaps.get(label_map)
    ids = s6f_subset(df, label_map)
    miss = set(ids) - set(mirrored["case_id"])
    if miss:
        raise ValueError(f"the mirrored run lacks {len(miss)} cases of the S6f subset")
    a = df[df["case_id"].isin(ids)].set_index("case_id")
    b = mirrored.set_index("case_id").loc[a.index]
    both = a["qc"].isin(KEEP_QC) & b["qc"].isin(KEEP_QC) & a[score_col].notna() & b[score_col].notna()
    a, b = a[both], b[both]
    d = (b[score_col] - a[score_col]).to_numpy(float)
    pos_a, neg_a = a.loc[a["cls"] == lm.positive, score_col], a.loc[a["cls"] == lm.reference, score_col]
    pos_b, neg_b = b.loc[a["cls"] == lm.positive, score_col], b.loc[a["cls"] == lm.reference, score_col]
    rho = S.spearman(a[score_col], b[score_col])[0] if len(a) > 2 else None
    return {"n_subset": len(ids), "n_paired_kept": int(both.sum()),
            "auc_primary_convention": _auc_ci(pos_a, neg_a), "auc_mirrored_convention": _auc_ci(pos_b, neg_b),
            "paired_difference_mirrored_minus_primary": {
                "median": float(np.median(d)), "q25": float(np.percentile(d, 25)), "q75": float(np.percentile(d, 75)),
                "n_mirrored_higher": int((d > 0).sum()), "n_lower": int((d < 0).sum())},
            "spearman_between_conventions": rho, "rule": "every positive-class case plus the first 30 reference-class cases by sorted case id",
            "labelled": "sensitivity; never replaces the primary result"}


# ----------------------------------------------------------------------------------------- descriptive output
def describe(df, cols=("alz", "p_ad", "bag", "corr_vs_mni")):
    """Score distributions by class over the primary-analysis rows (aggregate statistics only)."""
    out = {}
    for col in cols:
        if col not in df.columns:
            continue
        kept, _ = qc_policy(df, col)
        out[col] = {}
        for cls, g in kept.groupby("cls"):
            v = g[col].to_numpy(float)
            out[col][cls] = {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                             "median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
                             "q75": float(np.percentile(v, 75)), "min": float(v.min()), "max": float(v.max())}
    return out


def qc_counts(df):
    out = {}
    for cls, g in df.groupby("cls"):
        out[cls] = {q: int((g["qc"] == q).sum()) for q in KNOWN_QC}
        out[cls]["n"] = int(len(g))
    return out


def adni_comparison(res, adni_ref):
    """Side by side with the ADNI train-split references (context only; no pass/fail)."""
    a = adni_ref["ALZ"]
    out = {"alz": {"oasis_h1_auc": res["H1"]["auc"], "adni_cn_vs_ad_auc": a["CN_vs_AD"]["auc"],
                   "adni_ad_vs_rest_auc": a["AD_vs_rest"]["auc"], "difference_vs_cn_vs_ad": res["H1"]["auc"] - a["CN_vs_AD"]["auc"],
                   "adni_group_means": {g: v["mean"] for g, v in a["by_group"].items()}},
           "bag": {"adni_cn_vs_ad_auc": adni_ref["BAG"]["CN_vs_AD"]["auc"],
                   "adni_group_means_years": {g: v["mean"] for g, v in adni_ref["BAG"]["by_group"].items()}}}
    return out


def constants():
    return {"seed": S.SEED, "n_boot": S.N_BOOT, "h1_pass_auc": S.H1_PASS_AUC, "h1_pass_ci_lower_above": S.H1_PASS_CI_LOW,
            "h1_fail_auc_below": S.H1_FAIL_AUC, "s2_alpha_one_sided": S.S2_ALPHA, "s3_rho_at_most": S.S3_RHO_MAX,
            "s4_age_edges": list(S.S4_AGE_EDGES), "s6e_age_range": list(S.AGE_RESTRICT), "s6f_n_reference": S.S6F_N_CDR0,
            "cv_folds": S.CV_FOLDS, "cv_repeats": S.CV_REPEATS}


def run_all(df, adni_ref=None, mirrored=None, label_map="cdr_3class"):
    """The whole protocol analysis as one aggregate dictionary."""
    d = add_classes(normalise_qc(df), label_map)
    out = {"constants": constants(), "H1": h1(d, "alz", label_map), "H1_worst_case": h1_worst_case(d, "alz", label_map),
           "S1": s1(d, "alz", label_map)}
    for col in ("alz", "bag"):
        out[f"S2_{col}"] = s2_trend(d, col, label_map)
        out[f"S3_{col}"] = s3_mmse(d, col)
    out["S4"] = s4(d, "alz", label_map)
    if "p_ad" in d.columns:
        out["S5"] = s5(d, label_map, adni_ref)
    out["S6c"] = s6c_include(d, "alz", label_map)
    out["S6d"] = s6d_sex(d, "alz", label_map)
    out["S6e"] = s6e_age(d, "alz", label_map)
    if mirrored is not None:
        out["S6f"] = s6f(d, normalise_qc(mirrored), label_map)
    out["descriptive"] = {"score_distributions_by_class": describe(d), "qc_counts_by_class": qc_counts(d)}
    if adni_ref:
        out["descriptive"]["adni_reference_comparison"] = adni_comparison(out, adni_ref)
    return out


def main(argv=None):
    import argparse
    import subprocess
    from .aggregate_io import write_json
    ap = argparse.ArgumentParser(description="Stage D analysis. Refuses to open labels before RUN_COMPLETE. Aggregate output only.")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-json", required=True, help="under docs/validation/")
    ap.add_argument("--mirrored-run-root")
    ap.add_argument("--mirrored-manifest")
    a = ap.parse_args(argv)
    df = assemble(a.run_root, a.manifest)
    mirrored = load_scores_only(a.mirrored_run_root, a.mirrored_manifest) if a.mirrored_run_root else None
    adni = json.loads(ADNI_REFERENCE.read_text())
    res = run_all(df, adni, mirrored)
    res["provenance"] = {"analysis_code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                                               text=True).stdout.strip(), "n_cases": int(len(df))}
    dest = write_json(a.out_json, res, case_ids=list(df["case_id"]))
    print(f"H1: {res['H1']['verdict']}; written: {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
