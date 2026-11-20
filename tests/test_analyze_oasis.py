"""Stage D analysis (src/validation/analyze_oasis.py + stats.py) on SYNTHETIC data only: constants against the protocol
text, H1 boundaries, known-AUC simulations and CI coverage, trends, partial Spearman against scipy, cross-validated
benchmarks, S4-S6 rules, aggregate-only output and the blindness refusal. No OASIS data of any kind."""
import csv
import inspect
import json
import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from scipy.stats import norm

from src.assessment import batch as B
from src.assessment import labelmaps
from src.validation import aggregate_io as AIO
from src.validation import analyze_oasis as AN
from src.validation import stats as S

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL = (ROOT / "docs" / "external_validation_protocol.md").read_text()
OPENED = []


def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        if os.path.basename(os.fsdecode(args[0])) == "labels.csv":
            OPENED.append(os.fsdecode(args[0]))


sys.addaudithook(_hook)


# ------------------------------------------------------------------------------------------------ synthetic cohort
def cohort(seed=1, n=(98, 70, 30), means=(0.38, 0.46, 0.64), sd=0.17, n_warn=5, n_fail=4):
    """A 198-row table shaped like the real one (case ids interleave the classes, so id order carries no label)."""
    rng = np.random.default_rng(seed)
    cdr = np.concatenate([np.full(c, v) for c, v in zip(n, (0.0, 0.5, 1.0))])
    cdr = cdr[rng.permutation(len(cdr))]
    N = len(cdr)
    rank = np.where(cdr == 0, 0, np.where(cdr == 0.5, 1, 2))
    age = np.clip(np.round(rng.normal(76 + 1.5 * rank, 8)), 60, 96)
    mu = np.array(means)[rank]
    alz = np.clip(rng.normal(mu, sd), 0, 1)
    df = pd.DataFrame({
        "case_id": [f"X{i + 1:04d}" for i in range(N)], "alz": alz,
        "p_ad": np.clip(rng.normal(0.1 + 0.25 * rank, 0.12), 0, 1),
        "bag": rng.normal(2.5 * rank, 5), "corr_vs_mni": rng.normal(0.5, 0.05, N), "qc": "PASS",
        "volume_provider": "synthseg", "cdr": cdr, "mmse": np.clip(np.round(29 - 3.5 * rank + rng.normal(0, 1.5, N)), 10, 30),
        "etiv": rng.normal(1450, 100, N), "nwbv": rng.normal(0.74 - 0.015 * rank, 0.03), "age": age,
        "sex": rng.choice(["F", "M"], N, p=[0.62, 0.38])})
    df.loc[df.index[:n_warn], "qc"] = "WARN"
    bad = df.index[df["cdr"] == 1.0][-n_fail:]                          # the engine withheld the MRI of four dementia cases
    df.loc[bad, ["alz", "p_ad", "bag", "corr_vs_mni"]] = np.nan
    df.loc[bad, "qc"] = "FAIL"
    return df


@pytest.fixture(scope="module")
def df():
    return cohort()


@pytest.fixture(scope="module")
def adni():
    return json.loads(AN.ADNI_REFERENCE.read_text())


@pytest.fixture(scope="module")
def result(df, adni):
    return AN.run_all(df, adni)


# ------------------------------------------------------------------------------------------------ constants vs protocol
def test_constants_match_the_protocol_text():
    sec7 = PROTOCOL[PROTOCOL.index("## 7."):PROTOCOL.index("## 8.")]
    m = re.search(r"\*\*PASS:\*\* AUC >= ([0-9]+(?:\.[0-9]+)?) and CI lower bound > ([0-9]+(?:\.[0-9]+)?)", sec7)
    assert (float(m.group(1)), float(m.group(2))) == (S.H1_PASS_AUC, S.H1_PASS_CI_LOW)
    assert float(re.search(r"\*\*FAIL:\*\* AUC < ([0-9]+(?:\.[0-9]+)?)", sec7).group(1)) == S.H1_FAIL_AUC
    assert "includes AUC < 0.5" in sec7 and "inverted" in sec7
    assert float(re.search(r"PASS if rho <= (-[0-9]+(?:\.[0-9]+)?) for ALZ", sec7).group(1)) == S.S3_RHO_MAX
    assert int(re.search(r"bootstrap seed ([0-9]+)", sec7).group(1)) == S.SEED
    assert int(re.search(r"([0-9]+) resamples", sec7).group(1)) == S.N_BOOT
    assert float(re.search(r"one-sided Jonckheere-Terpstra \(alpha ([0-9]+(?:\.[0-9]+)?)\)", sec7).group(1)) == S.S2_ALPHA
    assert tuple(float(x) for x in re.search(r"age restricted to ([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)", sec7).groups()) == S.AGE_RESTRICT
    e = re.search(r"e1 = ([0-9]+), e2 = ([0-9]+)", PROTOCOL)
    assert (int(e.group(1)), int(e.group(2))) == S.S4_AGE_EDGES
    assert f"first {S.S6F_N_CDR0} CDR 0 cases" in sec7
    cv = re.search(r"repeated stratified ([0-9]+)-fold cross-validation, ([0-9]+) repeats, L2 penalty ([0-9]+(?:\.[0-9]+)?)", sec7)
    assert tuple(float(x) for x in cv.groups()) == (S.CV_FOLDS, S.CV_REPEATS, S.LOGIT_L2)
    stage_a = json.loads((ROOT / "docs" / "validation" / "stage_a_counts.json").read_text())["s4_age_tertiles"]["edges"]
    assert tuple(stage_a) == tuple(float(x) for x in S.S4_AGE_EDGES)


def test_constants_are_defined_once():
    src = (ROOT / "src" / "validation" / "analyze_oasis.py").read_text()
    for literal in ("0.75", "0.65", "20261113", "2000", "-0.30", "90.3", "(71, 80)"):
        assert literal not in src, literal                              # analyze_oasis only refers to stats.py


# ------------------------------------------------------------------------------------------------ H1
@pytest.mark.parametrize("auc_v,lo,verdict", [
    (0.75, 0.6500001, "PASS"), (0.75, 0.65, "PARTIAL"),          # CI lower exactly 0.65 is not > 0.65
    (0.7499999, 0.9, "PARTIAL"), (0.9, 0.64, "PARTIAL"), (0.65, 0.5, "PARTIAL"),
    (0.6499999, 0.6, "FAIL"), (0.5, 0.4, "FAIL"), (0.4999, 0.3, "FAIL"), (0.1, 0.0, "FAIL")])
def test_h1_boundaries(auc_v, lo, verdict):
    assert S.h1_verdict(auc_v, lo) == verdict


def test_auc_exactly_0_75_from_data_and_inverted_flag():
    pos, neg = [3, 5, 6.5, 8], [1, 2, 4, 7]                          # 12 of 16 pairs ordered
    assert S.auc(pos, neg) == 0.75
    d = pd.DataFrame({"case_id": [f"X{i}" for i in range(8)], "alz": pos + neg, "qc": "PASS",
                      "cdr": [1.0] * 4 + [0.0] * 4})
    h = AN.h1(AN.add_classes(d, "cdr_3class"))
    assert h["auc"] == 0.75 and not h["inverted"]


def test_h1_uses_the_ci_lower_bound_not_the_point_estimate_alone(df, monkeypatch):
    d = AN.add_classes(AN.normalise_qc(df), "cdr_3class")
    for lo, expect in ((0.6500001, "PASS"), (0.65, "PARTIAL")):
        monkeypatch.setattr(AN.S, "bootstrap_auc", lambda pos, neg, lo=lo: {"auc": 0.75, "ci95": [lo, 0.85], "n_pos": 1, "n_neg": 1})
        assert AN.h1(d)["verdict"] == expect


def test_inverted_scores_fail_and_are_flagged(df):
    d = df.copy()
    d["alz"] = 1 - d["alz"]
    h = AN.h1(AN.add_classes(AN.normalise_qc(d), "cdr_3class"))
    assert h["inverted"] and h["verdict"] == "FAIL" and h["auc"] < 0.35


def test_swapped_labels_look_inverted(df):
    d = df.copy()
    d["cdr"] = d["cdr"].map({0.0: 1.0, 0.5: 0.5, 1.0: 0.0})        # a labelling mistake: the classes swapped
    h = AN.h1(AN.add_classes(AN.normalise_qc(d), "cdr_3class"))
    assert h["inverted"] and h["verdict"] == "FAIL"


def test_class_order_matches_the_cdr_scale():
    lm = labelmaps.get("cdr_3class")
    assert lm.classes == ("CN-like", "CDR0.5", "dementia") and (lm.reference, lm.positive) == ("CN-like", "dementia")
    assert [lm.rank(v) for v in (0, 0.5, 1, 2, 3)] == [0, 1, 2, 2, 2]


# ------------------------------------------------------------------------------------------------ AUC, bootstrap, simulations
def test_known_auc_is_recovered_and_the_ci_covers_it():
    rng = np.random.default_rng(10)
    for d in (0.5, 1.0, 2.0):
        truth = norm.cdf(d / np.sqrt(2))
        a = S.auc(rng.normal(d, 1, 6000), rng.normal(0, 1, 6000))
        assert a == pytest.approx(truth, abs=0.012)


def test_percentile_bootstrap_coverage_over_many_simulations():
    rng = np.random.default_rng(11)
    truth = norm.cdf(1.0 / np.sqrt(2))                              # 0.760, with 30 vs 98 like the cohort
    hit, n_sim = 0, 400
    for _ in range(n_sim):
        r = S.bootstrap_auc(rng.normal(1.0, 1, 30), rng.normal(0, 1, 98), seed=int(rng.integers(1 << 30)), n_boot=400)
        hit += r["ci95"][0] <= truth <= r["ci95"][1]
    assert 0.91 <= hit / n_sim <= 0.99, hit / n_sim


def test_bootstrap_is_deterministic_with_the_fixed_seed():
    rng = np.random.default_rng(3)
    p, n = rng.normal(0.8, 1, 30), rng.normal(0, 1, 90)
    a, b = S.bootstrap_auc(p, n), S.bootstrap_auc(p, n)
    assert a == b and a["seed"] == S.SEED == 20261113 and a["n_boot"] == S.N_BOOT == 2000
    assert S.bootstrap_auc(p, n, seed=S.SEED) == a and S.bootstrap_auc(p, n, seed=S.SEED + 1)["ci95"] != a["ci95"]
    r1, r2 = S.bootstrap_spearman(p, p[::-1] + 0.1 * n[:30]), S.bootstrap_spearman(p, p[::-1] + 0.1 * n[:30])
    assert r1 == r2


def test_bootstrap_matches_a_naive_resampling_implementation_in_distribution():
    rng = np.random.default_rng(4)
    p, n = rng.normal(1, 1, 30), rng.normal(0, 1, 98)
    fast = S.bootstrap_auc(p, n, n_boot=4000)
    r = np.random.default_rng(99)
    naive = np.percentile([S.auc(p[r.integers(0, 30, 30)], n[r.integers(0, 98, 98)]) for _ in range(4000)], [2.5, 97.5])
    assert fast["ci95"] == pytest.approx(list(naive), abs=0.02)


def test_ties_and_degenerate_inputs():
    assert S.auc([1, 1, 1], [1, 1, 1]) == 0.5 and S.bootstrap_auc([1, 1, 1], [1, 1, 1])["ci95"] == [0.5, 0.5]
    assert S.auc([1, 2, 2], [2, 2, 3]) == pytest.approx(2 / 9)               # only the two (2, 2) ties of each 2 count half
    assert S.auc([2, 2], [1, 2]) == pytest.approx((1 + 0.5 + 1 + 0.5) / 4)
    with pytest.raises(ValueError):
        S.auc([], [1.0])
    with pytest.raises(ValueError):
        S.auc([1.0, np.nan], [0.0])


# ------------------------------------------------------------------------------------------------ S1, S2, S3
def test_h1_s1_counts_and_exclusions(df, result):
    h = result["H1"]
    assert h["contrast"] == "CN-like vs dementia" and h["n_pos"] == 26 and h["n_neg"] == 98 and h["verdict"] == "PASS"
    assert h["qc"]["n_total"] == 198 and h["qc"]["n_kept"] == 194 and h["qc"]["excluded_by_class_and_qc"] == {"dementia|FAIL": 4}
    assert result["H1_worst_case"]["n_filled"] == 4 and result["H1_worst_case"]["auc"] <= h["auc"] + 0.02
    assert result["S1"]["CDR0.5"]["auc"] > 0.5 and result["S1"]["CDR0.5"]["direction_higher_in_intermediate"]


def test_trend_known_monotone_flat_and_reversed():
    rng = np.random.default_rng(6)

    def table(means):
        rows = []
        for r, m in enumerate(means):
            for i in range(60):
                rows.append({"case_id": f"X{len(rows):04d}", "alz": rng.normal(m, 0.1), "qc": "PASS", "cdr": [0.0, 0.5, 1.0][r],
                             "mmse": 20.0, "age": 75.0, "sex": "F"})
        return AN.add_classes(pd.DataFrame(rows), "cdr_3class")
    up = AN.s2_trend(table((0.3, 0.45, 0.6)), "alz")
    assert up["outcome"] == "PASS" and up["jt"]["p"] < 1e-10 and up["direction_matches"] and up["monotone_medians"]
    down = AN.s2_trend(table((0.6, 0.45, 0.3)), "alz")
    assert down["outcome"] == "NOT MET" and down["jt"]["p"] > 0.99 and not down["direction_matches"]
    same = pd.DataFrame({"case_id": [f"X{i:04d}" for i in range(180)], "alz": np.tile(np.linspace(0.1, 0.9, 60), 3), "qc": "PASS",
                         "cdr": np.repeat([0.0, 0.5, 1.0], 60), "mmse": 20.0, "age": 75.0, "sex": "F"})
    flat = AN.s2_trend(AN.add_classes(same, "cdr_3class"), "alz")             # identical distributions: no trend at all
    assert flat["jt"]["z"] == pytest.approx(0, abs=1e-9) and flat["jt"]["p"] == pytest.approx(0.5) and flat["outcome"] == "NOT MET"
    nonmono = AN.s2_trend(table((0.3, 0.7, 0.5)), "alz")                       # significant trend but medians not monotone
    assert nonmono["monotone_medians"] is False and nonmono["outcome"] == "NOT MET"


def test_s2_and_s3_use_ranks_only(df):
    d = AN.add_classes(AN.normalise_qc(df), "cdr_3class")
    e = d.copy()
    e["bag"] = np.exp(e["bag"] / 5) * 100 + 3                         # any monotone transform
    a, b = AN.s2_trend(d, "bag"), AN.s2_trend(e, "bag")
    assert a["jt"]["z"] == pytest.approx(b["jt"]["z"]) and a["spearman_vs_rank"]["rho"] == pytest.approx(b["spearman_vs_rank"]["rho"])
    assert AN.s3_mmse(d, "bag")["rho"] == pytest.approx(AN.s3_mmse(e, "bag")["rho"])


def test_s3_threshold_is_inclusive_and_alz_only(df, monkeypatch):
    d = AN.add_classes(AN.normalise_qc(df), "cdr_3class")
    for rho, expect in ((-0.30, "PASS"), (-0.2999, "NOT MET"), (-0.8, "PASS"), (0.2, "NOT MET")):
        monkeypatch.setattr(AN.S, "bootstrap_spearman", lambda x, y, rho=rho: {"rho": rho, "ci95": [rho - 0.1, rho + 0.1], "n": 3})
        assert AN.s3_mmse(d, "alz")["outcome"] == expect
        assert "outcome" not in AN.s3_mmse(d, "bag")                      # BAG is descriptive


# ------------------------------------------------------------------------------------------------ S4
def test_partial_spearman_matches_a_scipy_computation():
    rng = np.random.default_rng(7)
    z = rng.normal(size=120)
    x = z + rng.normal(size=120)
    y = 0.7 * z + 0.4 * x + rng.normal(size=120)
    rxy, rxz, ryz = (sps.spearmanr(a, b)[0] for a, b in ((x, y), (x, z), (y, z)))
    expect = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    assert S.partial_spearman(x, y, z) == pytest.approx(expect, abs=1e-12)
    ty = sps.rankdata(y)                                              # and via residuals of ranks with scipy's linregress
    rx_res = sps.rankdata(x) - np.polyval(np.polyfit(sps.rankdata(z), sps.rankdata(x), 1), sps.rankdata(z))
    ry_res = ty - np.polyval(np.polyfit(sps.rankdata(z), ty, 1), sps.rankdata(z))
    assert S.partial_spearman(x, y, z) == pytest.approx(sps.pearsonr(rx_res, ry_res)[0], abs=1e-12)
    assert abs(S.partial_spearman(x, y, z) - sps.spearmanr(x, y)[0]) > 0.01      # adjustment matters here


def test_age_tertiles_are_right_closed_at_the_recorded_edges():
    assert S.S4_AGE_EDGES == (71, 80)
    ages = [60, 70, 71, 72, 79, 80, 81, 96]
    assert list(AN.age_tertile(ages)) == [0, 0, 0, 1, 1, 1, 2, 2]           # 71 -> first group, 80 -> middle group
    out = S.stratified_auc([1, 2, 3, 4, 5, 6.0], [1, 0, 1, 0, 1, 0], [71, 72, 80, 81, 60, 95], list(S.S4_AGE_EDGES))
    assert [o["n_pos"] + o["n_neg"] for o in out] == [2, 2, 2]


def test_s4_blocks(result):
    s = result["S4"]
    assert s["age_edges"] == [71, 80] and len(s["auc_by_age_tertile"]) == 3
    assert sum(t["n_pos"] + t["n_neg"] for t in s["auc_by_age_tertile"]) == 98 + 26
    assert s["partial_spearman_vs_rank_age_adjusted"]["rho"] > 0.3
    assert s["partial_spearman_vs_mmse_age_adjusted"]["rho"] < -0.3
    b = s["benchmarks"]
    assert b["n_cases"] == 98 + 30 and b["age_sex_logistic_repeated_cv"]["repeats"] == 20 and b["age_sex_logistic_repeated_cv"]["folds"] == 5
    assert b["nwbv_alone"]["auc"] > 0.6 and b["nwbv_alone"]["sign"].startswith("lower nWBV")


def test_logistic_fit_matches_a_closed_form_check_and_sklearn_when_available():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(150, 3))
    y = (X @ [1.0, -0.5, 0.0] + rng.logistic(size=150) > 0).astype(int)
    m = S.fit_logistic(X, y, l2=1.0)
    p = m(X)
    A = np.column_stack([np.ones(150), (X - X.mean(0)) / X.std(0)])
    grad = A.T @ (p - y) + np.diag([0, 1, 1, 1]) @ m.coef                  # zero gradient at the optimum
    assert np.abs(grad).max() < 1e-6
    sk = pytest.importorskip("sklearn.linear_model")
    ref = sk.LogisticRegression(C=1.0, tol=1e-12, max_iter=1000).fit((X - X.mean(0)) / X.std(0), y)
    assert np.abs(ref.predict_proba((X - X.mean(0)) / X.std(0))[:, 1] - p).max() < 1e-5


def test_cv_benchmark_is_not_optimistic_and_never_predicts_a_subject_from_a_model_that_saw_it(monkeypatch):
    rng = np.random.default_rng(9)
    n = 70
    X = rng.normal(size=(n, 30))                                           # 30 noise features, 70 subjects
    y = rng.integers(0, 2, n)
    fitted = S.fit_logistic(X, y, l2=1e-2)(X)
    insample = S.auc(fitted[y == 1], fitted[y == 0])
    cv = S.repeated_cv_auc(X, y, l2=1e-2, n_boot=200)
    assert insample > 0.9 and cv["auc_per_repeat_mean"] < 0.62            # in-sample fit is flattering, CV is not
    informative = S.repeated_cv_auc(np.column_stack([y + rng.normal(0, 1, n), rng.normal(size=n)]), y, n_boot=200)
    assert informative["auc"] > 0.7
    # leakage check: tag rows by index and record who trains and who is predicted in each fit
    Xi = np.column_stack([np.arange(n, dtype=float), rng.normal(size=n)])
    seen = []
    real = S.fit_logistic

    def spy(Xtr, ytr, l2):
        tr = set(Xtr[:, 0].astype(int))

        def predict(Xte):
            assert not (tr & set(Xte[:, 0].astype(int)))
            seen.append(len(Xte))
            return real(Xtr, ytr, l2)(Xte)
        return predict
    monkeypatch.setattr(S, "fit_logistic", spy)
    S.repeated_cv_auc(Xi, y, repeats=3, n_boot=50)
    assert sum(seen) == 3 * n                                              # every subject predicted once per repeat
    assert S.repeated_cv_auc(Xi, y, repeats=2, n_boot=50) == S.repeated_cv_auc(Xi, y, repeats=2, n_boot=50)


def test_cv_refuses_too_few_cases():
    with pytest.raises(ValueError):
        S.repeated_cv_auc(np.zeros((5, 2)), np.array([1, 0, 0, 0, 0]))


# ------------------------------------------------------------------------------------------------ S5, S6
def test_s5_states_the_volume_source_and_the_shift(result, adni):
    s = result["S5"]
    assert s["volume_source"] == ["synthseg"] and "model probability" in s["note"]
    assert s["adni_reference"]["cn_vs_ad_auc"] == adni["MRI_specialist_Pad"]["CN_vs_AD"]["auc"]
    assert s["adni_reference"]["shift_vs_cn_vs_ad"] == pytest.approx(s["auc"] - s["adni_reference"]["cn_vs_ad_auc"])


def test_s6c_includes_everything_scorable_and_bounds_the_failures(df):
    d = AN.add_classes(AN.normalise_qc(df), "cdr_3class")
    row = d.index[d["cls"] == "CN-like"][0]
    d.loc[row, "qc"] = "FAIL"                                          # a FAIL row that still has a score
    s = AN.s6c_include(d)
    assert s["include_all_scorable"]["n_scored"] == 194 and s["include_all_scorable"]["n_not_scored"] == 4
    assert s["worst_case_bound"]["n_filled"] == 5 and "median" in s["worst_case_bound"]["rule"]
    assert s["include_all_scorable"]["n_pos"] + s["include_all_scorable"]["n_neg"] == 26 + 98
    k = AN.h1(d)
    assert k["n_pos"] + k["n_neg"] == 26 + 98 - 1                      # the primary analysis drops the FAIL row


def test_worst_case_ties_failures_at_the_median():
    d = pd.DataFrame({"case_id": [f"X{i}" for i in range(6)], "alz": [0.9, 0.8, np.nan, 0.1, 0.2, np.nan],
                      "qc": ["PASS", "PASS", "FAIL", "PASS", "PASS", "failed"], "cdr": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]})
    d = AN.add_classes(d, "cdr_3class")
    w = AN.h1_worst_case(d)
    # kept scores 0.9 0.8 0.1 0.2 -> median 0.5; positives 0.9 0.8 0.5, negatives 0.1 0.2 0.5
    assert w["auc"] == pytest.approx(8.5 / 9) and w["n_filled"] == 2           # 0.9, 0.8 and 0.5 beat or tie 0.1, 0.2, 0.5


def test_s6d_sex_and_s6e_age_bounds(df):
    d = AN.add_classes(AN.normalise_qc(df), "cdr_3class")
    sx = AN.s6d_sex(d)
    assert set(sx) == {"F", "M"} and sx["F"]["n_neg"] + sx["M"]["n_neg"] == 98
    e = d.copy()
    e.loc[e.index[0], "age"] = 90.3
    e.loc[e.index[1], "age"] = 90.31
    e.loc[e.index[2], "age"] = 59.99
    e[["qc"]] = "PASS"
    r = AN.s6e_age(e)
    lo, hi = S.AGE_RESTRICT
    kept = e[e["alz"].notna()]
    assert r["n_excluded_by_age"] == int(((kept["age"] < lo) | (kept["age"] > hi)).sum()) and r["age_range"] == [60.0, 90.3]
    assert (kept["age"] == 90.3).any() and r["n_pos"] + r["n_neg"] == int(((kept["age"] >= lo) & (kept["age"] <= hi) & kept["cls"].isin(["CN-like", "dementia"])).sum())


def test_s6f_subset_rule_and_paired_differences(df):
    d = AN.add_classes(AN.normalise_qc(df), "cdr_3class")
    ids = AN.s6f_subset(d)
    pos = set(d.loc[d["cls"] == "dementia", "case_id"])
    first30 = list(d.sort_values("case_id").loc[d["cls"] == "CN-like", "case_id"][:30])
    assert len(ids) == 30 + 30 and set(ids) == pos | set(first30) and ids == sorted(ids)
    assert len(set(first30)) == 30 and max(first30) < max(d.loc[d["cls"] == "CN-like", "case_id"])   # not simply the last 30
    shuffled = d.sample(frac=1, random_state=3)
    assert AN.s6f_subset(shuffled) == ids                                    # independent of row order
    same = AN.s6f(d, d[["case_id", "alz", "qc"]])
    assert same["paired_difference_mirrored_minus_primary"]["median"] == 0 and same["spearman_between_conventions"] == pytest.approx(1.0)
    assert same["auc_primary_convention"]["auc"] == same["auc_mirrored_convention"]["auc"]
    flipped = d[["case_id", "alz", "qc"]].copy()
    flipped["alz"] = 1 - flipped["alz"]
    f = AN.s6f(d, flipped)
    assert f["auc_mirrored_convention"]["auc"] == pytest.approx(1 - f["auc_primary_convention"]["auc"])
    assert f["labelled"].startswith("sensitivity")
    with pytest.raises(ValueError):
        AN.s6f(d, d[["case_id", "alz", "qc"]].iloc[:100])


# ------------------------------------------------------------------------------------------------ missing QC, descriptive, output
def test_missing_or_unknown_qc_counts_as_failed_and_is_excluded(df):
    d = df.copy()
    d["qc"] = d["qc"].astype(object)
    d.loc[d.index[20], "qc"] = None
    d.loc[d.index[21], "qc"] = "mystery"
    d.loc[d.index[22], "qc"] = np.nan
    n = AN.normalise_qc(d)
    assert list(n.loc[n.index[[20, 21, 22]], "qc"]) == ["failed"] * 3
    res = AN.h1(AN.add_classes(n, "cdr_3class"))
    assert res["qc"]["n_kept"] == 194 - 3 and res["qc"]["n_total"] == 198      # rows 20-22 had scores but no usable verdict


def test_a_class_with_no_scored_cases_raises_a_clear_error(df):
    d = df.copy()
    d.loc[d["cdr"] >= 1, "qc"] = "FAIL"
    with pytest.raises(ValueError, match="both classes"):
        AN.h1(AN.add_classes(AN.normalise_qc(d), "cdr_3class"))


def test_descriptive_block_and_adni_comparison(result, adni):
    ds = result["descriptive"]
    assert set(ds["score_distributions_by_class"]["alz"]) == {"CN-like", "CDR0.5", "dementia"}
    q = ds["qc_counts_by_class"]
    assert sum(v["n"] for v in q.values()) == 198 and sum(v["PASS"] + v["WARN"] + v["FAIL"] + v["failed"] for v in q.values()) == 198
    c = ds["adni_reference_comparison"]["alz"]
    assert c["adni_cn_vs_ad_auc"] == adni["ALZ"]["CN_vs_AD"]["auc"] and c["difference_vs_cn_vs_ad"] == pytest.approx(result["H1"]["auc"] - c["adni_cn_vs_ad_auc"])


def test_output_is_aggregate_only_and_never_says_calibrated(result, df, tmp_path):
    AIO.check_aggregate(result, df["case_id"])
    p = AIO.write_json(tmp_path / "res.json", result, case_ids=list(df["case_id"]))
    text = p.read_text()
    assert not re.search(r"X\d{4}", text) and not AIO.ID_RE.search(text) and "calibrated" not in text.lower()
    for f in ("analyze_oasis.py", "stats.py"):
        src = (ROOT / "src" / "validation" / f).read_text()
        assert re.search(r"\bcalibrated\b", src) is None and "mri_target" not in src
    assert result["constants"]["h1_pass_auc"] == 0.75 and result["constants"]["seed"] == 20261113


def test_write_refuses_a_destination_inside_the_repo_outside_docs_validation():
    with pytest.raises(AIO.AggregateError):
        AIO.write_json(ROOT / "src" / "x.json", {"a": 1})
    with pytest.raises(AIO.AggregateError):
        AIO.write_json(ROOT / "docs" / "x.json", {"a": 1})
    with pytest.raises(AIO.AggregateError):
        AIO.write_json(ROOT / "docs" / "validation" / "x.json", {"id": "X0001"}, case_ids=["X0001"])


# ------------------------------------------------------------------------------------------------ blindness
def report(alz, p_ad, bag, qc="PASS"):
    return {"meta": {"mode": "MRI_ONLY", "volume_provider": "synthseg"},
            "ad": {"alz": alz, "model_probability": p_ad, "source": "mri_specialist"},
            "brain_age": {"corrected_gap": bag, "status": "ok"},
            "qc": {"overall": qc, "gates": [{"name": "registration corr_vs_mni", "value": 0.5, "status": qc}]}}


def make_run(tmp_path, n=12, finish=True, stage="full", name="run", man_name="manifest", subset_ids=None):
    """A synthetic finished run root plus manifest folder, written exactly as the runner writes them."""
    rng = np.random.default_rng(2)
    man, root = tmp_path / man_name, tmp_path / name
    man.mkdir(parents=True, exist_ok=True)
    (root / "cases").mkdir(parents=True, exist_ok=True)
    ids = subset_ids or [f"X{i + 1:04d}" for i in range(n)]
    cdr = {c: [0.0, 0.5, 1.0, 0.0][int(c[1:]) % 4] for c in ids}        # a class pattern fixed by the case number
    with open(man / "inputs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(B.EXTENDED_COLUMNS)
        for i, c in enumerate(ids):
            w.writerow([c, f"/data/{c}.hdr", "analyze75", "oasis1_primary", 70 + i, "F" if i % 2 else "M", 1, "synthseg"])
    with open(man / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "CDR", "MMSE", "eTIV", "nWBV"])
        for c in sorted(cdr):
            w.writerow([c, f"{cdr[c]:g}", 25 + int(c[1:]) % 3, 1400, f"{0.75 - 0.01 * (int(c[1:]) % 5):g}"])
    st = B.State(root / "state.jsonl")
    st.append({"type": "header", "n_cases": len(ids), "inputs_sha256": B.sha256_file(man / "inputs.csv"), "git_commit": "abc",
               "registry_sha256": "r", "mode": "MRI_ONLY", "stage": stage, "label_map": "cdr_3class"})
    n_ok = n_fail = 0
    for i, c in enumerate(ids):
        st.append({"type": "case", "case_id": c, "status": "start"})
        if i == 1:
            st.append({"type": "case", "case_id": c, "status": "fail", "error_class": "QCFail"})
            n_fail += 1
        elif i == 2:
            st.append({"type": "case", "case_id": c, "status": "fail", "error_class": "RuntimeError"})
            n_fail += 1
        else:
            (root / "cases" / c).mkdir()
            (root / "cases" / c / "report.json").write_text(json.dumps(report(float(rng.random()), float(rng.random()), float(rng.normal()))))
            st.append({"type": "case", "case_id": c, "status": "ok"})
            n_ok += 1
    if finish:
        st.append({"type": "footer", "n_ok": n_ok, "n_failed": n_fail, "n_cases": len(ids)})
        (root / ("RUN_COMPLETE" if stage == "full" else "PLUMBING_COMPLETE")).write_text(
            json.dumps({"n_ok": n_ok, "n_failed": n_fail, "n_cases": len(ids)}))
    return root, man


def opened_labels(fn):
    OPENED.clear()
    try:
        fn()
    except AN.BlindnessError:
        return OPENED.copy()
    raise AssertionError("expected a BlindnessError")


def test_assemble_reads_labels_only_after_a_complete_matching_run(tmp_path):
    root, man = make_run(tmp_path)
    OPENED.clear()
    d = AN.assemble(root, man, expected_labels_sha256=None)
    assert any(p.endswith("labels.csv") for p in OPENED)                       # positive control: the hook sees the open
    assert len(d) == 12 and set(d["qc"]) == {"PASS", "FAIL", "failed"} and d["alz"].isna().sum() == 2
    assert d.loc[d["case_id"] == "X0002", "qc"].item() == "FAIL" and d.loc[d["case_id"] == "X0003", "qc"].item() == "failed"
    assert list(d["case_id"]) == sorted(d["case_id"]) and set(d["sex"]) == {"F", "M"}


@pytest.mark.parametrize("what", ["no_marker", "plumbing", "unfinished", "hash", "footer", "start_left", "bad_marker", "labels_hash"])
def test_blindness_refusals_never_open_labels(tmp_path, what):
    root, man = make_run(tmp_path, finish=(what != "unfinished"), stage="plumbing" if what == "plumbing" else "full")
    if what == "no_marker":
        (root / "RUN_COMPLETE").unlink()
    elif what == "hash":
        (man / "inputs.csv").write_text((man / "inputs.csv").read_text().replace(",71,", ",72,"))
    elif what == "footer":
        (root / "RUN_COMPLETE").write_text(json.dumps({"n_ok": 99, "n_failed": 0, "n_cases": 12}))
    elif what == "start_left":
        B.State(root / "state.jsonl").append({"type": "case", "case_id": "X0005", "status": "start"})
    elif what == "bad_marker":
        (root / "RUN_COMPLETE").write_text("not json")
    exp = "0" * 64 if what == "labels_hash" else None
    if what == "labels_hash":
        with pytest.raises(AN.BlindnessError):                                  # the hash check is the only refusal here
            AN.assemble(root, man, expected_labels_sha256=exp)
        return
    assert opened_labels(lambda: AN.assemble(root, man, expected_labels_sha256=None)) == []


def test_default_labels_hash_is_the_stage_a_hash(tmp_path):
    root, man = make_run(tmp_path)
    with pytest.raises(AN.BlindnessError, match="Stage A"):                     # synthetic labels differ from the recorded hash
        AN.assemble(root, man)
    assert AN.stage_a_labels_sha256() == json.loads((ROOT / "docs" / "validation" / "stage_a_counts.json").read_text())["manifest_sha256"]["labels.csv"]


def test_mirrored_scores_need_their_own_complete_run_and_no_labels(tmp_path):
    main_root, main_man = make_run(tmp_path)
    sub = [f"X{i:04d}" for i in (1, 2, 4, 5, 6, 7)]
    m_root, m_man = make_run(tmp_path, name="mirror", man_name="mman", subset_ids=sub)
    OPENED.clear()
    s = AN.load_scores_only(m_root, m_man)
    assert list(s["case_id"]) == sub and not any(p.startswith(str(m_man)) for p in OPENED)
    (m_root / "RUN_COMPLETE").unlink()
    with pytest.raises(AN.BlindnessError):
        AN.load_scores_only(m_root, m_man)


def test_main_end_to_end_on_a_synthetic_run(tmp_path, monkeypatch):
    root, man = make_run(tmp_path, n=40)
    monkeypatch.setattr(AN, "stage_a_labels_sha256", lambda: None)
    out = tmp_path / "res.json"
    assert AN.main(["--run-root", str(root), "--manifest", str(man), "--out-json", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["H1"]["verdict"] in ("PASS", "PARTIAL", "FAIL") and res["provenance"]["n_cases"] == 40
    assert "S6f" not in res and not re.search(r"X\d{4}", out.read_text())


def test_the_runner_still_never_touches_the_analysis_side():
    assert "score_view" not in inspect.getsource(B) and "analyze_oasis" not in inspect.getsource(B)
