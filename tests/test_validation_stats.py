"""Statistics and analysis skeleton on SYNTHETIC data with known answers."""
import itertools
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.validation import aggregate_io as AIO
from src.validation import analyze_oasis as AN
from src.validation import stats as S
from src.validation.ack import ACK_TEXT, DOI, MARKER

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import oasis_guard as G  # noqa: E402

SESSION = "OAS" + "1" + "_" + "0007" + "_" + "MR1"


def test_auc_exact_and_ties():
    assert S.auc([2, 3], [1, 2]) == pytest.approx(0.875)
    assert S.auc([1, 1], [1, 1]) == 0.5
    assert S.auc([5], [1]) == 1.0 and S.auc([1], [5]) == 0.0
    with pytest.raises(ValueError):
        S.auc([], [1])


def test_auc_matches_known_binormal_value():
    rng = np.random.default_rng(1)
    pos, neg = rng.normal(1, 1, 6000), rng.normal(0, 1, 6000)
    assert S.auc(pos, neg) == pytest.approx(norm.cdf(1 / np.sqrt(2)), abs=0.012)   # 0.760


def test_bootstrap_is_reproducible_with_the_fixed_seed_and_covers_truth():
    rng = np.random.default_rng(2)
    pos, neg = rng.normal(1, 1, 120), rng.normal(0, 1, 150)
    a, b = S.bootstrap_auc(pos, neg), S.bootstrap_auc(pos, neg)
    assert a == b and a["seed"] == 20261113 and a["n_boot"] == 2000
    assert a["ci95"][0] < a["auc"] < a["ci95"][1] and a["ci95"][0] < norm.cdf(1 / np.sqrt(2)) < a["ci95"][1]
    c = S.bootstrap_auc(pos, neg, seed=1)
    assert c["ci95"] != a["ci95"] and c["auc"] == a["auc"]


@pytest.mark.parametrize("auc_v,lo,verdict", [(0.80, 0.70, "PASS"), (0.75, 0.651, "PASS"), (0.75, 0.65, "PARTIAL"),
                                              (0.80, 0.60, "PARTIAL"), (0.70, 0.60, "PARTIAL"), (0.65, 0.5, "PARTIAL"),
                                              (0.649, 0.5, "FAIL"), (0.45, 0.3, "FAIL"), (0.0, 0.0, "FAIL")])
def test_h1_verdict_rule(auc_v, lo, verdict):
    assert S.h1_verdict(auc_v, lo) == verdict


def perm_p(groups, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    obs = S.jonckheere_terpstra(groups)["J"]
    allv, sizes = np.concatenate(groups), [len(g) for g in groups]
    cnt = 0
    for _ in range(n):
        p = rng.permutation(allv)
        gs = np.split(p, np.cumsum(sizes)[:-1])
        cnt += S.jonckheere_terpstra(gs)["J"] >= obs
    return (cnt + 1) / (n + 1)


def test_jonckheere_terpstra_known_trend_and_null_and_permutation_agreement():
    rng = np.random.default_rng(3)
    trend = [rng.normal(m, 1, 40) for m in (0, 0.5, 1.2)]
    assert S.jonckheere_terpstra(trend)["p"] < 1e-4
    assert S.jonckheere_terpstra(trend[::-1])["p"] > 0.99                       # reversed order: no increasing trend
    assert S.jonckheere_terpstra(trend[::-1], "decreasing")["p"] < 1e-4         # the alternative is honoured
    null = [rng.normal(0, 1, 40) for _ in range(3)]
    assert S.jonckheere_terpstra(null)["p"] > 0.05
    small = [np.array([1, 2, 2, 3.0]), np.array([2, 3, 3, 4.0, 5]), np.array([3, 4, 5, 5, 6.0, 7])]   # with ties
    assert S.jonckheere_terpstra(small)["p"] == pytest.approx(perm_p(small), abs=0.015)


def test_jt_statistic_by_hand():
    # J = sum over group pairs of #(later > earlier) + half ties: ([1,2],[2,3]) -> 1+0.5? pairs (1,2)=1 (1,3)=1 (2,2)=.5 (2,3)=1
    assert S.jonckheere_terpstra([np.array([1.0, 2]), np.array([2.0, 3])])["J"] == 3.5


def test_spearman_and_partial_spearman():
    x = np.arange(50.0)
    assert S.spearman(x, x ** 3)[0] == pytest.approx(1.0)
    assert S.spearman(x, -x)[0] == pytest.approx(-1.0)
    rng = np.random.default_rng(4)
    z = rng.normal(size=400)
    a, b = z + 0.3 * rng.normal(size=400), z + 0.3 * rng.normal(size=400)
    assert S.spearman(a, b)[0] > 0.8 and abs(S.partial_spearman(a, b, z)) < 0.15
    bs = S.bootstrap_spearman(a, b, n_boot=200)
    assert bs["ci95"][0] < bs["rho"] < bs["ci95"][1]


def test_stratified_auc():
    score = np.array([1, 2, 3, 4, 1, 2, 3, 4.0])
    pos = np.array([0, 0, 1, 1, 0, 1, 0, 1], bool)
    age = np.array([60, 61, 62, 63, 80, 81, 82, 83.0])
    out = S.stratified_auc(score, pos, age, [70])
    assert out[0]["auc"] == 1.0 and out[1]["n_pos"] == 2 and out[1]["auc"] == pytest.approx(0.75)
    edge = S.stratified_auc([1, 2, 3, 4, 5, 6.0], [1, 0, 1, 0, 1, 0], [71, 72, 80, 81, 60, 95], [71, 80])
    assert [o["n_pos"] + o["n_neg"] for o in edge] == [2, 2, 2]               # 60 and 71 | 72 and 80 | 81 and 95
    only_one_class = S.stratified_auc([1, 2], [True, True], [1, 2], [])
    assert only_one_class[0]["auc"] is None


# ---------------- cohort-level skeleton ----------------

def synth(seed=5, n=(100, 70, 30), means=(0.35, 0.45, 0.65), sd=0.17, n_fail=6):
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    for cdr, m, cnt in zip((0, 0.5, 1), means, n):
        for _ in range(cnt):
            rows.append({"case_id": f"X{k:04d}", "alz": float(np.clip(rng.normal(m, sd), 0, 1)),
                         "bag": float(rng.normal(6 * cdr, 5)), "qc": "PASS", "cdr": cdr,
                         "mmse": float(np.clip(30 - 8 * cdr + rng.normal(0, 2), 0, 30))})
            k += 1
    df = pd.DataFrame(rows)
    df.loc[df.index[-n_fail:], ["alz", "qc"]] = [np.nan, "FAIL"]
    return df


def split_tables(df):
    scores = df[["case_id", "alz", "bag", "qc"]]
    labels = df[["case_id", "cdr", "mmse"]]
    return scores, labels


def test_align_is_independent_of_file_order_and_rejects_mismatches():
    df = synth()
    scores, labels = split_tables(df)
    a = AN.align_cases(scores, labels)
    b = AN.align_cases(scores.sample(frac=1, random_state=1), labels.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError):
        AN.align_cases(scores, labels.iloc[:-1])
    with pytest.raises(ValueError):
        AN.align_cases(pd.concat([scores, scores.iloc[:1]]), labels)
    with pytest.raises(ValueError):
        AN.align_cases(scores.rename(columns={"case_id": "id"}), labels)


def test_unknown_cdr_value_raises_instead_of_guessing():
    df = synth()
    df.loc[0, "cdr"] = 0.25
    with pytest.raises(ValueError):
        AN.add_classes(AN.align_cases(*split_tables(df)), "cdr_3class")


# ---------------- aggregate-only writer ----------------

def test_writer_refuses_ids_case_ids_and_vectors(tmp_path):
    with pytest.raises(AIO.AggregateError):
        AIO.write_json(tmp_path / "a.json", {"note": f"subject {SESSION}"})
    with pytest.raises(AIO.AggregateError):
        AIO.write_json(tmp_path / "b.json", {"rows": ["X0001"]})
    with pytest.raises(AIO.AggregateError):
        AIO.write_json(tmp_path / "c.json", {"x": "abc123"}, case_ids=["abc123"])
    with pytest.raises(AIO.AggregateError):
        AIO.write_json(tmp_path / "d.json", {"per_subject": list(np.linspace(0, 1, 120))})
    with pytest.raises(AIO.AggregateError):
        AIO.write_json(tmp_path / "e.json", {SESSION: 1})                         # keys are checked too
    with pytest.raises(AIO.AggregateError):
        AIO.write_json(tmp_path / "f.json", {"arr": np.zeros(200)})
    assert not list(tmp_path.glob("*.json"))
    with pytest.raises(AIO.AggregateError) as e:
        AIO.write_json(tmp_path / "g.json", {"note": SESSION})
    assert SESSION not in str(e.value)


def test_writer_accepts_aggregates_and_stamps_acknowledgement(tmp_path):
    df = AN.add_classes(AN.align_cases(*split_tables(synth())), "cdr_3class")
    res = {"H1": AN.h1(df), "S1": AN.s1(df), "S2_alz": AN.s2_trend(df, "alz")}
    p = AIO.write_json(tmp_path / "ok.json", res)
    text = p.read_text()
    assert MARKER in text and ACK_TEXT in text and DOI in text and json.loads(text)["H1"]["verdict"]
    m = AIO.write_markdown(tmp_path / "ok.md", "AUC table\n")
    t = m.read_text()
    assert MARKER in t and ACK_TEXT in t and DOI in t
    # the guard's own R5 rule accepts both and the tree scan is clean
    assert G.check_content("ok.md", t.encode()) == [] and G.check_content("ok.json", text.encode()) == []


def test_inside_the_repo_only_validation_dirs_are_allowed():
    for bad in (AIO.REPO / "docs" / "x.json", AIO.REPO / "analysis" / "y.json", AIO.REPO / "var" / "z.json"):
        with pytest.raises(AIO.AggregateError):
            AIO._check_dest(bad)
    for ok in (AIO.REPO / "docs" / "validation" / "x.json", AIO.REPO / "docs" / "figures" / "validation" / "a.png"):
        assert AIO._check_dest(ok)


def test_acknowledgement_constants_match_the_guard():
    assert (ACK_TEXT, DOI, MARKER) == (G.ACK_TEXT, G.DOI, G.RESULTS_MARKER)
