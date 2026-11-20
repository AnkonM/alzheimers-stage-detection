"""tools/make_s6f_manifest.py on synthetic data only: the S6f rule (every CDR >= 1 case plus the first 30 CDR 0 cases by
sorted case id), only the orientation column changes, no label content reaches inputs, output is counts and a hash.
Set S6F_TOOL to another file to run these tests against a mutated copy of the tool (mutation checks)."""
import csv
import hashlib
import importlib.util
import os
import pathlib
import stat

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = pathlib.Path(os.environ.get("S6F_TOOL", ROOT / "tools" / "make_s6f_manifest.py"))
IN_COLS = ["case_id", "scan_path", "scan_format", "orientation", "age", "sex", "scan_rank", "volume_source"]
LAB_COLS = ["case_id", "CDR", "MMSE", "eTIV", "nWBV"]
SENTINEL = "7.7777"          # a label value that must never appear in the output


def load_tool():
    spec = importlib.util.spec_from_file_location("s6f_tool_under_test", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def tool():
    return load_tool()


def make_cohort(tmp_path, n0=45, n05=25, n1=27, n2=3, shuffle=True):
    """Synthetic cohort. Ids are X0001...; classes are scattered so file order and sorted order differ."""
    import random
    rnd = random.Random(5)
    n = n0 + n05 + n1 + n2
    cdrs = ["0"] * n0 + ["0.5"] * n05 + ["1"] * n1 + ["2"] * n2
    rnd.shuffle(cdrs)
    ids = [f"X{i:04d}" for i in range(1, n + 1)]
    truth = dict(zip(ids, cdrs))
    order = list(ids)
    if shuffle:
        rnd.shuffle(order)
    ip, lp = tmp_path / "inputs.csv", tmp_path / "labels.csv"
    with open(ip, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(IN_COLS)
        for c in order:
            w.writerow([c, f"/data/{c}.hdr", "analyze75", "oasis1_primary", 70 + int(c[1:]) % 20, "F" if int(c[1:]) % 2 else "M", 1, "synthseg"])
    with open(lp, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(LAB_COLS)
        for c in reversed(order):
            w.writerow([c, truth[c], "not-a-number", "garbage", SENTINEL])
    return ip, lp, truth


def expected_ids(truth):
    pos = [c for c, v in truth.items() if float(v) >= 1]
    zero = sorted(c for c, v in truth.items() if float(v) == 0)[:30]
    return sorted(pos + zero)


def read_out(out_dir):
    with open(pathlib.Path(out_dir) / "inputs.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_selection_is_all_positive_plus_first_30_reference_in_sorted_id_order(tool, tmp_path):
    ip, lp, truth = make_cohort(tmp_path)
    res = tool.build(ip, lp, tmp_path / "out", expect_positive=30)
    rows = read_out(tmp_path / "out")
    got = [r["case_id"] for r in rows]
    assert got == expected_ids(truth)                       # same set AND sorted order
    assert (res["n_inputs"], res["n_positive"], res["n_reference"], res["n_rows"]) == (100, 30, 30, 60)
    assert sum(float(truth[c]) >= 1 for c in got) == 30 and sum(float(truth[c]) == 0 for c in got) == 30
    assert not any(float(truth[c]) == 0.5 for c in got)     # CDR 0.5 is never selected
    assert any(truth[c] == "2" for c in got)                # CDR 2 is selected (CDR >= 1)


def test_the_first_30_reference_cases_are_by_sorted_id_not_by_file_order(tool, tmp_path):
    ip, lp, truth = make_cohort(tmp_path, shuffle=True)
    file_order_zero = []
    with open(ip, newline="") as f:
        for r in csv.DictReader(f):
            if float(truth[r["case_id"]]) == 0:
                file_order_zero.append(r["case_id"])
    assert file_order_zero[:30] != sorted(file_order_zero)[:30]     # the fixture really distinguishes the two rules
    tool.build(ip, lp, tmp_path / "out")
    zero_in_out = [r["case_id"] for r in read_out(tmp_path / "out") if float(truth[r["case_id"]]) == 0]
    assert zero_in_out == sorted(file_order_zero)[:30]


def test_only_the_orientation_column_changes(tool, tmp_path):
    ip, lp, truth = make_cohort(tmp_path)
    tool.build(ip, lp, tmp_path / "out")
    with open(ip, newline="") as f:
        src = {r["case_id"]: r for r in csv.DictReader(f)}
    rows = read_out(tmp_path / "out")
    assert list(rows[0].keys()) == IN_COLS
    for r in rows:
        assert r["orientation"] == "oasis1_mirrored"
        assert {k: v for k, v in r.items() if k != "orientation"} == {k: v for k, v in src[r["case_id"]].items() if k != "orientation"}


def test_no_label_content_reaches_the_output_and_labels_are_not_copied(tool, tmp_path):
    ip, lp, truth = make_cohort(tmp_path)
    tool.build(ip, lp, tmp_path / "out")
    text = (tmp_path / "out" / "inputs.csv").read_text()
    for needle in (SENTINEL, "garbage", "not-a-number", "CDR", "MMSE", "eTIV", "nWBV"):
        assert needle not in text
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["inputs.csv"]


def test_labels_other_than_case_id_and_cdr_are_never_parsed(tool, tmp_path):
    ip, lp, truth = make_cohort(tmp_path)          # MMSE/eTIV/nWBV columns hold garbage that would break any parsing
    assert tool.build(ip, lp, tmp_path / "out")["n_rows"] == 60


def test_permissions_dir_700_file_600(tool, tmp_path):
    ip, lp, _ = make_cohort(tmp_path)
    tool.build(ip, lp, tmp_path / "out")
    assert stat.S_IMODE((tmp_path / "out").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "out" / "inputs.csv").stat().st_mode) == 0o600


def test_refuses_wrong_group_sizes_and_inconsistent_inputs(tool, tmp_path):
    ip, lp, truth = make_cohort(tmp_path)
    with pytest.raises(tool.S6fError):
        tool.build(ip, lp, tmp_path / "o1", expect_positive=29)          # wrong positive count
    (tmp_path / "x").mkdir()
    ip2, lp2, _ = make_cohort(tmp_path / "x", n0=29)
    with pytest.raises(tool.S6fError):
        tool.build(ip2, lp2, tmp_path / "o2")                            # fewer than 30 CDR 0 cases
    # id sets differ
    lines = lp.read_text().splitlines()
    lp.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(tool.S6fError):
        tool.build(ip, lp, tmp_path / "o3")


def test_refuses_duplicates_nonempty_output_and_unroundtrippable_rows(tool, tmp_path):
    ip, lp, _ = make_cohort(tmp_path)
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "keep").write_text("x")
    with pytest.raises(tool.S6fError):
        tool.build(ip, lp, tmp_path / "full")
    # a needlessly quoted field would be rewritten differently: refuse rather than alter a column silently
    t = ip.read_text().replace("analyze75", '"analyze75"', 1)
    ip.write_text(t)
    with pytest.raises(tool.S6fError):
        tool.build(ip, lp, tmp_path / "o")


def test_cli_prints_counts_and_the_hash_only(tool, tmp_path, capsys):
    ip, lp, truth = make_cohort(tmp_path)
    assert tool.main(["--inputs", str(ip), "--labels", str(lp), "--out-dir", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    digest = hashlib.sha256((tmp_path / "out" / "inputs.csv").read_bytes()).hexdigest()
    assert f"s6f inputs.csv sha256 {digest}" in out
    assert "selected CDR>=1: 30; CDR 0: 30; rows written 60" in out
    import re
    assert not re.search(r"X\d{4}", out) and "/data/" not in out and SENTINEL not in out
    assert len(out.strip().splitlines()) == 2


def test_cli_refusal_prints_no_ids(tool, tmp_path, capsys):
    ip, lp, _ = make_cohort(tmp_path, n0=29)
    assert tool.main(["--inputs", str(ip), "--labels", str(lp), "--out-dir", str(tmp_path / "out")]) == 2
    cap = capsys.readouterr()
    import re
    assert not re.search(r"X\d{4}", cap.out + cap.err)


def test_group_size_is_the_protocol_constant_defined_once(tool):
    from src.validation import stats
    assert tool.N_REFERENCE == stats.S6F_N_CDR0 == 30
