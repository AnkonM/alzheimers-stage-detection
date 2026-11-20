"""domain_shift on SYNTHETIC volumes with known answers; aggregate-only output."""
import json

import nibabel as nib
import numpy as np
import pytest

from src.validation import domain_shift as D


def nii(path, arr, zooms):
    img = nib.Nifti1Image(arr.astype(np.float32), np.diag([*zooms, 1.0]))
    img.header.set_zooms(zooms)
    nib.save(img, str(path))
    return path


def test_vol96_features_known_values():
    a = np.zeros((10, 10, 10))
    a[:5] = 0.5                                     # half the voxels at 0.5
    f = D.vol96_features(a)
    assert f["nonzero_frac"] == pytest.approx(0.5) and f["frac_zero"] == pytest.approx(0.5)
    assert f["nz_mean"] == pytest.approx(0.5) and f["nz_std"] == pytest.approx(0.0) and f["p50"] == pytest.approx(0.5)


def test_mask_volume_uses_header_zooms():
    m = np.zeros((20, 20, 20))
    m[:10, :10, :10] = 1                            # 1000 voxels
    assert D.mask_volume_mm3(m, (1.0, 1.0, 1.2)) == pytest.approx(1200.0)
    assert D.mask_volume_mm3(m, (2.0, 2.0, 2.0)) == pytest.approx(8000.0)


def test_case_features_from_files(tmp_path):
    v = np.zeros((8, 8, 8)); v[:4] = 1.0
    m = np.zeros((10, 10, 10)); m[:5, :5, :5] = 0.3
    f = D.case_features(nii(tmp_path / "v.nii.gz", v, (2.0, 2.0, 2.0)), nii(tmp_path / "m.nii.gz", m, (1.0, 1.0, 1.2)),
                        corr=0.42)
    assert f["nonzero_frac"] == pytest.approx(0.5) and f["mask_volume_mm3"] == pytest.approx(125 * 1.2, rel=1e-5)
    assert f["corr_vs_mni"] == 0.42


@pytest.mark.parametrize("c,v", [(0.5, "PASS"), (0.30, "PASS"), (0.299, "WARN"), (0.15, "WARN"), (0.149, "FAIL"), (0.05, "FAIL")])
def test_verdict_matches_engine_rule(c, v):
    assert D.verdict(c) == v


def test_qc_counts_by_group():
    r = D.qc_counts([0.5, 0.2, 0.1, 0.6], ["a", "a", "b", "b"])
    assert r["all"] == {"PASS": 2, "WARN": 1, "FAIL": 1} and r["by_group"]["b"] == {"PASS": 1, "WARN": 0, "FAIL": 1}


def test_compare_identical_and_shifted():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    same = D.compare(x, x)
    assert same["smd"] == pytest.approx(0) and same["ks_stat"] == pytest.approx(0) and same["median_diff"] == 0
    shifted = D.compare(x + 1.0, x)
    assert shifted["smd"] == pytest.approx(1.0, abs=0.05) and shifted["ks_p"] < 1e-6 and shifted["median_diff"] == pytest.approx(1.0)


def test_report_is_aggregate_only_and_writable(tmp_path):
    rng = np.random.default_rng(1)
    feats = [{"nz_mean": float(rng.normal(0.6, 0.02)), "corr_vs_mni": float(rng.normal(0.5, 0.06)),
              "mask_volume_mm3": float(rng.normal(1.4e6, 3e4))} for _ in range(120)]
    ref = [{"nz_mean": float(rng.normal(0.59, 0.02)), "corr_vs_mni": float(rng.normal(0.51, 0.06)),
            "mask_volume_mm3": float(rng.normal(1.4e6, 3e4))} for _ in range(150)]
    acq = [{"TR": "9.7", "TE": "4.0", "flip": "10"}] * 120
    rep = D.domain_shift_report(D.collect(feats), D.collect(ref), acq, groups=["g"] * 120)
    assert rep["acquisition_new"]["TR"] == {"9.7": 120} and rep["qc_new"]["n"] == 120
    text = json.dumps(rep, default=float)
    back = json.loads(text)
    assert "shift" in back["features"]["nz_mean"]

    def longest_numeric_list(o):                                   # nothing per-subject may be in the aggregate
        if isinstance(o, list):
            nums = len(o) if all(isinstance(x, (int, float)) for x in o) else 0
            return max([nums] + [longest_numeric_list(x) for x in o])
        if isinstance(o, dict):
            return max([0] + [longest_numeric_list(v) for v in o.values()])
        return 0
    assert longest_numeric_list(back) < 20


def test_adni_reference_reads_train_split_only_and_skips_failed_scans(tmp_path):
    import pandas as pd
    (tmp_path / "vol96").mkdir(); (tmp_path / "mni").mkdir()
    rows = []
    for i, (split, ok) in enumerate([("train", True), ("train", False), ("val", True), ("test", True), ("train", True)]):
        v = np.zeros((6, 6, 6)); v[:i + 1] = 0.5
        nii(tmp_path / "vol96" / f"P{i}.nii.gz", v, (2.0, 2.0, 2.0))
        nii(tmp_path / "mni" / f"P{i}.nii.gz", np.ones((4, 4, 4)), (1.0, 1.0, 1.2))
        rows.append((f"P{i}", split, ok, 0.4 + i / 100))
    pd.DataFrame(rows, columns=["PTID", "split", "ok", "corr_vs_mni"])[["PTID", "split"]].to_csv(tmp_path / "idx.csv", index=False)
    pd.DataFrame(rows, columns=["PTID", "split", "ok", "corr_vs_mni"])[["PTID", "ok", "corr_vs_mni"]].to_csv(tmp_path / "rep.csv", index=False)
    cols = D.adni_reference(tmp_path, tmp_path / "idx.csv", tmp_path / "rep.csv", split="train")
    assert len(cols["nonzero_frac"]) == 2                      # P0 and P4; P1 failed, P2/P3 are val/test
    assert cols["mask_volume_mm3"][0] == pytest.approx(64 * 1.2, rel=1e-5)
