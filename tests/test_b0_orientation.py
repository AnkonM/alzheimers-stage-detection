"""Stage B0 helpers on synthetic data. The ANTs-dependent E1 test is skipped when ANTsPy is not installed (it
runs in the m1 environment). Identifiers are assembled at run time."""
import io
import json
import pathlib
import struct
import tarfile

import numpy as np
import pytest

from src.validation import aggregate_io
from src.validation import b0_orientation as B

ROOT = pathlib.Path(__file__).resolve().parents[1]
N = 24


def sid(n, mr=1):
    return "OAS" + "1_" + f"{n:04d}" + f"_MR{mr}"


def blob(shape, centre, rad, val=1.0):
    g = np.indices(shape)
    d2 = sum((g[i] - centre[i]) ** 2 for i in range(3))
    return val * (d2 <= rad ** 2)


@pytest.fixture
def asym():
    """Template with a clear left-right asymmetry along axis 2 (a large blob left of the plane, a small one right)."""
    shape = (N, N, N)
    mid = (N - 1) / 2
    t = blob(shape, (12, 12, 7), 5, 1.0) + blob(shape, (12, 12, 17), 3, 0.5) + 0.2 * blob(shape, (12, 12, 12), 9)
    mask = blob(shape, (mid, mid, mid), 10) > 0
    return t.astype(float), mask.astype(float)


def test_corr_in_mask_matches_pearson_and_handles_constants():
    rng = np.random.default_rng(0)
    a, t = rng.normal(size=(6, 6, 6)), rng.normal(size=(6, 6, 6))
    mask = (rng.random((6, 6, 6)) > 0.4).astype(float)
    m = mask > 0.5
    assert B.corr_in_mask(a, t, mask) == pytest.approx(np.corrcoef(a[m], t[m])[0, 1], abs=1e-12)
    assert np.isnan(B.corr_in_mask(np.ones((6, 6, 6)), t, mask))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reflect_is_a_flip_and_shifts_with_zero_fill(axis):
    v = np.arange(5 * 5 * 5, dtype=float).reshape(5, 5, 5) + 1
    assert np.array_equal(B.reflect(v, axis, 4), np.flip(v, axis))
    assert np.array_equal(B.reflect(B.reflect(v, axis, 4), axis, 4), v)
    r = B.reflect(v, axis, 5)                       # plane moved by one voxel: first slice has no source
    assert np.all(np.take(r, 0, axis=axis) == 0) and np.array_equal(np.take(r, 1, axis=axis), np.take(v, 4, axis=axis))


def test_lr_axis_follows_the_x_component_of_the_direction_columns():
    assert B.lr_axis(np.eye(3)) == 0
    assert B.lr_axis(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)) == 1
    assert B.lr_axis(-np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)) == 2


def test_template_symmetry_finds_the_plane_and_separates_symmetric_from_asymmetric(asym):
    t, mask = asym
    s, r = B.template_symmetry(t, mask, 2)
    assert r < B.SYMMETRIC_R                        # clearly asymmetric
    mid = (N - 1) / 2
    sym = blob((N,) * 3, (12, 12, 8), 4) + blob((N,) * 3, (12, 12, 2 * mid - 8), 4)    # mirror pair about the plane
    s2, r2 = B.template_symmetry(sym, mask, 2)
    assert r2 == pytest.approx(1.0, abs=1e-9) and s2 == N - 1
    shifted = np.roll(sym, 1, axis=2)               # symmetric about a plane one voxel off centre
    s3, r3 = B.template_symmetry(shifted, mask, 2)
    assert s3 == N + 1 and r3 > 0.99          # s = 2 x plane: one voxel of shift moves it by 2


def test_d_score_sign_follows_orientation_and_vanishes_on_a_symmetric_template(asym):
    t, mask = asym
    rng = np.random.default_rng(1)
    s = N - 1
    vol = t + 0.05 * rng.normal(size=t.shape)
    assert B.d_score(vol, t, mask, 2, s) > 0.05
    assert B.d_score(B.reflect(vol, 2, s), t, mask, 2, s) < -0.05
    mid = (N - 1) / 2
    sym = blob((N,) * 3, (12, 12, 8), 4) + blob((N,) * 3, (12, 12, 2 * mid - 8), 4)
    assert abs(B.d_score(vol, sym, mask, 2, s)) < 1e-12        # symmetric template and mask: always zero


def test_control_passes_only_when_volumes_favour_the_template_orientation(asym, tmp_path):
    t, mask = asym
    rng = np.random.default_rng(2)
    vols = {f"A{i:03d}": t + 0.1 * rng.normal(size=t.shape) for i in range(20)}
    ok = B.run_control(sorted(vols), t, mask, 2, N - 1, vols.__getitem__, tmp_path / "c1.jsonl")
    assert ok == {"n": 20, "n_favouring_correct_orientation": 20, "passes": True}
    mirrored = {k: B.reflect(v, 2, N - 1) for k, v in vols.items()}
    bad = B.run_control(sorted(vols), t, mask, 2, N - 1, mirrored.__getitem__, tmp_path / "c2.jsonl")
    assert bad["passes"] is False and bad["n_favouring_correct_orientation"] == 0
    mix = {k: (v if i < 17 else mirrored[k]) for i, (k, v) in enumerate(vols.items())}
    assert B.run_control(sorted(vols), t, mask, 2, N - 1, mix.__getitem__, tmp_path / "c3.jsonl")["passes"] is False   # 17 < 18


def _stub_run(asym, tmp_path, truth="radiological", n=20, calls=None, memory_ok=lambda: None, noise_seed=3):
    t, mask = asym
    rng = np.random.default_rng(noise_seed)
    base = {f"X{i:04d}": t + 0.1 * rng.normal(size=t.shape) for i in range(1, n + 1)}
    rows = [(c, c) for c in base]

    def convert(path, conv):
        return (path, conv)

    def preprocess(token):
        if calls is not None:
            calls.append(token)
        case, conv = token
        v = base[case] if conv == truth else B.reflect(base[case], 2, N - 1)
        return v, B.corr_in_mask(v, t, mask)

    out = tmp_path / "e3.jsonl"
    B.run_e3_oasis(rows, convert, preprocess, t, mask, 2, N - 1, out, memory_ok=memory_ok)
    return out


def test_e3_run_scores_wins_and_is_resumable(asym, tmp_path):
    calls = []
    out = _stub_run(asym, tmp_path, calls=calls)
    assert len(calls) == 40
    s = B.e3_summary(out)
    assert s["n_scans"] == 20 and s["wins"] == {"radiological": 20, "neurological": 0, "tie": 0}
    assert s["supported"] == ["radiological"] and s["in_plane_sanity_ok"] is True
    t, mask = asym
    B.run_e3_oasis([(f"X{i:04d}", 0) for i in range(1, 21)], lambda p, c: (p, c), lambda tok: calls.append(tok), t, mask, 2, N - 1, out)
    assert len(calls) == 40                                      # nothing re-run
    other = tmp_path / "other"
    other.mkdir()
    assert B.e3_summary(_stub_run(asym, other, truth="neurological"))["supported"] == ["neurological"]


def test_e3_support_limit_and_in_plane_sanity(tmp_path):
    def write(wins_rad, corr=0.5):
        p = tmp_path / f"s{wins_rad}_{corr}.jsonl"
        with open(p, "w") as f:
            for i in range(20):
                win = i < wins_rad
                for conv in B.CONVENTIONS:
                    d = (0.1 if (conv == "radiological") == win else -0.1)
                    f.write(json.dumps({"case": f"X{i:04d}", "convention": conv, "corr": corr, "corr_reflected": corr - d, "D": d}) + "\n")
        return p
    assert B.e3_summary(write(15))["supported"] == ["radiological"]
    assert B.e3_summary(write(14))["supported"] == []
    assert B.e3_summary(write(5))["supported"] == ["neurological"]
    low = B.e3_summary(write(20, corr=0.2))
    assert low["in_plane_sanity_ok"] is False and low["median_of_larger_corr"] == 0.2


def test_e3_refuses_to_start_without_memory_and_flags_a_correlation_mismatch(asym, tmp_path):
    t, mask = asym

    def low():
        raise B.B0Error("memory")
    with pytest.raises(B.B0Error):
        _stub_run(asym, tmp_path, memory_ok=low)
    assert not (tmp_path / "e3.jsonl").exists()
    with pytest.raises(B.B0Error):
        B.run_e3_oasis([("X0001", 0)], lambda p, c: 0, lambda tok: (t, 0.123), t, mask, 2, N - 1, tmp_path / "m.jsonl")
    assert B.require_memory(6.0, probe=lambda: 7.0) is None
    with pytest.raises(B.B0Error):
        B.require_memory(6.0, probe=lambda: 5.9)


@pytest.mark.parametrize("kw,outcome,conv", [
    (dict(e1_consistent=False, in_plane_ok=True, e3_informative=True, e3_supported="radiological", egeo_convention=None, egeo_documented_basis=False), "stop", None),
    (dict(e1_consistent=True, in_plane_ok=False, e3_informative=True, e3_supported="radiological", egeo_convention=None, egeo_documented_basis=False), "stop", None),
    (dict(e1_consistent=True, in_plane_ok=True, e3_informative=False, e3_supported=None, egeo_convention=None, egeo_documented_basis=False), "stop", None),
    (dict(e1_consistent=True, in_plane_ok=True, e3_informative=False, e3_supported=None, egeo_convention="radiological", egeo_documented_basis=False), "stop", None),   # no documented basis
    (dict(e1_consistent=True, in_plane_ok=True, e3_informative=True, e3_supported="radiological", egeo_convention="neurological", egeo_documented_basis=True), "stop", None),
    (dict(e1_consistent=True, in_plane_ok=True, e3_informative=True, e3_supported=None, egeo_convention=None, egeo_documented_basis=False), "stop", None),
    (dict(e1_consistent=True, in_plane_ok=True, e3_informative=True, e3_supported="neurological", egeo_convention=None, egeo_documented_basis=False), "propose", "neurological"),
    (dict(e1_consistent=True, in_plane_ok=True, e3_informative=True, e3_supported="radiological", egeo_convention="radiological", egeo_documented_basis=True), "propose", "radiological"),
    (dict(e1_consistent=True, in_plane_ok=True, e3_informative=False, e3_supported=None, egeo_convention="neurological", egeo_documented_basis=True), "propose", "neurological"),
])
def test_decision_rule(kw, outcome, conv):
    o, c, why = B.decide(**kw)
    assert (o, c) == (outcome, conv) and why


T4 = """t4 file written by a registration tool
scale:   1.000000
   0.001000   0.002000   0.998000   -3.5
   1.001000   0.000000  -0.003000   10.25
   0.000000   0.999000   0.004000   -7.0
   0.000000   0.000000   0.000000    1.0
"""


def test_parse_t4_summary_and_errors():
    M, notes = B.parse_t4(T4)
    assert M.shape == (4, 4) and notes == ["t4 file written by a registration tool", "scale:   1.000000"]
    s = B.t4_summary(M)
    assert s["pattern"] == "atlas_x<-+raw_z,atlas_y<-+raw_x,atlas_z<-+raw_y" and s["det_sign"] == "+" and s["last_row_is_0001"]
    flipped = M.copy()
    flipped[0, :3] *= -1
    f = B.t4_summary(flipped)
    assert f["det_sign"] == "-" and f["pattern"].startswith("atlas_x<--raw_z")
    with pytest.raises(B.B0Error):
        B.parse_t4("a\n1 2 3 4\n")


def hdr_bytes(orient=0, dim=(176, 208, 176)):
    h = bytearray(348)
    h[0:4] = struct.pack(">i", 348)
    h[40:56] = struct.pack(">8h", 3, *dim, 1, 1, 1, 1)
    h[70:74] = struct.pack(">2h", 4, 16)
    h[76:108] = struct.pack(">8f", 0, 1.0, 1.0, 1.0, 0, 0, 0, 0)
    h[252] = orient
    return bytes(h)


def add(tar, name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    tar.addfile(ti, io.BytesIO(data))


def test_extract_geo_and_egeo_report_use_only_text_and_headers(tmp_path):
    arch = tmp_path / "oasis_cross-sectional_disc1.tar.gz"
    with tarfile.open(arch, "w:gz") as t:
        for n in (1, 2, 3):
            s = sid(n)
            b = f"disc1/{s}"
            add(t, f"{b}/{s}.txt", f"SESSION ID:   {s}\nHAND:         Right\nOrientation:  Sag\n".encode())
            add(t, f"{b}/{s}.xml", b"<a><orientation>Sag</orientation></a>")
            add(t, f"{b}/PROCESSED/MPRAGE/T88_111/t4_files/{s}_mpr-1_anon_to_711-2C_t4", T4.encode())
            add(t, f"{b}/PROCESSED/MPRAGE/T88_111/t4_files/{s}_mpr-2_anon_to_711-2C_t4", b"other scan")
            add(t, f"{b}/PROCESSED/MPRAGE/T88_111/{s}_mpr_n4_anon_111_t88_gfc.hdr", hdr_bytes())
            add(t, f"{b}/PROCESSED/MPRAGE/T88_111/{s}_mpr_n4_anon_111_t88_gfc.img", b"PAYLOAD")
            add(t, f"{b}/PROCESSED/MPRAGE/SUBJ_111/{s}_mpr_n4_anon_sbj_111.img", b"PAYLOAD")
    data = tmp_path / "data"
    (data / "meta").mkdir(parents=True)
    for n in (1, 2, 3):
        (data / "meta" / f"{sid(n)}.txt").write_text(f"SESSION ID: {sid(n)}\nHAND:         Right\nOrientation:  Sag\n")
        (data / "meta" / f"{sid(n)}.xml").write_text("<a><orientation>Sag</orientation></a>")
    n = B.extract_geo([arch], [sid(1), sid(3)], data)
    assert n == 4
    got = sorted(str(p.relative_to(data / "b0" / "egeo")) for p in (data / "b0" / "egeo").rglob("*") if p.is_file())
    assert got == sorted([f"{sid(1)}/atlas_gfc.hdr", f"{sid(1)}/raw_to_711-2C.t4", f"{sid(3)}/atlas_gfc.hdr", f"{sid(3)}/raw_to_711-2C.t4"])
    assert b"PAYLOAD" not in b"".join(p.read_bytes() for p in (data / "b0" / "egeo").rglob("*") if p.is_file())
    res = B.egeo_report([sid(1), sid(3)], data)
    assert res["determinant_sign_counts"] == {"+": 2} and res["signed_permutation_patterns"] == {"atlas_x<-+raw_z,atlas_y<-+raw_x,atlas_z<-+raw_y": 2}
    assert res["atlas_header"] == {"orient code 0, dims 176x208x176, voxel 1x1x1": 2}
    assert res["last_row_0001"] == 2 and any("orientation" in k.lower() for k in res["orientation_like_lines_in_xml_txt_masked"])
    aggregate_io.check_aggregate(res)
    assert not aggregate_io.ID_RE.search(json.dumps(res))


def test_orientation_lines_mask_digits():
    c = B.orientation_lines("Orientation:  Sag\nHAND:         Right\nAGE:   74\nflip 10 and 20.5\n")
    assert c["Orientation:  Sag"] == 1 and c["HAND:         Right"] == 1 and "flip # and #" in c and not any("74" in k for k in c)


def test_adni_control_ids_are_train_only_sorted_and_need_a_stored_volume(tmp_path):
    import pandas as pd
    (tmp_path / "analysis/fusion_feasibility").mkdir(parents=True)
    (tmp_path / "dataset/mri_prep/mni").mkdir(parents=True)
    rows = [(i, f"002_S_{i:04d}", "train" if i % 5 else "val") for i in range(100, 100 + 60)]
    pd.DataFrame({"RID": [r[0] for r in rows], "clin_split": [r[2] for r in rows]}).to_csv(tmp_path / "analysis/fusion_feasibility/alz_crossfit.csv", index=False)
    pd.DataFrame({"RID": [r[0] for r in rows], "PTID": [r[1] for r in rows], "ok": True}).to_csv(tmp_path / "dataset/mri_prep/m1_report_full.csv", index=False)
    for i, p, split in rows:
        if i != 101:                                  # a train subject without a stored volume
            (tmp_path / "dataset/mri_prep/mni" / f"{p}.nii.gz").write_bytes(b"x")
    ids = B.adni_control_ids(20, tmp_path)
    assert len(ids) == 20 and ids == sorted(ids) and "###_S_####" not in ids and "###_S_####" not in ids   # 100 % 5 == 0 is val
    assert ids[0] == "###_S_####"
    with pytest.raises(B.B0Error):
        B.adni_control_ids(500, tmp_path)


def test_sample_cases_are_the_first_sorted_case_ids(tmp_path):
    d = tmp_path
    (d / "private").mkdir()
    (d / "manifest").mkdir()
    (d / "private" / "mapping.csv").write_text("case_id,session_id,subject_id,scan_rank\n" + "".join(f"X{i:04d},{sid(i)},{i:04d},1\n" for i in (3, 1, 2, 4)))
    (d / "manifest" / "inputs.csv").write_text("case_id,scan_path\n" + "".join(f"X{i:04d},/p/{i}\n" for i in (3, 1, 2, 4)))
    assert B.sample_cases(d, 3) == [("X0001", sid(1), "/p/1"), ("X0002", sid(2), "/p/2"), ("X0003", sid(3), "/p/3")]


def test_report_command_writes_an_aggregate_with_the_decision(tmp_path):
    data = tmp_path / "data"
    b0 = data / "b0"
    b0.mkdir(parents=True)
    e1 = {"consistent": True, "checks": {"x": True}, "readers": {"r": {"marker_side": "L"}}, "after_m1_orientation_step": {}}
    eg = {"n_sessions": 20, "determinant_sign_counts": {"+": 20}}
    tc = {"lr_axis": 2, "plane_s": 95, "plane_s_centre": 95, "r_T": 0.999, "symmetric": True,
          "control": {"n": 20, "n_favouring_correct_orientation": 11, "passes": False}, "e3_informative": False, "template_files": ["x"]}
    for name, o in (("e1.json", e1), ("egeo.json", eg), ("template_control.json", tc)):
        (b0 / name).write_text(json.dumps(o))
    out = tmp_path / "b0_summary.json"
    assert B.main(["report", "--data", str(data), "--out-json", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["decision"]["outcome"] == "stop" and res["decision"]["proposed_convention"] is None
    assert res["e3"]["oasis_scans"] == "not run" and "template_files" not in res["e3"]["template_and_control"]
    assert B.main(["report", "--data", str(data), "--out-json", str(out), "--egeo-convention", "radiological",
                   "--egeo-documented-basis", "yes"]) == 0
    assert json.loads(out.read_text())["decision"] == {"outcome": "propose", "proposed_convention": "radiological",
                                                       "reason": "E-geo gives it with a documented basis; E3 did not corroborate it (uninformative)"}


def test_e1_on_synthetic_files_in_the_m1_environment(tmp_path):
    pytest.importorskip("ants")
    pytest.importorskip("nibabel")
    r = B.e1_report(tmp_path / "e1")
    assert r["consistent"], r["checks"]
    assert r["readers"]["loader NIfTI (radiological), ITK"]["marker_side"] != r["readers"]["loader NIfTI (neurological), ITK"]["marker_side"]


def test_source_carries_no_identifier_literal():
    assert not aggregate_io.ID_RE.search((ROOT / "src" / "validation" / "b0_orientation.py").read_text())
    assert not aggregate_io.ID_RE.search(pathlib.Path(__file__).read_text())


def _res_for_render(symmetric=False, passes=False, n_fav=4, oasis="not run"):
    return {"e1": {"checks": {"a check": True}, "consistent": True,
                   "readers": {"nibabel, .hdr read directly": {"axes_increase_toward": ["L", "A", "S"], "marker_side": "R"},
                               "ITK failed": {"status": "reader failed", "error": "X"}},
                   "after_m1_orientation_step": {c: {"array_axis_left_right": 2, "marker_array_position": [1, 2, 28 if c == "radiological" else 104],
                                                     "array_shape": [256, 256, 133], "marker_side": "L" if c == "radiological" else "R"} for c in B.CONVENTIONS}},
            "e_geo": {"n_sessions": 20, "determinant_sign_counts": {"+": 20}, "abs_determinant_range": [0.75, 1.06],
                      "signed_permutation_patterns": {"atlas_x<--raw_y": 20}, "atlas_header": {"orient code 0": 20},
                      "transform_file_non_numeric_lines_masked": {"scale: #": 20}, "orientation_like_lines_in_xml_txt_masked": {"Orientation:  Sag": 77},
                      "convention_read_from_it": "none", "documented_basis": "no"},
            "e3": {"template_and_control": {"r_T": 0.9827, "plane_s": 95, "plane_s_centre": 95, "symmetric": symmetric,
                                            "control": {"n": 20, "n_favouring_correct_orientation": n_fav, "passes": passes}, "e3_informative": passes and not symmetric},
                   "oasis_scans": oasis},
            "decision": {"outcome": "stop", "proposed_convention": None, "reason": "E3 is uninformative and E-geo gives no absolute convention"}}


def test_part2_renderer_states_the_stop_and_is_aggregate_only(tmp_path):
    md = B.render_part2(_res_for_render(), "abc1234")
    assert md.startswith(B.A1_BEGIN) and md.endswith(B.A1_END) and "commit `abc1234`" in md
    assert "Outcome: STOP. Proposed convention: none." in md and "E-geo gives no absolute convention" in md
    assert "below chance" in md and "not run on OASIS data" in md and "poor discriminator" in md
    aggregate_io.check_aggregate(md)
    assert not aggregate_io.ID_RE.search(md)
    f = tmp_path / "a1.md"
    f.write_text(f"top\n{B.A1_BEGIN}\nold\n{B.A1_END}\nbottom\n")
    assert B.update_addendum(md, f) is True and B.update_addendum(md, f) is False
    t = f.read_text()
    assert t.startswith("top\n") and t.endswith("bottom\n") and "old" not in t
    sym = B.render_part2(_res_for_render(symmetric=True, n_fav=10), "abc1234")
    assert "poor discriminator" not in sym and "below chance" not in sym


def test_scrub_notes_removes_names_and_home_paths_and_merges_counts():
    c = B.scrub_notes({"$Id: imgreg_#dfp.c,v # #/#/# #:#:# avi Exp rsachs $": 18, "$Id: imgreg_#dfp.c,v # #/#/# #:#:# rsachs Exp $": 2,
                       "/home/iac#/nil-tools/imgreg_#dfp /home/iac#/nil-tools/refdir/young#+#_atrophy_avg_#": 20, "scale: #": 20})
    assert "rsachs" not in str(c) and "avi" not in str(c).replace("avg", "") and "/home/" not in str(c)
    assert c["$Id: <revision line> $"] == 20 and c["scale: #"] == 20
    assert c["<path>/nil-tools/imgreg_#dfp <path>/nil-tools/refdir/young#+#_atrophy_avg_#"] == 20
