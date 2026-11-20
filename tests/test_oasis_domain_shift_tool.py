"""tools/oasis_domain_shift.py on SYNTHETIC volumes and text: the acquisition section is populated from scan mpr-1 only,
no label field is parsed, and the aggregate writer refuses identifiers."""
import importlib.util
import json
import pathlib

import nibabel as nib
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("oasis_domain_shift_tool", ROOT / "tools" / "oasis_domain_shift.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
from src.validation import domain_shift as DS  # noqa: E402
from src.validation.aggregate_io import AggregateError, write_json  # noqa: E402
from src.validation.ack import ACK_TEXT, DOI, MARKER  # noqa: E402

TEXT = """SESSION ID:   S1
AGE:          71
M/F:          F
CDR:          0.5
MMSE:         28

mpr-1      MPRAGE
mpr-2      MPRAGE

SCAN NUMBER:  mpr-1
TYPE:         MPRAGE
Vox res (mm): 1.0 x 1.0 x 1.25
Rect. Fov:    256/256
Orientation:  Sag
TR (ms):      9.7
TE (ms):      4.0
TI (ms):      20.0
Flip:         10

SCAN NUMBER:  mpr-2
TYPE:         MPRAGE
Vox res (mm): 2.0 x 2.0 x 2.0
Rect. Fov:    128/128
Orientation:  Cor
TR (ms):      1.1
TE (ms):      2.2
TI (ms):      3.3
Flip:         99
"""


def test_parse_acquisition_takes_mpr1_only_and_no_label_fields():
    a = T.parse_acquisition(TEXT)
    assert a == {"Vox res (mm)": "1.0 x 1.0 x 1.25", "Rect. Fov": "256/256", "Orientation": "Sag", "TR (ms)": "9.7",
                 "TE (ms)": "4.0", "TI (ms)": "20.0", "Flip": "10"}
    assert not any(k in a for k in ("CDR", "MMSE", "AGE", "M/F", "TYPE", "SESSION ID"))


def test_scan_one_literal_does_not_match():
    # the first version matched the number "1", but the files say "mpr-1": the section was empty
    assert T.parse_acquisition(TEXT.replace("mpr-1", "1")) == {}


def _case(tmp_path, i, seed, corr):
    rnd = np.random.default_rng(seed)
    v = np.where(rnd.random((8, 8, 8)) > 0.7, rnd.random((8, 8, 8)), 0.0)
    m = np.where(rnd.random((6, 6, 6)) > 0.5, 1.0, 0.0)
    nib.save(nib.Nifti1Image(v.astype("float32"), np.eye(4)), tmp_path / f"v{i}.nii.gz")
    nib.save(nib.Nifti1Image(m.astype("float32"), np.diag([1.0, 1.0, 1.2, 1.0])), tmp_path / f"m{i}.nii.gz")
    return {"vol96": tmp_path / f"v{i}.nii.gz", "mni": tmp_path / f"m{i}.nii.gz", "corr": corr,
            "group": "CDR 0" if i % 2 == 0 else "CDR 1", "acquisition": T.parse_acquisition(TEXT)}


@pytest.fixture
def report(tmp_path):
    rows = [_case(tmp_path, i, i, c) for i, c in enumerate((0.55, 0.35, 0.2, 0.5))]
    ref = DS.collect([DS.case_features(r["vol96"], r["mni"], corr=r["corr"] + 0.05) for r in rows])
    return T.build_report(rows, ref)


def test_report_has_populated_acquisition_and_group_qc_counts(report):
    assert report["acquisition_new"]["TR (ms)"] == {"9.7": 4} and report["acquisition_new"]["Orientation"] == {"Sag": 4}
    assert set(report["acquisition_new"]) == set(T.ACQ_FIELDS)
    assert report["qc_new"]["n"] == 4 and report["qc_new"]["all"] == {"PASS": 3, "WARN": 1, "FAIL": 0}
    # corr 0.55, 0.35, 0.20, 0.50 by case index; even index is "CDR 0": 0.55 PASS and 0.20 WARN; odd is "CDR 1": 0.35, 0.50 PASS
    assert report["qc_new"]["by_group"] == {"CDR 0": {"PASS": 1, "WARN": 1, "FAIL": 0}, "CDR 1": {"PASS": 2, "WARN": 0, "FAIL": 0}}
    assert report["scope"]["n_new"] == 4 and "corr_vs_mni" in report["features"] and "mask_volume_mm3" in report["features"]


def test_report_is_json_aggregate_and_writable(report, tmp_path):
    dest = write_json(tmp_path / "out.json", report, case_ids=["X0001", "X0002"])
    d = json.loads(dest.read_text())
    assert d["acquisition_new"] and MARKER in d["_notice"] and ACK_TEXT in d["_notice"] and DOI in d["_notice"]


def test_writer_refuses_case_ids_and_session_ids(report, tmp_path):
    bad = dict(report, scope=dict(report["scope"], note="X0007 leaked"))
    with pytest.raises(AggregateError):
        write_json(tmp_path / "a.json", bad, case_ids=["X0007"])
    bad2 = dict(report, scope=dict(report["scope"], note="OAS" + "1_0001_MR1"))
    with pytest.raises(AggregateError):
        write_json(tmp_path / "b.json", bad2, case_ids=[])


def test_session_id_is_extracted_from_a_scan_path_and_none_is_an_error():
    sid = "OAS" + "1_0009_MR1"
    assert T.session_of(f"/data/raw/{sid}/{sid}_mpr-1_anon.hdr") == sid
    with pytest.raises(ValueError):
        T.session_of("/data/raw/nothing/here.hdr")
