"""plumbing_view exposes statuses only; score_view exposes the model outputs; neither reads the mri_target field."""
import json
import pathlib

from src.assessment import report_reader as RR

FIX = pathlib.Path(__file__).parent / "fixtures" / "reports"
SCORE_KEYS = {"alz", "p_ad", "bag", "model_probability", "corrected_gap", "corr_vs_mni"}


def test_plumbing_view_has_no_scores():
    v = RR.plumbing_view(FIX / "mri_only.json")
    assert v["schema_valid"] is True and v["mode"] == "MRI_ONLY" and v["has_stage"] is False and v["has_alz"] is True
    assert not (SCORE_KEYS & set(v))
    assert all(isinstance(x, (str, bool, type(None), float, dict)) for x in v.values())
    flat = json.dumps(v)
    d = json.loads((FIX / "mri_only.json").read_text())
    assert str(d["ad"]["alz"]) not in flat and str(d["brain_age"]["corrected_gap"]) not in flat


def test_score_view_reads_the_protocol_quantities():
    d = json.loads((FIX / "mri_only.json").read_text())
    s = RR.score_view(FIX / "mri_only.json")
    assert s["alz"] == d["ad"]["alz"] and s["p_ad"] == d["ad"]["model_probability"]
    assert s["p_ad_source"] == d["ad"]["source"] and s["bag"] == d["brain_age"]["corrected_gap"]
    reg = [g for g in d["qc"]["gates"] if g["name"] == RR.REGISTRATION_GATE][0]
    assert s["corr_vs_mni"] == reg["value"] and s["qc"] == reg["status"]


def test_neither_view_touches_mri_target():
    import ast
    tree = ast.parse(pathlib.Path(RR.__file__).read_text())
    used = set()
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        doc = ast.get_docstring(fn, clean=False)
        for node in ast.walk(fn):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value != doc:
                used.add(node.value)
    assert "drivers" not in used and "mri_target" not in used and "stage" in used     # stage only as a presence flag


def test_invalid_report_is_flagged_not_crashed(tmp_path):
    d = json.loads((FIX / "mri_only.json").read_text())
    d["stage"] = json.loads((FIX / "ad.json").read_text())["stage"]            # MRI_ONLY must not have a stage
    p = tmp_path / "r.json"
    p.write_text(json.dumps(d))
    v = RR.plumbing_view(p)
    assert v["schema_valid"] is False and v["has_stage"] is True
