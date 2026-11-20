import ast
import json
import pathlib
import re

import numpy as np
import pytest

from src.assessment import report_builder as RB
from src.assessment.mri import qc as mqc
from src.assessment.schema import ALZ_BATCH_CAVEAT

ROOT = pathlib.Path(__file__).resolve().parents[1]
FEATS = json.load(open(ROOT / "analysis" / "kan_fusion" / "fusion_manifest.json"))["clinical_features"]
MRI_FEATS = ["Ventricles_ICV", "Hippocampus_ICV", "WholeBrain_ICV", "Entorhinal_ICV", "Fusiform_ICV", "MidTemp_ICV",
             "ALZ", "BAG", "AGE", "SEX"]


def synthetic_input(mode="FULL", provider="adni_fs"):
    rng = np.random.default_rng(1)
    p = np.array([0.13214, 0.30011, 0.41234, 0.15541], np.float32)
    p = (p / p.sum()).astype(np.float32)
    core = {"p": p, "add": np.float32(0.173), "alz": np.float64(0.37123456789), "pred_age": np.float32(71.23456),
            "unreliable": False, "age_at_scan": 68.4, "clin_p": np.array([0.2, 0.3, 0.4, 0.1]),
            "mri_p": np.array([0.2, 0.2, 0.2, 0.4]), "cshap": rng.normal(size=(len(FEATS), 4)).astype(np.float32),
            "clinical_features": np.array(FEATS), "cogindex": 0.4, "corrected_gap": 3.14159,
            "mshap_phi": rng.normal(size=(10, 4)).astype(np.float32), "mshap_imputed": np.zeros(10, bool),
            "mri_features": np.array(MRI_FEATS),
            "kan": {"phi_group": rng.normal(size=(25, 4, 9)).astype(np.float32),
                    "p_mri_masked": np.array([0.4, 0.3, 0.2, 0.1], np.float32),
                    "groups": np.array(["clinical logits", "MRI logits + has_mri", "age", "CDRSB + CogIndex + MMSE"]),
                    "targets": np.array([f"final_{n}" for n in RB.NAMES] + [f"residual_{n}" for n in RB.NAMES] + ["ADD"])}}
    vols = {"Ventricles": 40000.0, "Hippocampus": 7000.0, "WholeBrain": 1000000.0, "Entorhinal": 3500.0,
            "Fusiform": 17000.0, "MidTemp": 19000.0, "ICV": 1500000.0}
    clin = {"AGE": 70.0, "PTGENDER": "Male", "CDRSB": 1.5, "MMSE": 27.0, "ADAS13": 15.0}
    return RB.BuildInput(
        report_id="t1", created_at="2026-01-01T00:00:00Z", subject_label="synthetic", requested_mode=mode, mode=mode,
        input_provenance="adni_demo", volume_provider=provider, clinical=clin, completeness=[], n_imputed=0,
        qc_gates=[mqc.registration_gate(0.5)], vols_mm3=vols, core=core, cnn=None, has_brain_image=True)


def test_numbers_are_engine_outputs_exactly():
    bi = synthetic_input()
    r = RB.build(bi)
    c = bi.core
    assert list(r.stage.model_probability.values()) == [float(x) for x in c["p"]]
    assert r.stage.severity.value == float(np.dot([float(x) for x in c["p"]], np.arange(4)))
    assert r.ad.model_probability == float(c["add"]) and r.ad.alz == float(c["alz"])
    assert r.brain_age.predicted == float(c["pred_age"]) and r.brain_age.corrected_gap == float(c["corrected_gap"])
    assert r.brain_age.raw_gap == float(c["pred_age"] - c["age_at_scan"])
    k = int(np.argmax(c["clin_p"]))
    top = max(range(len(FEATS)), key=lambda j: abs(c["cshap"][j, k]))
    assert r.drivers.clinical[0].feature == FEATS[top] and r.drivers.clinical[0].shap == float(c["cshap"][top, k])
    assert any(x.id == "alz_batch_effect" and x.text == ALZ_BATCH_CAVEAT for x in r.caveats)


def test_clinical_only_has_no_mri_or_severity():
    bi = synthetic_input("CLINICAL_ONLY")
    bi.core.update(alz=None, pred_age=None, mri_p=None)
    bi.requested_mode = "CLINICAL_ONLY"
    r = RB.build(bi)
    assert r.stage.severity is None and not r.volumes and r.cnn is None and not r.drivers.mri
    assert r.brain_age.status == "not_available" and r.meta.mode_label == "fusion model run without scan inputs"
    assert all("MRI" not in g.group for g in r.drivers.stream_level)


def test_synthseg_volumes_follow_display_rule():
    bi = synthetic_input(provider="synthseg")
    bi.volume_agreement = "ICC summary"
    from src.assessment.volumes import agreement
    bi.qc_gates = [agreement.qc_gate(), mqc.registration_gate(0.5)]
    r = RB.build(bi)
    by = {v.structure: v for v in r.volumes}
    assert by["Hippocampus"].percentile is not None and by["Entorhinal"].percentile is None
    assert not any(f.id == "mri_changed_stage" for f in r.flags)


def test_adni_volumes_have_percentiles_from_committed_grid():
    r = RB.build(synthetic_input())
    q = RB.refs()["cn_volume_quantiles"]["Hippocampus_ICV"]["q"]
    h = next(v for v in r.volumes if v.structure == "Hippocampus")
    assert h.percentile == float(np.interp(h.ratio_to_icv, q, np.arange(101)))


# --- M9: no literal metric numbers in the builder or the frontend ------------------------------------------------
DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+|\b\d+(\.\d+)?\s?%")
ALLOWED = {"1.0"}


def test_no_literal_metric_in_builder_strings():
    src = (ROOT / "src" / "assessment" / "report_builder.py").read_text()
    bad = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            bad += [m.group(0) for m in DECIMAL.finditer(node.value) if m.group(0) not in ALLOWED]
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            bad.append(str(node.value))
    assert not bad, f"literal numbers in report_builder.py: {bad}"


def test_no_literal_metric_in_frontend():
    bad = []
    for f in (ROOT / "frontend" / "src" / "report").glob("*.tsx"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            code = re.sub(r"(className|style|viewBox|width|height)=(\"[^\"]*\"|\{[^}]*\})", "", line)
            code = re.sub(r"\b(toFixed|toPrecision)\(\d\)|\[\d\]|\b0\.\d\b", "", code)
            text = re.sub(r"[\w\-./:]*\d[\w\-./:]*(?=[\"'`])", "", code)
            for m in re.finditer(r"(AUC|ICC|ECE|accuracy|sensitivity|specificity)\D{0,12}\d", text):
                bad.append((f.name, i, m.group(0)))
            if re.search(r">[^<{]*\b\d+\.\d+\b[^<{]*<", code):
                bad.append((f.name, i, "decimal in JSX text"))
    assert not bad, bad


def test_literal_metric_checker_catches_the_m9_bug(tmp_path):
    assert DECIMAL.search("AUC 0.87 on test")
    assert DECIMAL.search("accuracy 84%")


# --- reference tables: train-569 only, aggregates only -------------------------------------------------------------
def test_reference_tables_population_and_no_subject_rows():
    d = json.load(open(RB.REFS / "reference_ranges.json"))
    assert d["population"]["split"] == "train" and d["population"]["n"] == 569
    assert sum(d["population"]["n_by_stage"].values()) == 569
    blob = json.dumps(d)
    assert not re.search(r"\b\d{3}_S_\d{4}\b", blob) and '"RID"' not in blob and '"PTID"' not in blob

    def lists(o):
        if isinstance(o, list):
            yield o
            for x in o:
                yield from lists(x)
        elif isinstance(o, dict):
            for v in o.values():
                yield from lists(v)
    assert max(len(x) for x in lists(d)) <= 101               # quantile grids only, never per-subject rows
    tr = ROOT / "dataset" / "processed" / "train_rids.csv"
    if tr.exists():
        assert len(tr.read_text().strip().splitlines()) - 1 == 569


def test_build_refs_never_reads_val_or_test_tables():
    src = (ROOT / "analysis" / "inference_refs" / "build_refs.py").read_text()
    assert "val_rids" not in src and src.count("test_rids") == 1      # the single use is the disjointness assertion
