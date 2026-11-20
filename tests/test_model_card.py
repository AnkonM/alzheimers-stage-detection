import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
CARD = (ROOT / "docs" / "model_card.md").read_text()


def test_wording_rules():
    assert "calibrat" not in CARD.lower()
    assert "model probability" in CARD
    assert "not age/sex-adjusted" in CARD
    assert "ALZ may differ from the published run by up to about 5e-5 because of GPU batch effects; the predicted stage is unaffected." in CARD
    assert "fusion model run without scan inputs" in CARD


def test_reference_population_and_synthseg_rule_present():
    assert "569 clinical-train subjects" in CARD
    assert "ICC(2,1) of at least 0.50" in CARD and "low agreement with training volumes, no percentile" in CARD


def test_external_validation_section_is_generated_from_the_json_and_carries_ack_doi_and_limits():
    import json
    from src.validation.ack import ACK_TEXT, DOI
    sec = CARD.split("## External validation (OASIS-1, cross-sectional)", 1)[1]
    assert "pending" not in CARD.lower().split("## external validation", 1)[1] and "Not yet performed" not in CARD
    assert ACK_TEXT in sec and DOI in sec
    a = json.loads((ROOT / "docs" / "validation" / "oasis_analysis.json").read_text())
    h = a["H1"]
    assert f"Outcome class **{h['verdict']}**" in sec
    assert f"{h['auc']:.3f}" in sec and f"[{h['ci95'][0]:.3f}, {h['ci95'][1]:.3f}]" in sec
    for needle in ("third-party documented, not documented by OASIS", "SynthSeg volumes", "Independence of the brain-age model from OASIS is not established",
                   "One scan per person", "the clinical stream", "do not validate staging", "Dated correction (2026-11-16)",
                   "MRI specialist (MRI-only mode)", "no ADNI cutoff applies to OASIS"):
        assert needle in sec, needle
    assert "calibrat" not in CARD.lower()


def test_oasis_tables_file_is_current_and_carries_ack_doi():
    import subprocess
    import sys
    from src.validation.ack import ACK_TEXT, DOI
    path = ROOT / "docs" / "validation" / "oasis_results_tables.md"
    before = path.read_text()
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_oasis_tables.py")], check=True, cwd=ROOT, capture_output=True)
    assert path.read_text() == before and ACK_TEXT in before and DOI in before and "calibrat" not in before.lower()


def test_card_is_current_with_its_generator(tmp_path):
    import subprocess
    import sys
    before = CARD
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_model_card.py")], check=True, cwd=ROOT,
                   capture_output=True)
    assert (ROOT / "docs" / "model_card.md").read_text() == before


import pytest  # noqa: E402


@pytest.mark.parametrize("name", ["build_model_card.py", "oasis_render.py", "build_oasis_tables.py"])
def test_generator_prose_has_no_literal_metrics(name):
    src = (ROOT / "tools" / name).read_text()
    # strip f-string placeholders, then look for decimals or percentages in the remaining prose
    prose = re.sub(r"\{[^{}]*\}", "", src)
    prose = prose.replace("95% CI", "").replace("FreeSurfer 5.1", "")        # interval label and software version
    prose = prose.replace("CDR 0.5", "")                                        # a class name of the CDR scale, not a metric
    bad = re.findall(r"(?<![\w.])\d+\.\d+(?![\w])|\b\d+\s?%", prose)
    assert not bad, bad


def _ext():
    return CARD.split("## External validation (OASIS-1, cross-sectional)", 1)[1]


def test_five_wording_additions_are_in_the_card():
    sec = _ext()
    # 1: directly under the two-cohort table, before the absolute-levels sentence
    t1 = ("The two tasks are not identical: OASIS positives are CDR 1 or higher (clinical diagnosis of probable AD) against CDR 0 elders from a "
          "research-centre cohort with screened images, whereas the ADNI reference is CN vs AD under ADNI criteria, from cross-fitted predictions "
          "on the training split. The comparison shows that the ranking transfers; it is not a measure of relative performance.")
    tail = sec.split("Same task in the two cohorts (AUC):", 1)[1]
    table_rows, rest = tail.strip().split("\n\n", 1)
    assert table_rows.startswith("| Task |") and rest.startswith(t1)
    assert rest[len(t1):].lstrip().startswith("Absolute score levels differ from ADNI")
    # 2
    assert "- OASIS-1 is a single cohort from a single research centre and scanner; this is one external check, not a multicentre validation." in sec
    # 3: appended to the brain-age independence bullet
    assert ("Independence of the brain-age model from OASIS is not established: OASIS is not named among its training cohorts, but the published\n"
            "  training list is incomplete. This also applies to the specialist result (S5), which uses the brain-age gap as an input.") in sec
    # 4: appended to the left-right bullet
    assert "for a subset of cases only. S6f does not establish which convention is correct." in sec
    # 5: opening provenance sentence names all three source files
    head = sec.split("### What this does", 1)[0]
    assert ("All numbers below are generated from `docs/validation/oasis_analysis.json`,\n`docs/validation/oasis_domain_shift.json` and "
            "`analysis/external_validation/adni_reference.json`; full tables are in") in head
