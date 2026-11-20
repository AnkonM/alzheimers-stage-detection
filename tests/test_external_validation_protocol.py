"""The protocol draft has a generated reference block that matches the committed aggregate JSON (numbers are
generated, never typed), and the reference run never reads the frozen test split."""
import json
import pytest
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
from analysis.external_validation import adni_reference as R  # noqa: E402

PROTOCOL = ROOT / "docs" / "external_validation_protocol.md"


def test_reference_block_matches_json():
    res = json.loads((ROOT / "analysis" / "external_validation" / "adni_reference.json").read_text())
    t = PROTOCOL.read_text()
    block = t[t.index(R.BEGIN): t.index(R.END) + len(R.END)]
    assert block == R.render_md(res)


def test_reference_run_never_reads_test_split():
    src = (ROOT / "analysis" / "external_validation" / "adni_reference.py").read_text()
    for forbidden in ("test_streams", "test_kan", "scan_inputs_test", "brainage_test", "frozen_run"):
        assert forbidden not in src.replace("test_split_read", "")
    res = json.loads((ROOT / "analysis" / "external_validation" / "adni_reference.json").read_text())
    assert res["test_split_read"] is False


def test_stage_a_block_matches_json_and_is_aggregate_only():
    from src.validation import aggregate_io
    from src.validation import oasis_manifest as M
    path = ROOT / "docs" / "validation" / "stage_a_counts.json"
    res = json.loads(path.read_text())
    aggregate_io.check_aggregate(res)
    assert aggregate_io.ID_RE.search(path.read_text()) is None
    res.pop("_notice")
    t = PROTOCOL.read_text()
    block = t[t.index(M.BEGIN.split(" (")[0]): t.index(M.END) + len(M.END)]
    assert block == M.render_md(res)
    assert all(g["pass"] for g in res["gate_a"])


def test_addendum_part2_matches_json_and_records_the_stop():
    import re
    from src.validation import aggregate_io
    from src.validation import b0_orientation as B
    path = ROOT / "docs" / "validation" / "b0_summary.json"
    res = json.loads(path.read_text())
    aggregate_io.check_aggregate(res)
    assert aggregate_io.ID_RE.search(path.read_text()) is None
    res.pop("_notice")
    text = (ROOT / "docs" / "external_validation_addendum_A1.md").read_text()
    block = text[text.index(B.A1_BEGIN): text.index(B.A1_END) + len(B.A1_END)]
    commit = re.search(r"commit `([0-9a-f]+)`", block).group(1)
    assert block == B.render_part2(res, commit)
    assert res["decision"]["outcome"] == "stop" and res["decision"]["proposed_convention"] is None
    assert "rsachs" not in block and "/home/" not in block


def test_protocol_states_the_orientation_decision_s6f_and_the_open_items():
    t = PROTOCOL.read_text()
    a = (ROOT / "docs" / "external_validation_addendum_A1.md").read_text()
    from src.assessment.readers import analyze75
    for needle in ("third-party documented, not OASIS-documented", "oasis1_primary", "S6f", "only by a dated addendum made **before Stage C**",
                   "[0 0 -1.25 0], [1 0 0 0], [0 1 0 0]", "oasis1_mirrored"):
        assert needle in t, needle
    for needle in ("third-party documented, not OASIS-documented", "withdrawn as an established fact", "Result: unverified",
                   "not derived from the Stage B0 evidence", "`1dc4f56c4b380884eefc8f5662f63bede9fe45da74185a3ad908be6fd8ddb874`",
                   "`7ede554d051a4442ff39dd912b06f4115ea651c0a008882f26c86d08011a30fd`", "the first 30 CDR 0 cases by sorted case id"):
        assert needle in a, needle
    import re
    assert "CDR 0.5 is not MCI" not in t and re.search(r"\bcalibrated\b", t) is None and "model probability" in t
    # S6f in the protocol: all CDR >= 1 (30) plus the first 30 CDR 0 in sorted order, mirrored convention, after the main run
    assert "all cases with CDR 1 or higher (30 cases) plus the first 30 CDR 0 cases in sorted case-id order" in t
    assert "Status: unresolved" in t and "OASIS is not named among the training data in any text read" in t
    assert "Table 3's printed age statistics" in t or "printed age statistics for CDR 0.5 and CDR 1 differ" in t
    assert set(analyze75.OASIS1_SFORMS) == {"oasis1_primary", "oasis1_mirrored"}
    assert "tag" in t


def test_declared_hashes_match_the_files_and_the_data_area_manifest():
    import hashlib
    t = PROTOCOL.read_text()

    def sha(path):
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    assert sha(ROOT / "analysis/fusion_v3/frozen/MANIFEST.sha256") in t
    assert sha(ROOT / "src/assessment/artifacts/registry.json") in t
    reg = json.loads((ROOT / "src/assessment/artifacts/registry.json").read_text())
    assert "9257e98e9aff88bfcd83acdb1e74ffde762967503ef6df16c9270aa39d9941ea" in t and "9257e98e9aff88bfcd83acdb1e74ffde762967503ef6df16c9270aa39d9941ea" in json.dumps(reg)
    man = (pathlib.Path(__import__("os").environ.get("OASIS_WORK", "oasis_work")) / "manifest")            # data area; absent on other machines
    if man.exists():
        assert sha(man / "inputs.csv") in t and sha(man / "labels.csv") in t
    assert "HEAD at tag" in t and "1dc4f56c4b380884eefc8f5662f63bede9fe45da74185a3ad908be6fd8ddb874" in t
    assert json.loads((ROOT / "docs/validation/stage_a_counts.json").read_text())["manifest_sha256"]["labels.csv"] in t


def test_memory_time_and_release_statements_are_current_and_consistent_with_the_code():
    from src.assessment import memguard
    t = PROTOCOL.read_text()
    assert "11.5 GB" not in t and "15 GB RAM" not in t
    for needle in ("12.62 GiB", "13.61 GiB", "13,231,404 kB", "19,999 MB", "stops the mri and dbn workers before every SynthSeg call",
                   "floor of 15.2 GiB", "about 79 s per scan warm and 104 s cold", "about 4.4 h for 198 scans", "cold bound 5.7 h"):
        assert needle in t, needle
    assert memguard.DEFAULT_FLOOR_GB == 15.2 and memguard.SYNTHSEG_PEAK_GB == 13.61
    assert memguard.DEFAULT_FLOOR_GB >= memguard.SYNTHSEG_PEAK_GB + 1.5
    assert 198 * 79 / 3600 == pytest.approx(4.35, abs=0.05) and 198 * 104 / 3600 == pytest.approx(5.7, abs=0.05)
    for f in ("src/assessment/volumes/synthseg.py", "src/assessment/batch.py"):
        assert "11.5 GB" not in (ROOT / f).read_text() and "13.61 GiB" in (ROOT / f).read_text()


def test_orientation_section_is_merged_with_dated_provenance_and_stage_b_has_its_qc_rule():
    t = PROTOCOL.read_text()
    sec = t[t.index("## 4a."):t.index("## 5.")]
    for needle in ("addendum A1, final text 2026-11-14", "1008ede", "third-party documented, not OASIS-documented", "not derived from the B0 evidence",
                   "symmetric ICBM152 2009a", "4 of 20", "Analyze 7.5 specification check: unverified", "withdrawn as an established fact",
                   "oasis1_mirrored", "before Stage C", "[0 0 -1.25 0], [1 0 0 0], [0 1 0 0], [0 0 0 1]"):
        assert needle in sec, needle
    st = t[t.index("**Stage B QC-plumbing rule.**"):t.index("## 11.")]
    assert "PASS or WARN in every case" in st and "no score value is read or printed" in st and "stops the run before Stage C" in st


def test_o1_headline_and_the_bashyam_count_inconsistency_are_stated():
    t = PROTOCOL.read_text()
    o1 = t[t.index("- **O1 DeepBrainNet"):t.index("- **O2 Field strength**")]
    assert "brain-age gap independence from OASIS is not established" in o1 and "18 studies" in o1 and "2 + 12 studies" in o1
    assert "7,326" in o1 and "11,729" in o1 and "OASIS is not named" in o1


TAG = "oasis-protocol-v1"
ADDENDA = "\n## Addenda\n"


def _git(*args):
    import subprocess
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def frozen_body(text):
    """Everything before the Addenda heading: the part the tag freezes."""
    assert text.count(ADDENDA) == 1, "the protocol must have exactly one Addenda heading"
    return text[:text.index(ADDENDA)]


def test_frozen_body_helper_ignores_addenda_only():
    base = "intro\nbody text\n\n## Addenda\n(none yet)\n"
    assert frozen_body(base + "- 2026-11-14 note\n") == frozen_body(base)
    assert frozen_body(base.replace("body text", "body tex7")) != frozen_body(base)
    with pytest.raises(AssertionError):
        frozen_body("no addenda heading here")


def test_protocol_tag_is_annotated_ancestor_and_body_is_frozen():
    if _git("rev-parse", "-q", "--verify", f"refs/tags/{TAG}").returncode != 0:
        pytest.skip(f"tag {TAG} does not exist yet, so there is no frozen body to compare with")
    assert _git("cat-file", "-t", f"refs/tags/{TAG}").stdout.strip() == "tag", "the tag must be annotated"
    commit = _git("rev-parse", f"{TAG}^{{commit}}").stdout.strip()
    assert _git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0, "the tag must point at an ancestor of HEAD"
    tagged = _git("show", f"{TAG}:docs/external_validation_protocol.md").stdout
    assert frozen_body(PROTOCOL.read_text()) == frozen_body(tagged), "the protocol body above Addenda changed after the tag"
