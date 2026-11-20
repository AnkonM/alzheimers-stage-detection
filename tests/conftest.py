"""Skip the tests that need material this repository does not distribute.

Trained weights, the frozen subject-level reference arrays and the OASIS-1 / ADNI
data areas are available only to users with their own data access approval
(see DATA.md). Those tests are skipped when the material is absent and run when
it has been restored to its original location.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HAVE_FROZEN = (ROOT / "analysis" / "fusion_v3" / "frozen" / "feature_stats.json").exists()

NEEDS_FROZEN_FILES = {"test_api.py"}
NEEDS_FROZEN_TESTS = {
    "test_registry_verifies_and_detects_tampering",
    "test_clinical_hard_range_rejects_and_unknown_field",
    "test_ravlt_derivation_matches_adni_definition",
    "test_completeness_imputes_with_median_and_flags_required",
    "test_mri_only_native_scan_needs_age_and_sex",
    "test_numbers_are_engine_outputs_exactly",
    "test_clinical_only_has_no_mri_or_severity",
    "test_synthseg_volumes_follow_display_rule",
    "test_adni_volumes_have_percentiles_from_committed_grid",
    "test_adni_control_ids_are_train_only_sorted_and_need_a_stored_volume",
    "test_case_ids_under_repo_var_detected",
    "test_memory_floor_halts_cleanly_records_state_and_resume_completes",
    "test_declared_hashes_match_the_files_and_the_data_area_manifest",
}


def pytest_collection_modifyitems(config, items):
    if HAVE_FROZEN:
        return
    skip = pytest.mark.skip(reason="needs trained weights / frozen reference data (not distributed; see DATA.md)")
    for item in items:
        if item.path.name in NEEDS_FROZEN_FILES or item.originalname in NEEDS_FROZEN_TESTS:
            item.add_marker(skip)
