"""Local check (skipped without local data): the engine's explanations for an ADNI demo case equal the card data
(demo/card_data, built once by src/card/*) for the same subject. Run after `assess predict --adni-rid <rid>
--report-id demo-cn`; set DEMO_RID to the case's RID."""
import os
import pathlib

import numpy as np
import pytest

from src.assessment import adni_demo

ROOT = pathlib.Path(__file__).resolve().parents[1]
RID = os.environ.get("DEMO_RID")
RUN = ROOT / "var" / "runs" / "demo-cn"
CARD = ROOT / "demo" / "card_data"
pytestmark = pytest.mark.skipif(not (RID and RUN.exists() and CARD.exists()), reason="local demo data not present")


def test_explanations_match_card_data():
    rid = int(RID)
    ptid = adni_demo.case(rid)["ptid"]
    import pandas as pd
    subj = pd.read_csv(CARD / "subjects.csv")
    i = int(np.flatnonzero(subj.PTID.to_numpy() == ptid)[0])
    z = np.load(RUN / "job" / "streams.npz")
    cs = np.load(CARD / "clinical_shap.npz", allow_pickle=True)["shap"][i]
    assert np.abs(cs - z["cshap"][0]).max() < 1e-4
    ms = np.load(CARD / "mri_shap.npz", allow_pickle=True)["phi"][i]
    assert np.nanmax(np.abs(ms - z["mshap_phi"][0])) < 1e-4
    ks = np.load(CARD / "kan_shapley.npz", allow_pickle=True)
    k = np.load(RUN / "job" / "shapley.npz")
    assert np.abs(ks["phi_group"][:, i] - k["phi_group"][:, 0]).max() < 1e-3
    assert np.abs(ks["p_mri_masked"][i] - k["p_mri_masked"][0]).max() < 1e-4
    kt = np.load(CARD / "kan_terms.npz")
    kn = np.load(RUN / "job" / "kan.npz")
    assert np.abs(kt["p"][i] - kn["p"][0]).max() < 1e-4 and abs(kt["add"][i] - kn["add"][0]) < 1e-4
