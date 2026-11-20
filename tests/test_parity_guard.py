"""The parity code reads only allowlisted fields (never labels) and computes no metrics."""
import pathlib
import re

import numpy as np
import pytest

from tests.parity import run_parity as RP


def test_allowlist_blocks_labels(tmp_path):
    f = tmp_path / "z.npz"
    np.savez(f, y=np.arange(3), rids=np.arange(3), clin_42=np.zeros((3, 4)), p=np.zeros((3, 4)))
    z = RP.Allowlisted(f, RP.STREAM_KEYS, RP.STREAM_PREFIXES)
    assert "y" not in z.files and set(z.files) == {"rids", "clin_42"}
    with pytest.raises(KeyError):
        z["y"]


def test_csv_reader_refuses_label_columns(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("RID,label\n1,2\n")
    with pytest.raises(AssertionError):
        RP.read_csv(f, ["RID", "label"])


def test_no_metric_code_in_parity():
    src = pathlib.Path(RP.__file__).read_text()
    assert not re.search(r"roc_auc|accuracy|f1_score|confusion|balanced_acc|sklearn|\[.y.\]", src)
