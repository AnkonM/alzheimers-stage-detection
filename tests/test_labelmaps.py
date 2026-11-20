"""Label-map registry: cdr_3class mapping, strict errors, plugin registration."""
import numpy as np
import pytest

from src.assessment import labelmaps as L


@pytest.mark.parametrize("raw,label", [(0, "CN-like"), ("0", "CN-like"), (0.0, "CN-like"), (np.float64(0.0), "CN-like"),
                                       (0.5, "CDR0.5"), ("0.5", "CDR0.5"), (1, "dementia"), ("1.0", "dementia"),
                                       ("2.0", "dementia"), (2, "dementia"), (3, "dementia")])
def test_cdr_3class_mapping(raw, label):
    assert L.get("cdr_3class")(raw) == label


@pytest.mark.parametrize("raw", [None, "", "  ", "nan", float("nan"), "n/a", -1, 0.25, 4, "abc", "1.5"])
def test_cdr_3class_rejects_unknown_values(raw):
    with pytest.raises(ValueError):
        L.get("cdr_3class")(raw)


def test_ranks_and_contrast_definition():
    m = L.get("cdr_3class")
    assert [m.rank(x) for x in (0, 0.5, 1, 2)] == [0, 1, 2, 2]
    assert (m.reference, m.positive) == ("CN-like", "dementia")
    assert m.classes == ("CN-like", "CDR0.5", "dementia")


def test_unknown_map_name_lists_registered():
    with pytest.raises(KeyError) as e:
        L.get("nope")
    assert "cdr_3class" in str(e.value)


def test_plugin_registration_duplicates_and_undeclared_class():
    @L.register("tmp_map", classes=("lo", "hi"), reference="lo", positive="hi")
    def tmp(raw):
        return "lo" if float(raw) < 1 else "hi"
    try:
        assert L.get("tmp_map")(0) == "lo" and "tmp_map" in L.names()
        with pytest.raises(ValueError):
            L.register("tmp_map", classes=("lo", "hi"), reference="lo", positive="hi")(tmp)
    finally:
        L.unregister("tmp_map")

    @L.register("bad_map", classes=("a", "b"), reference="a", positive="b")
    def bad(raw):
        return "c"
    try:
        with pytest.raises(ValueError):
            L.get("bad_map")(0)
    finally:
        L.unregister("bad_map")

    with pytest.raises(ValueError):
        L.register("x", classes=("a", "b"), reference="a", positive="z")
