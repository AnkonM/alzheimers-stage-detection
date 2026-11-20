import json

from src.assessment.volumes import agreement


def test_caveat_quotes_committed_numbers_and_flags_low_cortical():
    a = agreement.load()
    assert a["meta"]["n_ok"] == 25
    c = agreement.caveats(a)
    assert c[0]["id"] == "volume_provider" and "Entorhinal 0.15" in c[0]["text"]
    assert [x["id"] for x in c][1:] == ["volume_unreliable"]
    assert "calibrat" not in json.dumps(c).lower()


def test_no_flag_when_cortical_ok():
    a = agreement.load()
    a["cortical_below_floor"] = []
    assert [x["id"] for x in agreement.caveats(a)] == ["volume_provider"]
