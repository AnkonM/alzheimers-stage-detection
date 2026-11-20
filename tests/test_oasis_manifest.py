"""Stage A manifest builder on synthetic archives (identifiers are assembled at run time)."""
import io
import json
import os
import pathlib
import re
import stat
import struct
import tarfile

import pytest

from src.validation import aggregate_io
from src.validation import oasis_manifest as M

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sid(n, mr=1):
    return "OAS" + "1_" + f"{n:04d}" + f"_MR{mr}"


def hdr_bytes(dim=(4, 5, 2), pix=(1.0, 1.0, 1.25), datatype=4, bitpix=16, orient=2, endian=">"):
    h = bytearray(348)
    h[0:4] = struct.pack(endian + "i", 348)
    h[40:56] = struct.pack(endian + "8h", 4, *dim, 1, 1, 1, 1)         # Analyze headers declare 4 dimensions, the last is 1
    h[70:74] = struct.pack(endian + "2h", datatype, bitpix)
    h[76:108] = struct.pack(endian + "8f", 0, *pix, 0, 0, 0, 0)
    h[252] = orient
    return bytes(h)


def txt(session, age, sex, cdr, mmse, scans=(1,), tr="9.7", eol="\n"):
    lines = [f"SESSION ID:   {session}", f"AGE:          {age}", f"M/F:          {sex}", "HAND:         Right",
             "EDUC:         3", "SES:          2", f"CDR:          {cdr}", f"MMSE:         {mmse}",
             "eTIV:         1500.12", "ASF:          1.17", "nWBV:         0.750", ""]
    for n in scans:
        lines += [f"mpr-{n}      MPRAGE", ""]
    for n in scans:
        lines += [f"SCAN NUMBER:  mpr-{n}", "TYPE:         MPRAGE", "Vox res (mm): 1.0 x 1.0 x 1.25", "Rect. Fov:    256/256",
                  "Orientation:  Sag", f"TR (ms):      {tr}", "TE (ms):      4.0", "TI (ms):      20.0", "Flip:         10", ""]
    return eol.join(lines) + eol


def add(tar, name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    tar.addfile(ti, io.BytesIO(data))


# (n, mr, age, sex, cdr, mmse, scan numbers in tar order)
SESSIONS = [
    (3, 1, 70, "Female", "0", "29", (2, 1)),            # mpr-2 listed before mpr-1 in the archive
    (1, 1, 60, "Male", "0.5", "26", (1, 2, 3)),          # exactly 60: included
    (2, 1, 59, "Female", "0", "30", (1,)),               # under 60
    (4, 1, 81, "Male", "1", "20", (10, 2)),              # natural order: mpr-2 before mpr-10
    (5, 1, 77, "Female", "", "", (1,)),                  # age ok, CDR blank
    (6, 2, 25, "Male", "0", "30", (1,)),                 # retest, young
    (7, 1, 88, "Female", "2", "15", ()),                 # no raw scan
    (8, 1, 90, "Female", "0", "28", (2, 3)),             # mpr-1 absent: first scan is mpr-2
]


@pytest.fixture
def archives(tmp_path):
    d = tmp_path / "arch"
    d.mkdir()
    marks = {}
    for k, chunk in enumerate((SESSIONS[:4], SESSIONS[4:])):
        with tarfile.open(d / f"oasis_cross-sectional_disc{k + 1}.tar.gz", "w:gz") as t:
            add(t, f"disc{k + 1}/not-a-session.txt", b"x")
            for n, mr, age, sex, cdr, mmse, scans in chunk:
                s = sid(n, mr)
                base = f"disc{k + 1}/{s}"
                add(t, f"{base}/{s}.txt", txt(s, age, sex, cdr, mmse, scans, eol="\r\n").encode("latin-1"))    # the real files use CRLF
                add(t, f"{base}/{s}.xml", b"<x/>")
                for sc in scans:
                    payload = bytes([sc]) * 40                       # 4*5*2*2 bytes
                    marks[(s, sc)] = payload
                    add(t, f"{base}/RAW/{s}_mpr-{sc}_anon.hdr", hdr_bytes())
                    add(t, f"{base}/RAW/{s}_mpr-{sc}_anon.img", payload)
                    add(t, f"{base}/RAW/{s}_mpr-{sc}_anon_sag_66.gif", b"GIF89a")
                add(t, f"{base}/PROCESSED/MPRAGE/T88_111/{s}_mpr_n4_anon_111_t88_gfc.img", b"p" * 8)
                add(t, f"{base}/PROCESSED/MPRAGE/T88_111/{s}_mpr_n4_anon_111_t88_gfc.hdr", hdr_bytes())
                add(t, f"{base}/PROCESSED/MPRAGE/T88_111/t4_files/{s}_mpr-1_anon_to_711-2C_t4", b"t4")
                add(t, f"{base}/RAW/../../escape.txt", b"nope")   # traversal decoy
    return d, marks


@pytest.fixture
def built(archives, tmp_path):
    arch, marks = archives
    data = tmp_path / "data"
    files = sorted(arch.glob("*.tar.gz"))
    index = M.scan_archives(files, data)
    sessions_idx = {k: {**v, "raw": {int(n): r for n, r in v["raw"].items()}} for k, v in index["sessions"].items()}
    records = M.load_sessions(data)
    rows, ex = M.select_cohort(records, sessions_idx)
    return dict(files=files, data=data, index=index, sidx=sessions_idx, records=records, rows=rows, ex=ex, marks=marks)


def test_parse_txt_fields_and_blanks():
    r = M.parse_txt(txt(sid(1), 74, "Female", "0.5", "27", scans=(1, 2)))
    assert (r["session"], r["age"], r["sex"], r["cdr"], r["mmse"], r["etiv"]) == (sid(1), 74, "F", 0.5, 27, 1500.12)
    assert len(r["scans"]) == 2 and r["scans"][0]["vox"] == (1.0, 1.0, 1.25) and r["scans"][0]["fov"] == (256.0, 256.0)
    b = M.parse_txt(txt(sid(2), 77, "Male", "", ""))
    assert b["cdr"] is None and b["mmse"] is None
    with pytest.raises(ValueError):
        M.parse_txt(txt(sid(3), 77, "Male", "0.25", "20"))        # CDR off the scale


def test_pass1_writes_only_small_text_files_and_indexes_raw(built):
    meta = sorted(p.name for p in (built["data"] / "meta").iterdir())
    assert meta == sorted([f"{sid(n, mr)}.{e}" for n, mr, *_ in SESSIONS for e in ("txt", "xml")])
    assert sorted(p.name for p in built["data"].iterdir()) == ["meta"]          # no payload, no raw folder
    s = built["sidx"][sid(3)]
    assert sorted(s["raw"]) == [1, 2] and s["raw"][1]["hdr"]["orient"] == 2 and s["raw"][1]["img_size"] == 40
    assert s["t88_hdr"] is True and s["t4_711"] == [1]
    assert built["index"]["counts"]["unexpected_members"] >= 1                   # the decoy text file


def test_cohort_rule_and_exclusions(built):
    rows, ex = built["rows"], built["ex"]
    ages = sorted(r["age"] for r in rows)
    assert ages == [60, 70, 81, 90]                                              # 59, blank-CDR, retest, no-raw out
    assert ex == {"sessions": 8, "excluded_not_first_session": 1, "excluded_age_missing": 0, "excluded_age_under_60": 1,
                  "excluded_cdr_missing": 1, "excluded_sex_missing": 0, "excluded_no_raw_scan": 1, "first_scan_is_not_mpr1": 2, "cohort": 4}


def test_scan_choice_is_first_by_number_and_case_ids_follow_sorted_session_ids(built):
    by_age = {r["age"]: r for r in built["rows"]}
    assert by_age[70]["scan"] == 1                  # mpr-2 listed first in the archive, mpr-1 chosen
    assert by_age[81]["scan"] == 2                  # natural order, not string order (10 after 2)
    assert by_age[90]["scan"] == 2                  # mpr-1 absent
    order = [(r["case_id"], r["session"]) for r in built["rows"]]
    assert order == sorted(order, key=lambda x: x[1]) and [c for c, _ in order] == ["X0001", "X0002", "X0003", "X0004"]
    assert built["rows"][0]["age"] == 60            # session number 1


def test_manifest_files_separate_inputs_from_labels(built):
    h = M.write_manifest(built["rows"], built["data"])
    man = built["data"] / "manifest"
    inp = (man / "inputs.csv").read_text().splitlines()
    lab = (man / "labels.csv").read_text().splitlines()
    assert inp[0] == ",".join(M.INPUT_COLUMNS) and lab[0] == ",".join(M.LABEL_COLUMNS)
    assert not set(M.INPUT_COLUMNS) & {"CDR", "MMSE", "eTIV", "nWBV"}
    first = inp[1].split(",")
    assert first[0] == "X0001" and first[2:4] == ["analyze75", "UNSET"] and first[-1] == "synthseg" and first[6] == "1"
    assert first[1].endswith("_mpr-1_anon.hdr") and str(built["data"]) in first[1]
    assert lab[1].split(",")[:2] == ["X0001", "0.5"]
    assert h["inputs.csv"] == M.sha256_file(man / "inputs.csv") and len(h["labels.csv"]) == 64
    assert (built["data"] / "private" / "mapping.csv").read_text().splitlines()[0] == ",".join(M.MAPPING_COLUMNS)
    for p in (man / "inputs.csv", man / "labels.csv"):
        assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_input_columns_extend_the_runner_columns():
    from src.assessment import batch
    assert M.INPUT_COLUMNS[:6] == batch.INPUT_COLUMNS       # the runner currently demands exactly these six


def test_extract_raw_copies_only_cohort_raw_pairs_byte_exact(built):
    n = M.extract_raw(built["files"], built["rows"], built["data"])
    raw = built["data"] / "raw"
    assert n == sum(2 * len(s[6]) for s in SESSIONS if s[0] in (1, 3, 4, 8))
    assert sorted(p.name for p in raw.iterdir()) == sorted(r["session"] for r in built["rows"])
    names = [p.name for d in raw.iterdir() for p in d.iterdir()]
    pat = "OAS" + "1_" + r"\d{4}_MR1_mpr-\d+_anon\.(hdr|img)"
    assert all(re.fullmatch(pat, x) for x in names)
    assert not any("gif" in x or "t88" in x for x in names)
    assert not (built["data"] / "escape.txt").exists() and not (raw.parent.parent / "escape.txt").exists()
    for (s, sc), payload in built["marks"].items():
        p = raw / s / f"{s}_mpr-{sc}_anon.img"
        if p.parent.exists():
            assert p.read_bytes() == payload
    d = next(raw.iterdir())
    assert stat.S_IMODE(d.stat().st_mode) == 0o700 and stat.S_IMODE(next(d.iterdir()).stat().st_mode) == 0o600


def test_second_code_path_agrees_and_detects_a_changed_text_file(built):
    second = M.recount_second_path(built["data"])
    assert second["sessions"] == 8 and second["subjects"] == 8 and second["not_first_session"] == 1
    checks = dict(M.gate_a(built["rows"], built["records"], built["sidx"], second))
    key = [k for k in checks if k.startswith("second code path")][0]
    assert checks[key] is True
    # mutate one age in a text file AFTER the first path parsed it: the awk recount must now disagree
    p = built["data"] / "meta" / f"{sid(3)}.txt"
    p.write_bytes(p.read_bytes().replace(b"AGE:          70", b"AGE:          71"))
    second2 = M.recount_second_path(built["data"])
    assert dict(M.gate_a(built["rows"], built["records"], built["sidx"], second2))[key] is False


def test_gate_a_reports_failures_instead_of_forcing_counts(built):
    checks = dict(M.gate_a(built["rows"], built["records"], built["sidx"], M.recount_second_path(built["data"])))
    assert checks["sessions in the archives = 436"] is False and checks["cohort by CDR = 98/70/28/2"] is False
    assert checks["no duplicate case ids"] is True
    assert checks["every session that meets the age and CDR rule has a raw scan and a sex"] is False   # session without raw scan


def test_table3_comparison_agree_and_mismatch():
    rows = [{"cdr": 2.0, "age": 78, "sex": "F", "mmse": 15}, {"cdr": 2.0, "age": 86, "sex": "M", "mmse": 15}]
    t = M.table3_comparison(rows)["2"]
    assert t["agree"] == {"n": True, "sex": True, "age": True, "mmse": True} and t["age"] == [82.0, 5.7, 78, 86]
    rows[1]["age"] = 87
    assert M.table3_comparison(rows)["2"]["agree"]["age"] is False
    rows[1]["mmse"] = 14
    assert M.table3_comparison(rows)["2"]["agree"]["mmse"] is False
    rows[1]["mmse"] = None
    assert M.table3_comparison(rows)["2"]["mmse_missing"] == 1
    assert M.table3_comparison([])["0.5"]["n"] == 0


def test_s4_tertile_edges_are_observed_ages():
    rows = [{"cdr": 0.0, "age": a} for a in (60, 62, 64, 70, 71, 72, 80, 85, 90)] + [{"cdr": 1.0, "age": 99}]
    e = M.s4_age_tertile_edges(rows)
    assert e["edges"] == [64.0, 72.0] and e["group_sizes_cdr0"] == [3, 3, 3] and e["n_cdr0"] == 9
    with pytest.raises(M.ManifestError):
        M.s4_age_tertile_edges(rows[:2])


def test_aggregate_is_clean_and_protocol_block_is_replaced_idempotently(built, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "s4_age_tertile_edges", lambda rows: {"edges": [64.0, 72.0], "n_cdr0": 9, "method": "stub",
                                                              "group_sizes_cdr0": [3, 3, 3]})   # tested separately
    h = M.write_manifest(built["rows"], built["data"])
    second = M.recount_second_path(built["data"])
    checks = M.gate_a(built["rows"], built["records"], built["sidx"], second)
    res = M.build_aggregate(built["rows"], built["ex"], built["records"], built["sidx"], checks, h,
                            {"archives": 2, **built["index"]["counts"]})
    cases = [r["case_id"] for r in built["rows"]]
    aggregate_io.check_aggregate(res, cases)
    text = json.dumps(res)
    assert not aggregate_io.ID_RE.search(text) and not any(c in text for c in cases)
    assert res["retest_sessions"] == 1 and res["retest_max_age"] == 25 and res["cohort"]["n"] == 4
    assert res["raw_headers_vs_documentation"]["disagreeing_scans_by_property"]["dimensions"] > 0   # synthetic dims differ
    md = M.render_md(res)
    aggregate_io.check_aggregate(md, cases)
    proto = tmp_path / "p.md"
    proto.write_text(f"head\n{M.BEGIN}\nold\n{M.END}\ntail\n")
    assert M.update_protocol(md, proto) is True and M.update_protocol(md, proto) is False
    t = proto.read_text()
    assert t.startswith("head\n") and t.endswith("tail\n") and "old" not in t and md in t


def test_header_and_text_disagreements_are_counted_per_session(tmp_path):
    recs = {sid(1): M.parse_txt(txt(sid(1), 70, "Male", "0", "29")), sid(2): M.parse_txt(txt(sid(2), 70, "Male", "0", "29", tr="9.8"))}
    assert M._agree_txt_with_docs(recs) == {"TR": 1, "any field": 1}
    ok = {"archive": "a", "raw": {1: {"hdr": M.parse_header(hdr_bytes((256, 256, 128))), "img_size": 256 * 256 * 128 * 2}}}
    bad = {"archive": "a", "raw": {1: {"hdr": M.parse_header(hdr_bytes((256, 256, 128), endian="<")), "img_size": 7}}}
    r = M._agree_headers_with_docs({sid(1): ok, sid(2): bad})
    assert r["sessions_with_a_disagreeing_scan"] == 1 and r["disagreeing_scans_by_property"] == {"byte order": 1, "image file size": 1}
    assert r["headers_declaring_4_dimensions_with_a_trailing_1"] == 2           # the 4th dimension is not a disagreement
    h4 = M.parse_header(hdr_bytes((256, 256, 128)))
    h4["dim"] = h4["dim"][:3] + (2,)                                            # a real fourth dimension is one
    assert M._agree_headers_with_docs({sid(3): {"archive": "a", "raw": {1: {"hdr": h4, "img_size": 256 * 256 * 128 * 4}}}}
                                      )["disagreeing_scans_by_property"] == {"dimensions": 1}
    three = M.parse_header(hdr_bytes((256, 256, 128)))
    three["dim"], three["pixdim"] = three["dim"][:3], three["pixdim"][:3]       # 3-D header
    assert M._agree_headers_with_docs({sid(4): {"archive": "a", "raw": {1: {"hdr": three, "img_size": 256 * 256 * 128 * 2}}}}
                                      )["sessions_with_a_disagreeing_scan"] == 0


def test_source_carries_no_identifier_literal():
    assert not aggregate_io.ID_RE.search((ROOT / "src" / "validation" / "oasis_manifest.py").read_text())
    assert not aggregate_io.ID_RE.search(pathlib.Path(__file__).read_text())


def test_extract_raw_skips_complete_files_and_rewrites_short_ones(built):
    M.extract_raw(built["files"], built["rows"], built["data"])
    a = next((built["data"] / "raw").rglob("*.img"))
    mtime = a.stat().st_mtime_ns
    b = next(p for p in (built["data"] / "raw").rglob("*.img") if p != a)
    b.write_bytes(b"short")
    n = M.extract_raw(built["files"], built["rows"], built["data"])
    assert n == 18 and a.stat().st_mtime_ns == mtime and b.stat().st_size == 40


def test_table2_comparison_agrees_only_on_the_exact_cells():
    ages = {"60s": [60, 65, 69], "70s": [70, 80 - 1]}
    rows = [{"cdr": 0.0, "age": a} for a in ages["60s"]] + [{"cdr": 0.5, "age": 66}, {"cdr": 1.0, "age": 66}]
    t = M.table2_comparison(rows)
    assert t["60s"]["nondemented"] == {"n": 3, "mean": 64.67} and t["60s"]["demented"]["cdr_0.5_1_2"] == [1, 1, 0]
    assert t["60s"]["agree"] is False and t["90+"]["nondemented"]["n"] == 0
    full = ([{"cdr": 0.0, "age": 64.88}] * 25)                                  # constructs the 60s non-demented cell exactly
    full += [{"cdr": 0.5, "age": 66.13}] * 12 + [{"cdr": 1.0, "age": 66.13}] * 3
    assert M.table2_comparison(full)["60s"]["agree"] is True


def test_set_orientation_changes_only_that_column_and_is_idempotent(built):
    M.write_manifest(built["rows"], built["data"])
    path = built["data"] / "manifest" / "inputs.csv"
    old = path.read_text().splitlines()
    h = M.set_orientation(built["data"], "oasis1_primary")
    new = path.read_text().splitlines()
    assert h["before"] != h["after"] == M.sha256_file(path) and len(new) == len(old)
    for a, b in zip(old[1:], new[1:]):
        fa, fb = a.split(","), b.split(",")
        assert fa[3] == "UNSET" and fb[3] == "oasis1_primary" and fa[:3] + fa[4:] == fb[:3] + fb[4:]
    assert new[0] == old[0]
    assert M.set_orientation(built["data"], "oasis1_primary")["after"] == h["after"]          # idempotent
    with pytest.raises(M.ManifestError):                                                       # no silent switch
        M.set_orientation(built["data"], "oasis1_mirrored")
    with pytest.raises(M.ManifestError):
        M.set_orientation(built["data"], "radiological-ish")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
