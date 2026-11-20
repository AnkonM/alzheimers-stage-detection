"""Stage A: build the OASIS-1 cohort manifest from the downloaded archives. No model, no image decoding.

Pass 1 (`scan_archives`) streams every archive once. It writes only the small per-session text and XML files
into the data area, and reads the 348-byte header of each raw scan; it extracts no image payload.
`select_cohort` applies the protocol cohort rule to the parsed text files.
Pass 2 (`extract_raw`) streams the archives again and copies the raw `.hdr`/`.img` pairs of the cohort
sessions into the data area, byte for byte. Nothing is decoded or viewed.

Everything per-subject lives in the data area (outside every git work tree). The functions return and print
aggregates only. Session ids are built at run time so that this file carries no identifier.
"""
import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import tarfile
from collections import Counter

from ..assessment.labelmaps import parse_cdr
from ..assessment.readers.analyze75 import RefusedInput, parse_header

SESSION_RE = re.compile("OAS" + r"1_([0-9]{4})_MR([0-9]+)")
MIN_AGE = 60
SCAN_FORMAT = "analyze75"
ORIENTATION_UNSET = "UNSET"
VOLUME_SOURCE = "synthseg"
INPUT_COLUMNS = ("case_id", "scan_path", "scan_format", "orientation", "age", "sex", "scan_rank", "volume_source")
LABEL_COLUMNS = ("case_id", "CDR", "MMSE", "eTIV", "nWBV")
MAPPING_COLUMNS = ("case_id", "session_id", "subject_id", "scan_rank")

# Documented acquisition (paper Table 1, page 1500; fact sheet Table 1) checked against the per-session text.
DOC_SCAN = {"TR": 9.7, "TE": 4.0, "TI": 20.0, "flip": 10.0, "orientation": "Sag", "vox": (1.0, 1.0, 1.25), "fov": (256.0, 256.0)}
DOC_HEADER = {"dim": (256, 256, 128), "pixdim": (1.0, 1.0, 1.25), "datatype": 4, "bitpix": 16, "endian": ">", "orient": 2}
DOC_SCANS_PER_SESSION = (3, 4)

# Table 3 of the paper (page 1503), older subjects: n, female/male, age (mean, SD, min, max), MMSE (mean, SD, min, max).
TABLE3 = {
    "0": {"n": 98, "f": 72, "m": 26, "age": (75.9, 9.0, 60, 94), "mmse": (29.0, 1.2, 25, 30)},
    "0.5": {"n": 70, "f": 39, "m": 31, "age": (76.4, 7.0, 63, 92), "mmse": (25.6, 3.5, 14, 30)},
    "1": {"n": 28, "f": 19, "m": 9, "age": (77.2, 7.5, 62, 96), "mmse": (21.7, 3.8, 15, 29)},
    "2": {"n": 2, "f": 1, "m": 1, "age": (82.0, 5.7, 78, 86), "mmse": (15.0, 0.0, 15, 15)},
}
# Table 2 of the paper (page 1502; also the fact sheet), older decades: non-demented (n, mean age), demented (n, mean age, CDR 0.5/1/2)
TABLE2 = {"60s": ((25, 64.88), (15, 66.13, (12, 3, 0))), "70s": ((35, 73.37), (48, 74.42, (32, 15, 1))),
          "80s": ((30, 84.07), (32, 82.88, (22, 9, 1))), "90+": ((8, 91.00), (5, 92.00, (4, 1, 0)))}
DOC_COUNTS = {"sessions": 436, "subjects": 416, "mr2_sessions": 20}   # fact sheet: 416 subjects; paper: 20 retests

_SESSION_KEYS = {"SESSION ID": "session", "AGE": "age", "M/F": "sex", "HAND": "hand", "EDUC": "educ", "SES": "ses",
                 "CDR": "cdr", "MMSE": "mmse", "eTIV": "etiv", "ASF": "asf", "nWBV": "nwbv"}
_SCAN_KEYS = {"TYPE": "type", "Vox res (mm)": "vox", "Rect. Fov": "fov", "Orientation": "orientation",
              "TR (ms)": "TR", "TE (ms)": "TE", "TI (ms)": "TI", "Flip": "flip"}


class ManifestError(RuntimeError):
    """Configuration or data error. Messages never contain identifiers or per-subject values."""


# ---------------------------------------------------------------- parsing
def _num(v, cast=float):
    v = v.strip()
    if not v:
        return None
    try:
        return cast(float(v)) if cast is int else cast(v)
    except ValueError:
        raise ManifestError("a numeric field could not be parsed") from None


def parse_txt(text):
    """Per-session text file -> dict. Blank fields become None; a CDR off the scale raises."""
    rec = {"scans": []}
    cur = None
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if k == "SCAN NUMBER":
            cur = {"scan": v}
            rec["scans"].append(cur)
        elif k in _SCAN_KEYS and cur is not None:
            name = _SCAN_KEYS[k]
            if name in ("TR", "TE", "TI", "flip"):
                cur[name] = _num(v)
            elif name == "vox":
                cur[name] = tuple(float(x) for x in re.split(r"\s*x\s*", v)) if v else None
            elif name == "fov":
                cur[name] = tuple(float(x) for x in v.split("/")) if v else None
            else:
                cur[name] = v or None
        elif k in _SESSION_KEYS:
            name = _SESSION_KEYS[k]
            if name == "age" or name == "mmse":
                rec[name] = _num(v, int)
            elif name in ("etiv", "asf", "nwbv"):
                rec[name] = _num(v)
            elif name == "cdr":
                rec[name] = parse_cdr(v) if v else None
            elif name == "sex":
                rec[name] = {"M": "M", "F": "F", "Male": "M", "Female": "F"}.get(v)
            else:
                rec[name] = v or None
    return rec


# ---------------------------------------------------------------- pass 1
def _mkdir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True, mode=0o700)
    return pathlib.Path(p)


def _write_private(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def scan_archives(archives, data_dir):
    """Stream every archive once. Writes `<data>/meta/<session>.txt|xml` and returns the index:
    {"sessions": {session: {"archive", "raw": {n: {"hdr", "img_size"}}, "t4_711": [n], "t88_hdr": bool}}, "counts": {...}}.
    The index holds identifiers, so it is written only into the data area by the caller."""
    meta = _mkdir(pathlib.Path(data_dir) / "meta")
    sessions, counts = {}, Counter()
    for arch in sorted(archives):
        with tarfile.open(arch, "r|gz") as tar:
            for m in tar:
                counts["members"] += 1
                if not m.isfile():
                    continue
                parts = m.name.split("/")
                sm = SESSION_RE.fullmatch(parts[1]) if len(parts) >= 3 else None
                if sm is None:
                    counts["unexpected_members"] += 1
                    continue
                sid = parts[1]
                s = sessions.setdefault(sid, {"archive": os.path.basename(arch), "raw": {}, "t4_711": [], "t88_hdr": False})
                rel = parts[2:]
                if len(rel) == 1 and rel[0] in (sid + ".txt", sid + ".xml"):
                    _write_private(meta / rel[0], tar.extractfile(m).read())
                elif len(rel) == 2 and rel[0] == "RAW":
                    rm = re.fullmatch(re.escape(sid) + r"_mpr-([0-9]+)_anon\.(hdr|img)", rel[1])
                    if rm is None:
                        continue                                   # preview pictures and anything else: skipped
                    n = int(rm.group(1))
                    r = s["raw"].setdefault(n, {"hdr": None, "img_size": None})
                    if rm.group(2) == "hdr":
                        try:
                            h = parse_header(tar.extractfile(m).read(348))
                            r["hdr"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in h.items()}
                        except RefusedInput:
                            r["hdr"] = {"invalid": True}
                    else:
                        r["img_size"] = int(m.size)
                elif len(rel) == 5 and rel[:4] == ["PROCESSED", "MPRAGE", "T88_111", "t4_files"]:
                    tm = re.fullmatch(re.escape(sid) + r"_mpr-([0-9]+)_anon_to_711-2C_t4", rel[4])
                    if tm:
                        s["t4_711"].append(int(tm.group(1)))
                elif len(rel) == 4 and rel[:3] == ["PROCESSED", "MPRAGE", "T88_111"]:
                    if re.fullmatch(re.escape(sid) + r"_mpr_n[0-9]+_anon_111_t88_gfc\.hdr", rel[3]):
                        s["t88_hdr"] = True
    return {"sessions": sessions, "counts": dict(counts), "archives": len(list(archives))}


def load_sessions(data_dir):
    """Parse every `meta/*.txt` in the data area -> {session: record}."""
    out = {}
    for p in sorted((pathlib.Path(data_dir) / "meta").glob("*.txt")):
        rec = parse_txt(p.read_text(encoding="latin-1"))
        if rec.get("session") != p.stem:
            raise ManifestError("a text file's session id does not match its file name")
        out[p.stem] = rec
    return out


# ---------------------------------------------------------------- cohort
def _natural_first_scan(raw):
    """Lowest scan number that has both a header and an image."""
    for n in sorted(raw):
        if raw[n].get("hdr") and raw[n].get("img_size") is not None:
            return n
    return None


def select_cohort(records, index_sessions):
    """Protocol cohort: MR1 session, age >= 60, CDR recorded, a raw scan present. Returns (rows, exclusions).
    Scan = the first raw scan by number (mpr-1 whenever it exists). Rows are sorted by session id."""
    ex = Counter({k: 0 for k in ("sessions", "excluded_not_first_session", "excluded_age_missing", "excluded_age_under_60",
                                 "excluded_cdr_missing", "excluded_sex_missing", "excluded_no_raw_scan", "first_scan_is_not_mpr1")})
    rows = []
    for sid in sorted(records):
        rec = records[sid]
        mr = int(SESSION_RE.fullmatch(sid).group(2))
        ex["sessions"] += 1
        if mr != 1:
            ex["excluded_not_first_session"] += 1
        elif rec.get("age") is None:
            ex["excluded_age_missing"] += 1
        elif rec["age"] < MIN_AGE:
            ex["excluded_age_under_60"] += 1
        elif rec.get("cdr") is None:
            ex["excluded_cdr_missing"] += 1
        elif rec.get("sex") is None:
            ex["excluded_sex_missing"] += 1
        else:
            n = _natural_first_scan(index_sessions.get(sid, {}).get("raw", {}))
            if n is None:
                ex["excluded_no_raw_scan"] += 1
                continue
            if n != 1:
                ex["first_scan_is_not_mpr1"] += 1
            rows.append({"session": sid, "subject": SESSION_RE.fullmatch(sid).group(1), "age": rec["age"], "sex": rec["sex"],
                         "cdr": rec["cdr"], "mmse": rec.get("mmse"), "etiv": rec.get("etiv"), "nwbv": rec.get("nwbv"),
                         "educ": rec.get("educ"), "ses": rec.get("ses"), "scan": n})
    for i, r in enumerate(rows, 1):
        r["case_id"] = f"X{i:04d}"
    ex["cohort"] = len(rows)
    return rows, dict(ex)


def s4_age_tertile_edges(rows):
    """Age tertile edges from the CDR 0 cohort members: the 1/3 and 2/3 empirical quantiles (inverted CDF, so
    both edges are observed ages). Groups: age <= e1, e1 < age <= e2, age > e2."""
    import numpy as np
    ages = np.array([r["age"] for r in rows if r["cdr"] == 0.0], float)
    if len(ages) < 3:
        raise ManifestError("too few CDR 0 subjects for tertiles")
    e = np.quantile(ages, [1 / 3, 2 / 3], method="inverted_cdf")
    return {"method": "numpy.quantile inverted_cdf on cohort CDR 0 ages; groups: age <= e1, e1 < age <= e2, age > e2",
            "edges": [float(e[0]), float(e[1])], "n_cdr0": int(len(ages)),
            "group_sizes_cdr0": [int((ages <= e[0]).sum()), int(((ages > e[0]) & (ages <= e[1])).sum()), int((ages > e[1]).sum())]}


# ---------------------------------------------------------------- files for the runner and the analysis
def _csv_bytes(columns, rows):
    import io
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    w.writerows(rows)
    return buf.getvalue().encode()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def raw_hdr_path(data_dir, sid, n):
    return pathlib.Path(data_dir) / "raw" / sid / f"{sid}_mpr-{n}_anon.hdr"


def write_manifest(rows, data_dir):
    """inputs.csv (runner input) and labels.csv (analysis only) in <data>/manifest, mapping in <data>/private.
    Returns {"inputs.csv": sha256, "labels.csv": sha256}."""
    man, priv = _mkdir(pathlib.Path(data_dir) / "manifest"), _mkdir(pathlib.Path(data_dir) / "private")
    inputs = [(r["case_id"], str(raw_hdr_path(data_dir, r["session"], r["scan"])), SCAN_FORMAT, ORIENTATION_UNSET,
               r["age"], r["sex"], r["scan"], VOLUME_SOURCE) for r in rows]
    labels = [(r["case_id"], f"{r['cdr']:g}", "" if r["mmse"] is None else r["mmse"],
               "" if r["etiv"] is None else f"{r['etiv']:g}", "" if r["nwbv"] is None else f"{r['nwbv']:g}") for r in rows]
    mapping = [(r["case_id"], r["session"], r["subject"], r["scan"]) for r in rows]
    _write_private(man / "inputs.csv", _csv_bytes(INPUT_COLUMNS, inputs))
    _write_private(man / "labels.csv", _csv_bytes(LABEL_COLUMNS, labels))
    _write_private(priv / "mapping.csv", _csv_bytes(MAPPING_COLUMNS, mapping))
    return {"inputs.csv": sha256_file(man / "inputs.csv"), "labels.csv": sha256_file(man / "labels.csv")}


def set_orientation(data_dir, name):
    """Set the `orientation` column of inputs.csv from UNSET to a named loader convention. Nothing else changes.
    Refuses if any row already holds a different value. Returns {"before": sha256, "after": sha256}."""
    from ..assessment.readers.analyze75 import NAMED
    if name not in NAMED:
        raise ManifestError("not a named loader convention")
    path = pathlib.Path(data_dir) / "manifest" / "inputs.csv"
    before = sha256_file(path)
    with open(path, newline="") as f:
        rd = list(csv.reader(f))
    if tuple(rd[0]) != INPUT_COLUMNS:
        raise ManifestError("inputs.csv has unexpected columns")
    k = INPUT_COLUMNS.index("orientation")
    for r in rd[1:]:
        if r[k] not in (ORIENTATION_UNSET, name):
            raise ManifestError("a row already holds a different orientation")
        r[k] = name
    _write_private(path, _csv_bytes(rd[0], rd[1:]))
    return {"before": before, "after": sha256_file(path)}


# ---------------------------------------------------------------- pass 2
def extract_raw(archives, rows, data_dir):
    """Copy the RAW .hdr/.img pairs of every cohort session (all repetitions) into <data>/raw/<session>/. The
    destination name is rebuilt from the matched session id and scan number, never from the member path."""
    want = {r["session"] for r in rows}
    raw = _mkdir(pathlib.Path(data_dir) / "raw")
    n_files = 0
    for arch in sorted(archives):
        with tarfile.open(arch, "r|gz") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                parts = m.name.split("/")
                if len(parts) != 4 or parts[2] != "RAW" or parts[1] not in want:
                    continue
                sid = parts[1]
                rm = re.fullmatch(re.escape(sid) + r"_mpr-([0-9]+)_anon\.(hdr|img)", parts[3])
                if rm is None:
                    continue
                dest_dir = _mkdir(raw / sid)
                dest = dest_dir / f"{sid}_mpr-{int(rm.group(1))}_anon.{rm.group(2)}"
                if dest.exists() and dest.stat().st_size == m.size:      # complete copy from an earlier run
                    n_files += 1
                    continue
                fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "wb") as f:
                    shutil.copyfileobj(tar.extractfile(m), f, 1 << 20)
                if dest.stat().st_size != m.size:
                    raise ManifestError("an extracted file has the wrong size")
                n_files += 1
    return n_files


# ---------------------------------------------------------------- aggregate checks
def _agree_txt_with_docs(records):
    """Disagreeing session counts against the documented acquisition (per-session text)."""
    dis = Counter()
    for rec in records.values():
        bad = set()
        for sc in rec["scans"]:
            for k in ("TR", "TE", "TI", "flip"):
                if sc.get(k) is None or abs(sc[k] - DOC_SCAN[k]) > 1e-6:
                    bad.add(k)
            if sc.get("orientation") != DOC_SCAN["orientation"]:
                bad.add("orientation")
            if sc.get("vox") is None or any(abs(a - b) > 1e-6 for a, b in zip(sc["vox"], DOC_SCAN["vox"])):
                bad.add("voxel size")
            if sc.get("fov") is None or tuple(sc["fov"]) != DOC_SCAN["fov"]:
                bad.add("field of view")
        if not rec["scans"]:
            bad.add("no scan block")
        for k in bad:
            dis[k] += 1
        if bad:
            dis["any field"] += 1
    return dict(dis)


def _agree_headers_with_docs(index_sessions):
    """Raw header facts: scan counts per file property and sessions with a disagreeing scan."""
    scans, dis_sessions, per_prop, four_dim = 0, set(), Counter(), 0
    per_session_scans = Counter()
    for sid, s in index_sessions.items():
        per_session_scans[len(s["raw"])] += 1
        for n, r in s["raw"].items():
            scans += 1
            h = r.get("hdr") or {}
            bad = []
            if h.get("invalid") or not h:
                bad.append("header unreadable")
            else:
                dims = tuple(h["dim"])
                if len(dims) == 4 and dims[3] == 1:
                    four_dim += 1
                    dims = dims[:3]
                if dims != DOC_HEADER["dim"]:
                    bad.append("dimensions")
                if len(h["pixdim"]) < 3 or any(abs(a - b) > 1e-4 for a, b in zip(h["pixdim"][:3], DOC_HEADER["pixdim"])):
                    bad.append("voxel size")
                if h["datatype"] != DOC_HEADER["datatype"]:
                    bad.append("data type")
                if h["bitpix"] != DOC_HEADER["bitpix"]:
                    bad.append("bits per pixel")
                if h["endian"] != DOC_HEADER["endian"]:
                    bad.append("byte order")
                if h["orient"] != DOC_HEADER["orient"]:
                    bad.append("orient code")
                expect = 1
                for d in h["dim"]:
                    expect *= d
                if r.get("img_size") != expect * h["bitpix"] // 8:
                    bad.append("image file size")
            for b in bad:
                per_prop[b] += 1
            if bad:
                dis_sessions.add(sid)
    return {"raw_scans": scans, "headers_declaring_4_dimensions_with_a_trailing_1": four_dim,
            "sessions_with_a_disagreeing_scan": len(dis_sessions),
            "disagreeing_scans_by_property": dict(per_prop),
            "sessions_by_scan_count": {str(k): v for k, v in sorted(per_session_scans.items())}}


def table3_comparison(rows):
    """Recount vs the paper's Table 3 (printed precision). Aggregate only."""
    out = {}
    for g, ref in TABLE3.items():
        sel = [r for r in rows if f"{r['cdr']:g}" == g]
        ages = [r["age"] for r in sel]
        mm = [r["mmse"] for r in sel if r["mmse"] is not None]
        res = {"n": len(sel), "female": sum(r["sex"] == "F" for r in sel), "male": sum(r["sex"] == "M" for r in sel),
               "mmse_missing": sum(r["mmse"] is None for r in sel)}
        if ages:
            res["age"] = [round(statistics.mean(ages), 1), round(statistics.stdev(ages), 1) if len(ages) > 1 else 0.0,
                          min(ages), max(ages)]
        if mm:
            res["mmse"] = [round(statistics.mean(mm), 1), round(statistics.stdev(mm), 1) if len(mm) > 1 else 0.0,
                           min(mm), max(mm)]
        res["paper"] = {"n": ref["n"], "female": ref["f"], "male": ref["m"], "age": list(ref["age"]), "mmse": list(ref["mmse"])}
        res["agree"] = {"n": res["n"] == ref["n"], "sex": (res["female"], res["male"]) == (ref["f"], ref["m"]),
                        "age": res.get("age") == list(ref["age"]), "mmse": res.get("mmse") == list(ref["mmse"])}
        out[g] = res
    return out


def _decade(a):
    return "60s" if a < 70 else "70s" if a < 80 else "80s" if a < 90 else "90+"


def table2_comparison(rows):
    """Recount vs the paper's Table 2 (decade x dementia status). Aggregate only."""
    out = {}
    for d, ((nn, nm), (dn, dm, cc)) in TABLE2.items():
        nd = [r["age"] for r in rows if _decade(r["age"]) == d and r["cdr"] == 0]
        de = [r for r in rows if _decade(r["age"]) == d and r["cdr"] > 0]
        cnt = [sum(r["cdr"] == v for r in de) for v in (0.5, 1.0, 2.0)]
        res = {"nondemented": {"n": len(nd), "mean": round(statistics.mean(nd), 2) if nd else None},
               "demented": {"n": len(de), "mean": round(statistics.mean(r["age"] for r in de), 2) if de else None, "cdr_0.5_1_2": cnt},
               "paper": {"nondemented": {"n": nn, "mean": nm}, "demented": {"n": dn, "mean": dm, "cdr_0.5_1_2": list(cc)}}}
        res["agree"] = (res["nondemented"] == res["paper"]["nondemented"] and res["demented"] == res["paper"]["demented"])
        out[d] = res
    return out


# ---------------------------------------------------------------- second code path (awk), Gate A
_AWK = r"""
BEGIN { FS = ":" }
FNR == 1 && NR > 1 { print sid "\t" age "\t" sex "\t" cdr "\t" mm; sid = age = sex = cdr = mm = "" }
/^SESSION ID/ { sid = $2; gsub(/[ \r]/, "", sid) }
/^AGE/        { age = $2; gsub(/[ \r]/, "", age) }
/^M\/F/       { sex = $2; gsub(/[ \r]/, "", sex) }
/^CDR/        { cdr = $2; gsub(/[ \r]/, "", cdr) }
/^MMSE/       { mm = $2; gsub(/[ \r]/, "", mm) }
END { print sid "\t" age "\t" sex "\t" cdr "\t" mm }
"""


def recount_second_path(data_dir):
    """Independent recount straight from the text files with awk (different code and different tool from
    `parse_txt`). Returns aggregates for the cohort rule and the session/subject counts."""
    files = sorted(str(p) for p in (pathlib.Path(data_dir) / "meta").glob("*.txt"))
    out = subprocess.run(["awk", _AWK] + files, capture_output=True, text=True, check=True).stdout
    sessions, subjects, mr2 = 0, set(), 0
    groups = {}
    for line in out.splitlines():
        sid, age, sex, cdr, mm = (line.split("\t") + [""] * 5)[:5]
        sessions += 1
        m = re.fullmatch("OAS" + r"1_([0-9]{4})_MR([0-9]+)", sid)
        subjects.add(m.group(1))
        if m.group(2) != "1":
            mr2 += 1
            continue
        if age == "" or int(float(age)) < MIN_AGE or cdr == "":
            continue
        g = groups.setdefault(cdr if cdr != "0.0" else "0", {"age": [], "mmse": [], "F": 0, "M": 0})
        g["age"].append(float(age))
        if mm != "":
            g["mmse"].append(float(mm))
        g[{"Female": "F", "Male": "M"}.get(sex, sex)] += 1
    res = {"sessions": sessions, "subjects": len(subjects), "not_first_session": mr2, "by_cdr": {}}
    for k, g in groups.items():
        key = f"{float(k):g}"
        res["by_cdr"][key] = {"n": len(g["age"]), "age_mean": statistics.mean(g["age"]),
                              "age_sd": statistics.stdev(g["age"]) if len(g["age"]) > 1 else 0.0,
                              "mmse_n": len(g["mmse"]), "mmse_mean": statistics.mean(g["mmse"]) if g["mmse"] else None,
                              "female": g["F"], "male": g["M"]}
    return res


def gate_a(rows, records, index_sessions, second):
    """Gate A checks as a list of (name, bool). Counts compared with the documentation and the second path."""
    by = Counter(f"{r['cdr']:g}" for r in rows)
    # sessions that meet the cohort rule apart from the raw-scan requirement, straight from the parsed text files
    elig = [dict(age=rec["age"], cdr=rec["cdr"], mmse=rec.get("mmse"), sex=rec.get("sex")) for s, rec in records.items()
            if int(SESSION_RE.fullmatch(s).group(2)) == 1 and rec.get("age") is not None and rec["age"] >= MIN_AGE
            and rec.get("cdr") is not None]
    checks = [
        ("sessions in the archives = 436", len(records) == DOC_COUNTS["sessions"]),
        ("distinct subjects = 416", len({SESSION_RE.fullmatch(s).group(1) for s in records}) == DOC_COUNTS["subjects"]),
        ("retest (MR2) sessions = 20", sum(int(SESSION_RE.fullmatch(s).group(2)) != 1 for s in records) == DOC_COUNTS["mr2_sessions"]),
        ("every MR2 session is under age 60 (excluded by the age rule)",
         all((records[s].get("age") or 0) < MIN_AGE for s in records if int(SESSION_RE.fullmatch(s).group(2)) != 1)),
        ("cohort by CDR = 98/70/28/2", tuple(by.get(k, 0) for k in ("0", "0.5", "1", "2")) == (98, 70, 28, 2)),
        ("every cohort case has a raw scan, age and sex",
         all(r["scan"] is not None and r["age"] is not None and r["sex"] in ("F", "M") for r in rows)),
        ("every session that meets the age and CDR rule has a raw scan and a sex", len(elig) == len(rows)),
        ("no duplicate case ids", len({r["case_id"] for r in rows}) == len(rows)),
        ("no duplicate sessions in the cohort", len({r["session"] for r in rows}) == len(rows)),
        ("every session text file has an index entry", set(records) == set(index_sessions)),
    ]
    sec_ok = (second["sessions"] == len(records) and second["subjects"] == len({SESSION_RE.fullmatch(s).group(1) for s in records}))
    for g, v in second["by_cdr"].items():
        sel = [r for r in elig if f"{r['cdr']:g}" == g]
        ages = [r["age"] for r in sel]
        mm = [r["mmse"] for r in sel if r["mmse"] is not None]
        sec_ok &= (v["n"] == len(sel) and abs(v["age_mean"] - statistics.mean(ages)) < 1e-9
                   and abs(v["age_sd"] - (statistics.stdev(ages) if len(ages) > 1 else 0.0)) < 1e-9
                   and v["mmse_n"] == len(mm) and (not mm or abs(v["mmse_mean"] - statistics.mean(mm)) < 1e-9)
                   and (v["female"], v["male"]) == (sum(r["sex"] == "F" for r in sel), sum(r["sex"] == "M" for r in sel)))
    sec_ok &= set(second["by_cdr"]) == {f"{r['cdr']:g}" for r in elig}
    checks.append(("second code path (awk) reproduces n, age mean/SD, MMSE mean/n, sex by CDR group and the session counts", bool(sec_ok)))
    return checks


def build_aggregate(rows, exclusions, records, index_sessions, checks, hashes, extra_counts):
    by = Counter(f"{r['cdr']:g}" for r in rows)
    return {
        "stage": "A",
        "archives": extra_counts.get("archives"),
        "archive_members": extra_counts.get("members"),
        "unexpected_archive_members": extra_counts.get("unexpected_members", 0),
        "sessions": len(records),
        "subjects": len({SESSION_RE.fullmatch(s).group(1) for s in records}),
        "retest_sessions": sum(int(SESSION_RE.fullmatch(s).group(2)) != 1 for s in records),
        "retest_max_age": max((records[s].get("age") or 0) for s in records if int(SESSION_RE.fullmatch(s).group(2)) != 1),
        "exclusions": exclusions,
        "cohort": {"n": len(rows), "by_cdr": {k: by.get(k, 0) for k in ("0", "0.5", "1", "2")},
                   "first_scan_is_mpr1": sum(r["scan"] == 1 for r in rows),
                   "mmse_missing": sum(r["mmse"] is None for r in rows),
                   "educ_or_ses_missing": sum(r["educ"] is None or r["ses"] is None for r in rows)},
        "table3": table3_comparison(rows),
        "table2_decades": table2_comparison(rows),
        "txt_vs_documentation_sessions_disagreeing": _agree_txt_with_docs(records),
        "raw_headers_vs_documentation": _agree_headers_with_docs(index_sessions),
        "s4_age_tertiles": s4_age_tertile_edges(rows),
        "manifest_sha256": hashes,
        "gate_a": [{"check": n, "pass": bool(ok)} for n, ok in checks],
    }


# ---------------------------------------------------------------- protocol block
EXCLUSION_ORDER = ("excluded_not_first_session", "excluded_age_missing", "excluded_age_under_60", "excluded_cdr_missing",
                   "excluded_sex_missing", "excluded_no_raw_scan")       # the order of the elif chain in select_cohort
BEGIN = "<!-- STAGE_A:BEGIN (generated by src/validation/oasis_manifest.py from docs/validation/stage_a_counts.json) -->"
END = "<!-- STAGE_A:END -->"


def render_md(res):
    c = res["cohort"]
    t3 = res["table3"]
    L = [BEGIN, "",
         f"Stage A (no inference). Archives: {res['archives']}; sessions {res['sessions']}; subjects {res['subjects']}; "
         f"retest sessions {res['retest_sessions']} (largest age {res['retest_max_age']}, so all are excluded by the age rule; "
         "the paper describes the retest subjects as being in their twenties).", "",
         "Exclusions applied in this order: " + "; ".join(f"{k.replace('excluded_', '').replace('_', ' ')} {res['exclusions'][k]}"
                                                         for k in EXCLUSION_ORDER) + f"; cohort {res['exclusions']['cohort']} of "
         f"{res['exclusions']['sessions']} sessions. Sessions whose first raw scan is not `mpr-1`: {res['exclusions']['first_scan_is_not_mpr1']}.", "",
         f"Cohort n = {c['n']}; by CDR 0 / 0.5 / 1 / 2 = " + " / ".join(str(c["by_cdr"][k]) for k in ("0", "0.5", "1", "2")) +
         f". First raw scan is `mpr-1` in {c['first_scan_is_mpr1']} of {c['n']}. MMSE missing: {c['mmse_missing']}; education or SES missing: {c['educ_or_ses_missing']}.", "",
         "| CDR | n (ours / paper) | female/male (ours / paper) | age mean +/- SD, range (ours) | age (paper) | MMSE mean +/- SD, range (ours) | MMSE (paper) | agree |",
         "|---|---|---|---|---|---|---|---|"]
    for g in ("0", "0.5", "1", "2"):
        r = t3[g]
        p = r["paper"]

        def fmt(v):
            return f"{v[0]:g} +/- {v[1]:g} ({v[2]:g}-{v[3]:g})" if v else "n/a"
        ok = r["agree"]
        L.append(f"| {g} | {r['n']} / {p['n']} | {r['female']}/{r['male']} / {p['female']}/{p['male']} | {fmt(r.get('age'))} | {fmt(p['age'])} | "
                 f"{fmt(r.get('mmse'))} | {fmt(p['mmse'])} | n {'yes' if ok['n'] else 'NO'}, sex {'yes' if ok['sex'] else 'NO'}, "
                 f"age {'yes' if ok['age'] else 'NO'}, MMSE {'yes' if ok['mmse'] else 'NO'} |")
    t2 = res["table2_decades"]
    L += ["", "Check against Table 2 of the paper (decade by dementia status; counts, CDR 0.5/1/2 split and mean age to two decimals): "
          + "; ".join(f"{d} {'agrees' if v['agree'] else 'DISAGREES'}" for d, v in t2.items()) + ".",
          "", "Where Table 3's printed age statistics differ from the recount (the rows marked NO above), the recount agrees with Table 2 and "
          "Table 3's MMSE statistics; the cause is not resolved here. The analysis uses the age in the per-session text file.", ""]
    d = res["txt_vs_documentation_sessions_disagreeing"]
    h = res["raw_headers_vs_documentation"]
    L += ["", "Disagreement with the documented acquisition (sessions, over all sessions in the archives): per-session text "
          + (", ".join(f"{k} {v}" for k, v in d.items()) if d else "none") + ". Raw headers: "
          f"{h['raw_scans']} raw scans; sessions with a disagreeing scan {h['sessions_with_a_disagreeing_scan']}"
          + (" (" + ", ".join(f"{k}: {v} scans" for k, v in h["disagreeing_scans_by_property"].items()) + ")" if h["disagreeing_scans_by_property"] else "")
          + ". Sessions by number of raw scans: " + ", ".join(f"{k} scans: {v}" for k, v in h["sessions_by_scan_count"].items())
          + f"; sessions outside the documented 3 or 4 scans: "
          f"{sum(v for k, v in h['sessions_by_scan_count'].items() if int(k) not in DOC_SCANS_PER_SESSION)}.", ""]
    s4 = res["s4_age_tertiles"]
    L += [f"S4 age-tertile edges (fixed here from the CDR 0 cohort members, n = {s4['n_cdr0']}): e1 = {s4['edges'][0]:g}, e2 = {s4['edges'][1]:g}; "
          "groups are age <= e1, e1 < age <= e2, age > e2 "
          f"(sizes {s4['group_sizes_cdr0'][0]} / {s4['group_sizes_cdr0'][1]} / {s4['group_sizes_cdr0'][2]}).", "",
          "Manifest sha256 as written at Stage A, when the `orientation` column was still `UNSET` (the pinned `inputs.csv` hash after addendum A1 is in section 2): "
          "inputs.csv " + res["manifest_sha256"]["inputs.csv"] + "; labels.csv " + res["manifest_sha256"]["labels.csv"] + ".", "",
          "Gate A: " + "; ".join(f"{'PASS' if g['pass'] else 'FAIL'} {g['check']}" for g in res["gate_a"]) + ".", "", END]
    return "\n".join(L)


def update_protocol(md, protocol_path):
    p = pathlib.Path(protocol_path)
    t = p.read_text()
    a, b = t.index(BEGIN.split(" (")[0]), t.index(END) + len(END)
    new = t[:a] + md + t[b:]
    if new != t:
        p.write_text(new)
    return new != t


# ---------------------------------------------------------------- command line
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="OASIS-1 Stage A manifest (no model). Prints aggregates only.")
    ap.add_argument("--archives", required=True, help="folder with the downloaded tar.gz archives")
    ap.add_argument("--data", required=True, help="data area (outside every git work tree)")
    ap.add_argument("--out-json", required=True, help="aggregate JSON destination (docs/validation/...)")
    ap.add_argument("--protocol", help="protocol file whose Stage A block is regenerated")
    a = ap.parse_args(argv)
    from ..assessment.pathguard import assert_outside_git
    from .aggregate_io import write_json
    os.umask(0o077)
    data = assert_outside_git(a.data, "data area")
    archives = sorted(pathlib.Path(a.archives).glob("oasis_cross-sectional_disc*.tar.gz"))
    if not archives:
        raise SystemExit("no archives found")
    idx_path = data / "private" / "index.json"
    if idx_path.exists():
        index = json.loads(idx_path.read_text())
        print("pass 1: reusing the index in the data area")
    else:
        print(f"pass 1: reading {len(archives)} archives")
        index = scan_archives(archives, data)
        _mkdir(data / "private")
        _write_private(idx_path, json.dumps(index).encode())
    sessions_idx = {k: {**v, "raw": {int(n): r for n, r in v["raw"].items()}} for k, v in index["sessions"].items()}
    records = load_sessions(data)
    rows, exclusions = select_cohort(records, sessions_idx)
    print(f"cohort selected: n={len(rows)}")
    n_files = extract_raw(archives, rows, data)
    print(f"pass 2: extracted {n_files} raw files")
    hashes = write_manifest(rows, data)
    second = recount_second_path(data)
    checks = gate_a(rows, records, sessions_idx, second)
    res = build_aggregate(rows, exclusions, records, sessions_idx, checks, hashes,
                          {"archives": index["archives"], **index["counts"]})
    dest = write_json(a.out_json, res, case_ids=[r["case_id"] for r in rows])
    if a.protocol:
        from .aggregate_io import check_aggregate
        md = render_md(res)
        check_aggregate(md, [r["case_id"] for r in rows])
        update_protocol(md, a.protocol)
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    print(f"aggregate written: {dest.name}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
