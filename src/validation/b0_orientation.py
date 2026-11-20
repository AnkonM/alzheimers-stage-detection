"""Stage B0: evidence for the left-right convention of the OASIS-1 Analyze scans. Rules: Part 1 of
docs/external_validation_addendum_A1.md, committed before any of this was run.

No model, no engine, no worker, no SynthSeg. m1 is the only code that processes an OASIS scan, one scan at a
time. Per-subject values go to the data area; printed and committed output is aggregate only.

    E1      reader behaviour on synthetic Analyze files (no OASIS data)
    E-geo   raw-to-atlas transform and XML/TXT text of the B0 sample (text and 348-byte headers only)
    E3      template chirality test on m1's own QC correlation, with an ADNI positive control
    decide  the pre-registered decision rule
"""
import csv
import json
import os
import pathlib
import re
import statistics
import struct
import sys
import tarfile
import time
from collections import Counter

import numpy as np

from .oasis_manifest import SESSION_RE, _mkdir, _write_private

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONVENTIONS = ("radiological", "neurological")
N_SAMPLE = 20
SYMMETRIC_R = 0.99            # template reflection correlation at or above this: symmetric within tolerance
CONTROL_PASS = 18             # of 20 ADNI volumes must favour the correct orientation
CONV_SUPPORT = 15             # of 20 OASIS scans must favour a convention
INPLANE_MEDIAN_MIN = 0.30     # median over scans of the larger corr_vs_mni
MIN_AVAILABLE_GB = 6.0
SEARCH = 8


class B0Error(RuntimeError):
    """Never contains identifiers or per-subject values."""


# ---------------------------------------------------------------- numpy helpers (E3)
def corr_in_mask(a, t, mask):
    """m1's corr_vs_mni formula: Pearson correlation inside the brain mask (mask > 0.5)."""
    m = np.asarray(mask) > 0.5
    x, y = np.asarray(a)[m], np.asarray(t)[m]
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")


def reflect(vol, axis, s):
    """out[i] = vol[s - i] along `axis` where 0 <= s - i < n, else 0. s = n - 1 is the plain flip."""
    vol = np.asarray(vol)
    n = vol.shape[axis]
    idx = s - np.arange(n)
    ok = ((idx >= 0) & (idx < n)).reshape([n if i == axis else 1 for i in range(vol.ndim)])
    return np.where(ok, np.take(vol, np.clip(idx, 0, n - 1), axis=axis), 0).astype(vol.dtype, copy=False)


def lr_axis(direction):
    """Array axis along which ITK's x (left-right) changes most. direction columns are array axes."""
    d = np.asarray(direction, float).reshape(3, 3)
    return int(np.argmax(np.abs(d[0, :])))


def template_symmetry(t, mask, axis, search=SEARCH):
    """(s*, r_T): the reflection plane maximising in-mask correlation between the template and its reflection."""
    n = np.asarray(t).shape[axis]
    best = (None, -2.0)
    for s in range(n - 1 - search, n + search):
        r = corr_in_mask(reflect(t, axis, s), t, mask)
        if r > best[1]:
            best = (s, r)
    return best


def d_score(vol, t, mask, axis, s):
    """corr(vol, T) - corr(reflected vol, T), both inside the mask. Positive: matches the template's asymmetry."""
    return corr_in_mask(vol, t, mask) - corr_in_mask(reflect(vol, axis, s), t, mask)


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    raise B0Error("MemAvailable not readable")


def require_memory(minimum=MIN_AVAILABLE_GB, probe=mem_available_gb):
    if probe() < minimum:
        raise B0Error(f"less than {minimum:g} GB of memory available: not starting a scan")


# ---------------------------------------------------------------- E1: readers on synthetic Analyze files
def build_synthetic_analyze(stem, centre=(60, 100, 100), half=2, shape=(256, 256, 128)):
    """Big-endian int16 Analyze 7.5 file (orient code 2, 1 x 1 x 1.25 mm) holding one cube of 1000s."""
    stem = pathlib.Path(stem)
    h = bytearray(348)
    h[0:4] = struct.pack(">i", 348)
    h[40:56] = struct.pack(">8h", 3, *shape, 1, 1, 1, 1)
    h[70:74] = struct.pack(">2h", 4, 16)
    h[76:108] = struct.pack(">8f", 0, 1.0, 1.0, 1.25, 0, 0, 0, 0)
    h[252] = 2
    arr = np.zeros(shape, ">i2")
    sl = tuple(slice(c - half, c + half + 1) for c in centre)
    arr[sl] = 1000
    stem.with_suffix(".hdr").write_bytes(bytes(h))
    stem.with_suffix(".img").write_bytes(arr.tobytes(order="F"))
    return stem.with_suffix(".hdr"), centre


def _lps_letters(direction):
    """For each array axis: the ITK (LPS) world direction in which its index increases."""
    d = np.asarray(direction, float).reshape(3, 3)
    out = []
    for i in range(3):
        k = int(np.argmax(np.abs(d[:, i])))
        out.append(("L" if d[k, i] > 0 else "R", "P" if d[k, i] > 0 else "A", "S" if d[k, i] > 0 else "I")[k])
    return out


def _side_lps(img, idx, centre_idx):
    v = np.asarray(img.direction).reshape(3, 3) @ (np.asarray(img.spacing) * (np.asarray(idx) - np.asarray(centre_idx)))
    return "L" if v[0] > 0 else "R"


def _centroid(arr):
    return np.argwhere(arr > 0.5 * arr.max()).mean(axis=0)


def e1_report(workdir):
    """Reader behaviour on one synthetic file. Needs nibabel and ANTsPy (the m1 environment)."""
    import ants
    import nibabel as nib
    from ..assessment.readers.analyze75 import convert_to_nifti
    work = _mkdir(workdir)
    hdr, centre = build_synthetic_analyze(work / "synthetic_scan")
    shape = (256, 256, 128)
    mid = [(n - 1) / 2 for n in shape]
    out = {"synthetic_marker_array_index": list(centre), "array_midline": mid, "readers": {}}

    img = nib.load(str(hdr))
    aff = img.affine
    d = (aff @ np.r_[centre, 1])[:3] - (aff @ np.r_[mid, 1])[:3]
    out["readers"]["nibabel, .hdr read directly"] = {
        "axes_increase_toward": list(nib.aff2axcodes(aff)), "marker_side": "R" if d[0] > 0 else "L"}

    try:
        a = ants.image_read(str(hdr))
        out["readers"]["ITK (ants.image_read), .hdr read directly"] = {
            "axes_increase_toward": _lps_letters(a.direction), "marker_side": _side_lps(a, centre, mid)}
    except Exception as e:                                                       # noqa: BLE001  evidence, not control flow
        out["readers"]["ITK (ants.image_read), .hdr read directly"] = {"status": "reader failed", "error": type(e).__name__}

    after_m1 = {}
    for conv in CONVENTIONS:
        nii = work / f"synthetic_{conv}.nii.gz"
        convert_to_nifti(hdr, nii, orientation=conv)
        nb = nib.load(str(nii))
        d = (nb.affine @ np.r_[centre, 1])[:3] - (nb.affine @ np.r_[mid, 1])[:3]
        out["readers"][f"loader NIfTI ({conv}), nibabel"] = {"axes_increase_toward": list(nib.aff2axcodes(nb.affine)),
                                                              "marker_side": "R" if d[0] > 0 else "L"}
        a = ants.image_read(str(nii))
        out["readers"][f"loader NIfTI ({conv}), ITK"] = {"axes_increase_toward": _lps_letters(a.direction),
                                                          "marker_side": _side_lps(a, centre, mid)}
        # the orientation and resampling steps of m1.preprocess (N4 omitted: irrelevant to geometry)
        b = ants.resample_image(ants.reorient_image2(a, "IAL"), (1.0, 1.0, 1.2), False, 4)
        arr = b.numpy()
        c = _centroid(arr)
        axis = lr_axis(b.direction)
        after_m1[conv] = {"array_axis_left_right": axis, "marker_array_position": [round(float(x), 1) for x in c],
                          "array_shape": list(arr.shape), "marker_side": _side_lps(b, c, [(n - 1) / 2 for n in arr.shape])}
    out["after_m1_orientation_step"] = after_m1
    r, n_ = after_m1["radiological"], after_m1["neurological"]
    ax = r["array_axis_left_right"]
    other = [i for i in range(3) if i != ax]
    out["checks"] = {
        "nibabel and ITK agree on the loader NIfTI (axes and marker side), both conventions":
            all(out["readers"][f"loader NIfTI ({c}), nibabel"] == out["readers"][f"loader NIfTI ({c}), ITK"] for c in CONVENTIONS),
        "the two conventions put the marker on opposite sides after the m1 orientation step": r["marker_side"] != n_["marker_side"],
        "after the m1 step the conventions differ only along the left-right array axis (mirror within 1 voxel)":
            bool(r["array_axis_left_right"] == n_["array_axis_left_right"]
                 and all(abs(r["marker_array_position"][i] - n_["marker_array_position"][i]) <= 1.0 for i in other)
                 and abs(r["marker_array_position"][ax] + n_["marker_array_position"][ax] - (r["array_shape"][ax] - 1)) <= 1.5),
    }
    out["consistent"] = all(out["checks"].values())
    return out


# ---------------------------------------------------------------- E-geo
def parse_t4(text):
    """4x4 matrix from the first four numeric lines of four numbers; every other line is returned as a note."""
    rows, notes = [], []
    for line in text.splitlines():
        parts = line.split()
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            vals = None
        if vals is not None and len(vals) == 4 and len(rows) < 4:
            rows.append(vals)
        elif line.strip():
            notes.append(line.strip())
    if len(rows) != 4:
        raise B0Error("transform file does not contain a 4 x 4 numeric block")
    return np.array(rows, float), notes


def t4_summary(M):
    """Determinant sign and signed-permutation pattern of the 3 x 3 block, read as atlas = M @ raw (assumption)."""
    A = M[:3, :3]
    det = float(np.linalg.det(A))
    names = "xyz"
    pattern = ",".join(f"atlas_{names[i]}<-{'+' if A[i, j] > 0 else '-'}raw_{names[j]}"
                       for i in range(3) for j in [int(np.argmax(np.abs(A[i])))])
    dominance = float(min(np.max(np.abs(A[i])) / max(np.sum(np.abs(A[i])), 1e-12) for i in range(3)))
    return {"det_sign": "+" if det > 0 else "-", "abs_det": abs(det), "pattern": pattern,
            "min_row_dominance": dominance, "last_row_is_0001": bool(np.allclose(M[3], [0, 0, 0, 1]))}


_ORIENT_LINE = re.compile(r"(orient|left|right|radiolog|neurolog|laterali|mirror|flip|handed|frame)", re.I)


def orientation_lines(*texts):
    """Distinct lines mentioning orientation-like words, digits masked, '<value>' for the right-hand side of the
    per-session handedness field."""
    found = Counter()
    for t in texts:
        for line in t.splitlines():
            if _ORIENT_LINE.search(line):
                found[re.sub(r"[0-9]+(\.[0-9]+)?", "#", line.strip())[:120]] += 1
    return found


def extract_geo(archives, sessions, data_dir):
    """For the B0 sample only: copy the session's mpr-1 -> 711-2C transform text and the atlas-image HEADER
    (348 bytes, no payload) into <data>/b0/egeo/<session>/. Returns the number of files copied."""
    want = set(sessions)
    dest = _mkdir(pathlib.Path(data_dir) / "b0" / "egeo")
    n = 0
    for arch in sorted(archives):
        with tarfile.open(arch, "r|gz") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                parts = m.name.split("/")
                if len(parts) < 6 or parts[1] not in want or parts[2] != "PROCESSED":
                    continue
                s = parts[1]
                t4 = len(parts) == 7 and parts[3:6] == ["MPRAGE", "T88_111", "t4_files"] and parts[6] == f"{s}_mpr-1_anon_to_711-2C_t4"
                hd = (len(parts) == 6 and parts[3:5] == ["MPRAGE", "T88_111"]
                      and re.fullmatch(re.escape(s) + r"_mpr_n[0-9]+_anon_111_t88_gfc\.hdr", parts[5]))
                if t4 or hd:
                    _write_private(_mkdir(dest / s) / ("raw_to_711-2C.t4" if t4 else "atlas_gfc.hdr"), tar.extractfile(m).read())
                    n += 1
    return n


def egeo_report(sessions, data_dir):
    from ..assessment.readers.analyze75 import parse_header
    base = pathlib.Path(data_dir)
    dets, patterns, orient, notes = Counter(), Counter(), Counter(), Counter()
    absdet, dom, atlas = [], [], Counter()
    last_ok = 0
    for s in sessions:
        d = base / "b0" / "egeo" / s
        M, note = parse_t4((d / "raw_to_711-2C.t4").read_text())
        sm = t4_summary(M)
        dets[sm["det_sign"]] += 1
        patterns[sm["pattern"]] += 1
        absdet.append(sm["abs_det"])
        dom.append(sm["min_row_dominance"])
        last_ok += sm["last_row_is_0001"]
        notes.update(re.sub(r"[0-9]+(\.[0-9]+)?", "#", x)[:100] for x in note)
        h = parse_header((d / "atlas_gfc.hdr").read_bytes())
        atlas[f"orient code {h['orient']}, dims {'x'.join(map(str, h['dim']))}, voxel {'x'.join(f'{p:g}' for p in h['pixdim'])}"] += 1
        orient.update(orientation_lines((base / "meta" / f"{s}.txt").read_text(encoding="latin-1"),
                                         (base / "meta" / f"{s}.xml").read_text(encoding="latin-1")))
    return {"n_sessions": len(sessions), "determinant_sign_counts": dict(dets),
            "abs_determinant_range": [round(min(absdet), 4), round(max(absdet), 4)],
            "min_row_dominance_over_sessions": round(min(dom), 3),
            "signed_permutation_patterns": dict(patterns), "last_row_0001": last_ok,
            "transform_file_non_numeric_lines_masked": dict(notes.most_common(8)),
            "atlas_header": dict(atlas), "orientation_like_lines_in_xml_txt_masked": dict(orient.most_common(8))}


# ---------------------------------------------------------------- E3
def adni_control_ids(n=N_SAMPLE, root=ROOT):
    """First n PTIDs (ascending) among ADNI train-split subjects with a stored m1 mni volume."""
    import pandas as pd
    cf = pd.read_csv(root / "analysis/fusion_feasibility/alz_crossfit.csv")
    train = set(cf[cf.clin_split == "train"].RID.astype(int))
    rep = pd.read_csv(root / "dataset/mri_prep/m1_report_full.csv")
    rep = rep[rep.RID.astype(int).isin(train) & rep.ok.astype(bool)]
    have = [p for p in sorted(rep.PTID) if (root / "dataset/mri_prep/mni" / f"{p}.nii.gz").exists()]
    if len(have) < n:
        raise B0Error("fewer than the required ADNI control volumes")
    return have[:n]


def load_m1():
    sys.path.insert(0, str(ROOT / "src" / "mri"))
    import m1_preprocess as m1
    return m1


def template_arrays(m1):
    """(template array, mask array, direction) at 96 cubed, exactly as m1 builds them."""
    head_w, mask_w, brain_w, tmpl_o, mask_o = m1.load_template(96)
    return tmpl_o.numpy(), mask_o.numpy(), np.asarray(tmpl_o.direction), (head_w, mask_w, brain_w, tmpl_o, mask_o)


def run_control(ptids, t, mask, axis, s, load_full_mni, out_jsonl):
    """ADNI positive control: D for each stored volume (down-sampled to 96 cubed as m1 does)."""
    res = []
    for p in ptids:
        arr = load_full_mni(p)
        D = d_score(arr, t, mask, axis, s)
        res.append(D)
        _append(out_jsonl, {"id": p, "D": D})
    n_pos = int(sum(d > 0 for d in res))
    return {"n": len(res), "n_favouring_correct_orientation": n_pos, "passes": n_pos >= CONTROL_PASS}


def _append(path, rec):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_e3_oasis(rows, convert, preprocess, t, mask, axis, s, out_jsonl, memory_ok=require_memory):
    """For each (case, convention): convert, run m1 only, record corr and D. `rows` = [(case_id, scan_path)].
    `convert(path, convention) -> nifti path`, `preprocess(nifti) -> (small array, m1 corr_vs_mni)`.
    Resumable: finished (case, convention) pairs in out_jsonl are skipped. Per-scan values stay in out_jsonl."""
    done = set()
    if pathlib.Path(out_jsonl).exists():
        for line in open(out_jsonl):
            r = json.loads(line)
            done.add((r["case"], r["convention"]))
    for case, path in rows:
        for conv in CONVENTIONS:
            if (case, conv) in done:
                continue
            memory_ok()
            t0 = time.time()
            small, corr_m1 = preprocess(convert(path, conv))
            small = np.asarray(small)
            c, cr = corr_in_mask(small, t, mask), corr_in_mask(reflect(small, axis, s), t, mask)
            if not abs(c - corr_m1) < 1e-6:
                raise B0Error("own correlation differs from m1's corr_vs_mni")
            _append(out_jsonl, {"case": case, "convention": conv, "corr": c, "corr_reflected": cr, "D": c - cr,
                                "seconds": round(time.time() - t0, 1)})


def e3_summary(out_jsonl):
    per = {}
    for line in open(out_jsonl):
        r = json.loads(line)
        per.setdefault(r["case"], {})[r["convention"]] = r
    full = {c: v for c, v in per.items() if set(v) == set(CONVENTIONS)}
    wins = Counter()
    best_corr = []
    for c, v in full.items():
        a, b = v["radiological"], v["neurological"]
        wins["radiological" if a["D"] > b["D"] else "neurological" if b["D"] > a["D"] else "tie"] += 1
        best_corr.append(max(a["corr"], b["corr"]))
    n = len(full)
    med = float(statistics.median(best_corr)) if best_corr else float("nan")
    supported = [c for c in CONVENTIONS if wins[c] >= CONV_SUPPORT]
    return {"n_scans": n, "wins": {c: wins[c] for c in (*CONVENTIONS, "tie")}, "supported": supported,
            "median_of_larger_corr": round(med, 3), "in_plane_sanity_ok": bool(n and med >= INPLANE_MEDIAN_MIN),
            "corr_pass_scans": int(sum(x >= 0.30 for x in best_corr))}


# ---------------------------------------------------------------- decision rule (1.6)
def decide(e1_consistent, in_plane_ok, e3_informative, e3_supported, egeo_convention, egeo_documented_basis):
    """Returns (outcome, convention, reason). outcome is "propose" or "stop"; convention is None when stopping.
    e3_supported: None, or the convention E3 supports (None if neither reached the limit); egeo_convention: None
    or the convention E-geo gives; egeo_documented_basis: whether that convention rests on a documented frame."""
    egeo = egeo_convention if egeo_documented_basis else None
    if not e1_consistent:
        return "stop", None, "E1 is not consistent: pipeline defect"
    if e3_informative and not in_plane_ok:
        return "stop", None, "in-plane sanity failed: the assumed in-plane axes are refuted"
    if not e3_informative and egeo is None:
        return "stop", None, "E3 is uninformative and E-geo gives no absolute convention"
    if e3_informative and egeo is not None and e3_supported is not None and e3_supported != egeo:
        return "stop", None, "E3 and E-geo are both informative and disagree"
    if e3_informative and e3_supported is None:
        return "stop", None, "E3 supports neither convention"
    if e3_informative:
        return "propose", e3_supported, "E3 supports it" + (" and E-geo agrees" if egeo == e3_supported else "; E-geo gives none")
    return "propose", egeo, "E-geo gives it with a documented basis; E3 did not corroborate it (uninformative)"


def scrub_notes(counts):
    """Drop user names (CVS revision lines) and home-directory paths from the quoted transform-file text; merge counts."""
    out = Counter()
    for k, v in counts.items():
        k = re.sub(r"\$Id:.*\$", "$Id: <revision line> $", k)
        k = re.sub(r"/home/\S*?/", "<path>/", k)
        out[k] += v
    return dict(out.most_common())


# ---------------------------------------------------------------- addendum Part 2 (generated)
A1_BEGIN = "<!-- B0_RESULTS:BEGIN -->"
A1_END = "<!-- B0_RESULTS:END -->"


def _items(d):
    """Deterministic order for rendering: by count (descending), then by name."""
    return sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))


def render_part2(res, run_commit):
    e1, eg, e3, dec = res["e1"], res["e_geo"], res["e3"], res["decision"]
    tc = e3["template_and_control"]
    L = [A1_BEGIN, "", f"Run of 2026-11-14 with the Stage B0 code at commit `{run_commit}`. Generated from `docs/validation/b0_summary.json`; "
         "per-subject values are in the data area only.", "",
         "### E1: readers on a synthetic Analyze file (no OASIS data)", "",
         "| Reader | Array axes increase toward | Marker side |", "|---|---|---|"]
    for k, v in sorted(e1["readers"].items()):
        L.append(f"| {k} | {', '.join(v['axes_increase_toward']) if 'axes_increase_toward' in v else v.get('status', 'n/a')} | {v.get('marker_side', 'n/a')} |")
    a = e1["after_m1_orientation_step"]
    L += ["", "After the orientation and resampling steps of m1 (marker array position on the left-right axis, array length "
          f"{a['radiological']['array_shape'][a['radiological']['array_axis_left_right']]}): "
          + "; ".join(f"{c} {a[c]['marker_array_position'][a[c]['array_axis_left_right']]:g} ({a[c]['marker_side']})" for c in CONVENTIONS) + ".", "",
          *[f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in e1["checks"].items()], "",
          "m1 relies on ITK's reading (`ants.image_read`) of the file it is given, followed by header-driven reorientation. Read directly, the two "
          "readers disagree with each other about the front-back axis and both put the first array axis on left-right, which is not the layout of a "
          "sagittal scan; this is why the explicit loader exists. E1 shows that the loader's NIfTI is read identically by both readers under each "
          "convention and that the two conventions give mirror images after m1's orientation step. **E1 does not say which convention is right.**", "",
          "### E-geo: raw-to-atlas transform and XML/TXT text (20 sessions)", "",
          f"- Determinant sign of the 3 x 3 block: {', '.join(f'{k} in {v}' for k, v in eg['determinant_sign_counts'].items())} sessions; "
          f"absolute determinant {eg['abs_determinant_range'][0]:g} to {eg['abs_determinant_range'][1]:g}.",
          f"- Signed-permutation pattern of the block (read as atlas = M x raw, an assumption): "
          + "; ".join(f"`{k}` in {v} sessions" for k, v in _items(eg["signed_permutation_patterns"])) + ".",
          "- Atlas header: " + "; ".join(f"{k} ({v} sessions)" for k, v in _items(eg["atlas_header"])) + ".",
          "- Text in the transform files other than numbers (digits masked): " + "; ".join(f"`{k}` ({v})" for k, v in _items(eg["transform_file_non_numeric_lines_masked"])) + ".",
          "- Orientation-like lines in the XML and TXT (digits masked): " + "; ".join(f"`{k}` ({v})" for k, v in _items(eg["orientation_like_lines_in_xml_txt_masked"]))
          + ". The `flip` fields are the flip angle, not a left-right flip.", "",
          "Derivable: the raw-to-atlas transform preserves handedness (positive determinant) in every session. Not derivable: the pattern shows the "
          "transform's raw frame is ordered differently from the Analyze voxel axes, and the relation between that frame and the voxel axes, and the "
          "left-right sense of the atlas x axis in these files, are not stated in the transform text, the XML, the TXT or the documents supplied. "
          f"Stated outcome: **E-geo gives no absolute convention** (convention read from it: {eg['convention_read_from_it']}; documented basis: {eg['documented_basis']}).", "",
          "### E3: template chirality test", "",
          f"- Template: the ICBM152 2009 files m1 loads through nilearn are named `nlin_sym_09a` (symmetric). Pre-registered numeric check: in-mask correlation of "
          f"the template with its reflection r_T = {tc['r_T']:.4f} at plane {tc['plane_s']} (centre plane {tc['plane_s_centre']}), limit {SYMMETRIC_R:g}. "
          + ("This is below the limit, so by the numeric rule alone the template counts as asymmetric. The files are symmetric by construction, so the limit was "
             "a poor discriminator at 96 voxels; it was not adjusted afterwards. The ADNI control below decides in any case." if not tc["symmetric"]
             else "This is at or above the limit: symmetric."),
          f"- ADNI positive control (first {tc['control']['n']} train-split PTIDs with a stored volume): {tc['control']['n_favouring_correct_orientation']} of "
          f"{tc['control']['n']} volumes match their stored orientation better than the mirror image (pass needs {CONTROL_PASS}). "
          + ("This is below chance, in the wrong direction." if tc["control"]["n_favouring_correct_orientation"] < tc["control"]["n"] / 2 else ""),
          f"- E3 informative: **{'yes' if tc['e3_informative'] else 'no'}**. "
          + ("It was therefore not run on OASIS data: no OASIS scan was processed by m1 in Stage B0." if e3["oasis_scans"] == "not run" else ""), "",
          "### Decision (rule 1.6)", "",
          f"**Outcome: {dec['outcome'].upper()}. Proposed convention: {dec['proposed_convention'] or 'none'}.** Reason: {dec['reason']}. "
          "No convention is picked, and the loader, the manifest (`orientation` is `UNSET`) and the protocol body are unchanged.", "",
          "Not tested at all in Stage B0: the in-plane axis directions assumed by the loader (A, S). The in-plane sanity rule is attached to E3, which did not run. "
          "A registration-quality check of the in-plane axes does not depend on the left-right sign and can be done on a few scans in either convention, "
          "but only under a new dated addendum.", "", A1_END]
    return "\n".join(L)


def update_addendum(md, path):
    p = pathlib.Path(path)
    t = p.read_text()
    a, b = t.index(A1_BEGIN), t.index(A1_END) + len(A1_END)
    new = t[:a] + md + t[b:]
    if new != t:
        p.write_text(new)
    return new != t


# ---------------------------------------------------------------- command line
def sample_cases(data, n=N_SAMPLE):
    """The B0 sample: the first n case ids in sorted order, as (case_id, session_id, scan_path)."""
    data = pathlib.Path(data)
    with open(data / "private" / "mapping.csv", newline="") as f:
        sess = {r["case_id"]: r["session_id"] for r in csv.DictReader(f)}
    with open(data / "manifest" / "inputs.csv", newline="") as f:
        paths = {r["case_id"]: r["scan_path"] for r in csv.DictReader(f)}
    ids = sorted(paths)[:n]
    return [(c, sess[c], paths[c]) for c in ids]


def _jwrite(path, obj):
    _write_private(path, (json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n").encode())


def _load_full_mni_factory(m1, axis):
    import ants

    def load(ptid):
        img = ants.image_read(str(ROOT / "dataset/mri_prep/mni" / f"{ptid}.nii.gz"))
        if lr_axis(img.direction) != axis:
            raise B0Error("a stored ADNI volume has a different left-right array axis than the template")
        return ants.resample_image(img, (96,) * 3, True, 0).numpy()
    return load


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Stage B0 (no model). Prints aggregates only.")
    ap.add_argument("cmd", choices=["e1", "egeo", "template", "oasis", "report"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--archives")
    ap.add_argument("--out-json", help="report: aggregate destination under docs/validation/")
    ap.add_argument("--addendum", help="report: addendum file whose Part 2 block is regenerated")
    ap.add_argument("--run-commit", help="report: git commit of the code that ran")
    ap.add_argument("--egeo-convention", choices=["none", *CONVENTIONS], default="none")
    ap.add_argument("--egeo-documented-basis", choices=["yes", "no"], default="no")
    a = ap.parse_args(argv)
    from ..assessment.pathguard import assert_outside_git
    os.umask(0o077)
    data = assert_outside_git(a.data, "data area")
    b0 = _mkdir(data / "b0")

    if a.cmd == "e1":
        res = e1_report(b0 / "e1_work")
        _jwrite(b0 / "e1.json", res)
        for k, v in res["readers"].items():
            print(k, v)
        print("after m1 orientation step:", json.dumps(res["after_m1_orientation_step"]))
        for k, v in res["checks"].items():
            print(("PASS  " if v else "FAIL  ") + k)
        print("E1 consistent:", res["consistent"])
    elif a.cmd == "egeo":
        cases = sample_cases(data)
        n = extract_geo(sorted(pathlib.Path(a.archives).glob("oasis_cross-sectional_disc*.tar.gz")), [s for _, s, _ in cases], data)
        res = egeo_report([s for _, s, _ in cases], data)
        _jwrite(b0 / "egeo.json", res)
        print("files copied:", n)
        print(json.dumps(res, indent=1))
    elif a.cmd == "template":
        m1 = load_m1()
        t, mask, direction, _ = template_arrays(m1)
        axis = lr_axis(direction)
        s, r_T = template_symmetry(t, mask, axis)
        res = {"lr_axis": axis, "plane_s": s, "plane_s_centre": t.shape[axis] - 1, "r_T": r_T,
               "symmetric": bool(r_T >= SYMMETRIC_R), "template_files": sorted(p.name for p in (pathlib.Path.home() / "nilearn_data/icbm152_2009").glob("*/*t1_tal*.nii.gz"))}
        ids = adni_control_ids()
        control = run_control(ids, t, mask, axis, s, _load_full_mni_factory(m1, axis), b0 / "control_per_volume.jsonl")
        res["control"] = control
        res["e3_informative"] = bool((not res["symmetric"]) and control["passes"])
        _jwrite(b0 / "template_control.json", res)
        print(json.dumps(res, indent=1))
    elif a.cmd == "oasis":
        tc = json.loads((b0 / "template_control.json").read_text())
        if not tc["e3_informative"]:
            raise SystemExit("E3 is not informative (template symmetric or control failed): not running it on OASIS data")
        from ..assessment.readers.analyze75 import convert_to_nifti
        m1 = load_m1()
        t, mask, direction, (head_w, mask_w, brain_w, tmpl_o, mask_o) = template_arrays(m1)
        work = _mkdir(b0 / "nifti")

        def convert(path, conv):
            out = work / f"{pathlib.Path(path).stem}_{conv}.nii.gz"
            convert_to_nifti(path, out, orientation=conv)
            return str(out)

        def preprocess(nii):
            full, small = m1.preprocess(nii, head_w, mask_w, brain_w, 96)
            return small.numpy(), m1.qc_metrics(small, tmpl_o, mask_o)["corr_vs_mni"]

        rows = [(c, p) for c, _, p in sample_cases(data)]
        run_e3_oasis(rows, convert, preprocess, t, mask, tc["lr_axis"], tc["plane_s"], b0 / "e3_per_scan.jsonl")
        res = e3_summary(b0 / "e3_per_scan.jsonl")
        _jwrite(b0 / "e3_summary.json", res)
        print(json.dumps(res, indent=1))
    else:
        from .aggregate_io import write_json
        e1 = json.loads((b0 / "e1.json").read_text())
        eg = json.loads((b0 / "egeo.json").read_text())
        tc = json.loads((b0 / "template_control.json").read_text())
        e3 = json.loads((b0 / "e3_summary.json").read_text()) if (b0 / "e3_summary.json").exists() else None
        supported = None
        if e3 and len(e3["supported"]) == 1:
            supported = e3["supported"][0]
        egeo_conv = None if a.egeo_convention == "none" else a.egeo_convention
        out, conv, why = decide(e1["consistent"], bool(e3 and e3["in_plane_sanity_ok"]) if e3 else True, tc["e3_informative"], supported,
                                egeo_conv, a.egeo_documented_basis == "yes")
        res = {"stage": "B0", "e1": {"checks": e1["checks"], "consistent": e1["consistent"], "readers": e1["readers"],
                                      "after_m1_orientation_step": e1["after_m1_orientation_step"]},
               "e_geo": {**eg, "transform_file_non_numeric_lines_masked": scrub_notes(eg.get("transform_file_non_numeric_lines_masked", {})),
                         "convention_read_from_it": a.egeo_convention, "documented_basis": a.egeo_documented_basis},
               "e3": {"template_and_control": {k: v for k, v in tc.items() if k != "template_files"},
                      "oasis_scans": e3 if e3 else "not run"},
               "decision": {"outcome": out, "proposed_convention": conv, "reason": why}}
        dest = write_json(a.out_json, res)
        if a.addendum:
            from .aggregate_io import check_aggregate
            md = render_part2({**res, "e_geo": res["e_geo"]}, a.run_commit or "unrecorded")
            check_aggregate(md)
            update_addendum(md, a.addendum)
        print(f"decision: {out}; proposed convention: {conv}; reason: {why}; written: {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
