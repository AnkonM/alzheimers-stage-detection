"""Temporal-scan evaluation, steps 1-2 (run with LYP/.venv-fusion): inventory, ground truth, per-visit records.

PLAN.md sections 1-3. No model is loaded and no prediction is computed here.
Output (gitignored): private/scans.csv, one row per grid row of visits_index.csv,
with group, usable flag / failure reason, matched DX visit (+-90 and +-180 days),
true stage (4-class and 3-class), the clinical record of the same visit and the
FreeSurfer volumes. Aggregate counts -> results/inventory.md (committed).
--check-builder: the G1c builder check on the 814 cohort subjects' baseline.
"""
import argparse
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
import step0_join_tables as S0  # noqa: E402

RAW = os.path.join(ROOT, "dataset", "raw")
TAB = os.path.join(RAW, "adnimerge2_tables")
PROCESSED = os.path.join(ROOT, "dataset", "processed")
PRIVATE = os.path.join(HERE, "private")
RESULTS = os.path.join(HERE, "results")
GROUPS = ["TEST", "VAL", "TRAIN", "NOT_IN_COHORT"]
NAMES = ["CN", "EMCI", "LMCI", "AD"]
LABEL_MAP = {"CN": 0, "EMCI": 1, "LMCI": 2, "AD": 3}
BASELINE_CODES = ("sc", "scmri", "bl")
VOLUMES = ["Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV"]
ECOG = [f"Ecog{s}{d}" for s in ("Pt", "SP") for d in ("Mem", "Lang", "Visspat", "Plan", "Organ", "Divatt", "Total")]
VISIT_FEATURES = ["CDRSB", "ADAS11", "ADAS13", "MMSE", "RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting",
                  "RAVLT_perc_forgetting", "FAQ", "MOCA"] + ECOG
# per-visit tables: (tag, loader, {feature: column}); baseline codes in preference order (step 0 tiers)
BL_CODES = {"CDR": ("bl", "sc"), "ADAS": ("bl",), "ECOGPT": ("bl",), "ECOGSP": ("bl",), "MOCA": ("bl",),
            "MMSE": ("bl", "sc"), "FAQ": ("bl", "sc"), "NEUROBAT": ("bl", "sc")}


def check_manifest():
    r = subprocess.run(["sha256sum", "-c", "--quiet", "analysis/fusion_v3/frozen/MANIFEST.sha256"], cwd=ROOT,
                       capture_output=True, text=True)
    n = sum(1 for _ in open(os.path.join(ROOT, "analysis/fusion_v3/frozen/MANIFEST.sha256")))
    assert r.returncode == 0 and n == 109, f"frozen checksum failure:\n{r.stdout}{r.stderr}"
    return n


def groups():
    g = {}
    for name, f in (("TRAIN", "train"), ("VAL", "val"), ("TEST", "test")):
        for rid in pd.read_csv(os.path.join(PROCESSED, f"{f}_rids.csv")).RID:
            assert rid not in g, f"RID {rid} in two splits"
            g[rid] = name
    return g


# ---------------------------------------------------------------- per-visit sources
def _num(s):
    return pd.to_numeric(s, errors="coerce")


def visit_tables():
    """{tag: frame with RID (int), VISCODE2, _date and feature columns}, one frame per source table."""
    out = {}
    cdr = S0.load_domain_table(os.path.join(RAW, "CDR.csv"))
    out["CDR"] = cdr.assign(CDRSB=_num(cdr.CDRSB), _date=cdr.VISDATE)[["RID", "VISCODE2", "_date", "CDRSB"]]
    ad = S0.load_domain_table(os.path.join(RAW, "ADAS_ADNIGO2.csv"))
    out["ADAS"] = ad.assign(ADAS11=_num(ad.TOTSCORE), ADAS13=_num(ad.TOTAL13), _date=ad.VISDATE)[
        ["RID", "VISCODE2", "_date", "ADAS11", "ADAS13"]]
    mm = S0.load_domain_table(os.path.join(TAB, "MMSE.csv"))
    out["MMSE"] = mm.assign(MMSE=_num(mm.MMSCORE), _date=mm.VISDATE)[["RID", "VISCODE2", "_date", "MMSE"]]
    fq = S0.load_domain_table(os.path.join(TAB, "FAQ.csv"))
    out["FAQ"] = fq.assign(FAQ=_num(fq.FAQTOTAL), _date=fq.VISDATE)[["RID", "VISCODE2", "_date", "FAQ"]]
    nb = S0.load_domain_table(os.path.join(TAB, "NEUROBAT.csv"))
    t = {k: _num(nb[f"AVTOT{k}"]) for k in range(1, 6)}
    d30 = _num(nb.AVDEL30MIN)
    imm = sum(t.values())
    fg = t[5] - d30
    out["NEUROBAT"] = pd.DataFrame({"RID": nb.RID, "VISCODE2": nb.VISCODE2, "_date": nb.VISDATE,
                                    "RAVLT_immediate": imm, "RAVLT_learning": t[5] - t[1],
                                    "RAVLT_forgetting": fg,
                                    "RAVLT_perc_forgetting": (100 * fg / t[5]).where(t[5] > 0)})
    mo = S0.load_moca_table(os.path.join(TAB, "MOCA.csv"))
    out["MOCA"] = mo.assign(_date=mo.VISDATE)[["RID", "VISCODE2", "_date", "MOCA"]]
    for tag, pre in (("ECOGPT", "EcogPt"), ("ECOGSP", "EcogSP")):
        e = S0.load_ecog_table(os.path.join(TAB, f"{tag}.csv"), pre)
        cols = [c for c in ECOG if c.startswith(pre)]
        out[tag] = e.assign(_date=e.VISDATE)[["RID", "VISCODE2", "_date"] + cols]
    for f in out.values():
        f["RID"] = f.RID.astype(int)
        f["_date"] = pd.to_datetime(f["_date"], errors="coerce")
    return out


def pick(frame, rid, viscode2, baseline):
    """The row of `rid` at this visit only. Baseline: step-0 tiers over the table's baseline codes, earliest date
    within a tier, no earliest-visit fallback. Follow-up: same VISCODE2, earliest date."""
    r = frame[frame.RID == rid]
    if baseline:
        for code in baseline:
            hit = r[r.VISCODE2 == code]
            if len(hit):
                return hit.sort_values("_date", na_position="last").iloc[0]
        return None
    hit = r[r.VISCODE2 == viscode2]
    return hit.sort_values("_date", na_position="last").iloc[0] if len(hit) else None


def record(tables, rid, viscode2, baseline):
    """Clinical features of one visit (NaN where that visit's table has no row)."""
    rec = {f: np.nan for f in VISIT_FEATURES}
    for tag, frame in tables.items():
        row = pick(frame, rid, viscode2, BL_CODES[tag] if baseline else None)
        if row is not None:
            for c in frame.columns:
                if c in rec:
                    rec[c] = row[c]
    return rec


def freesurfer():
    fs = S0.load_mri_volumetrics(os.path.join(TAB, "UCSFFSX51.csv"))
    fs["RID"] = fs.RID.astype(int)
    fs["_date"] = pd.to_datetime(fs.EXAMDATE, errors="coerce")
    iu = pd.read_csv(os.path.join(TAB, "UCSFFSX51.csv"), usecols=["IMAGEUID"]).IMAGEUID
    fs["IMAGEUID"] = pd.to_numeric(iu, errors="coerce")
    return fs


def fs_row(fs, rid, image_id, viscode2):
    r = fs[fs.RID == rid]
    hit = r[r.IMAGEUID == int(str(image_id).lstrip("I"))] if isinstance(image_id, str) else r.iloc[:0]
    if len(hit):
        return hit.iloc[0], "IMAGEUID"
    hit = r[r.VISCODE2 == viscode2].sort_values("_date", na_position="last")
    if len(hit):
        return hit.iloc[0], "VISCODE2"
    return None, "UNAVAILABLE"


# ---------------------------------------------------------------- ground truth
def dx_tables():
    dx = pd.read_csv(os.path.join(TAB, "DXSUM.csv"), usecols=["RID", "VISCODE2", "EXAMDATE", "DIAGNOSIS"])
    dx["EXAMDATE"] = pd.to_datetime(dx.EXAMDATE, errors="coerce")
    dx = dx.dropna(subset=["EXAMDATE", "DIAGNOSIS"])
    bl = S0.build_dx_bl(RAW)                                   # main-analysis baseline label (arm + DXSUM)
    bl["RID"] = bl.RID.astype(int)
    return dx, bl.set_index("RID").DX_bl


def match_dx(dx, rid, date, window):
    r = dx[dx.RID == rid]
    if not len(r) or pd.isna(date):
        return None
    d = (r.EXAMDATE - date).dt.days
    r = r.assign(_ad=d.abs(), _d=d).sort_values(["_ad", "EXAMDATE"])
    best = r.iloc[0]
    return best if best._ad <= window else None


def truth(best, dx_bl_label):
    """(four-stage label or None, reason, three-class label or None)."""
    if best is None:
        return None, "NO_DX_MATCH", None
    three = {"CN": 0, "MCI": 1, "Dementia": 2}[best.DIAGNOSIS]
    if best.VISCODE2 in BASELINE_CODES:
        lab = LABEL_MAP.get(dx_bl_label)
        return lab, ("OK" if lab is not None else f"UNDEFINED_BASELINE_{dx_bl_label}"), three
    if best.DIAGNOSIS == "MCI":
        return None, "UNDEFINED_PLAIN_MCI", three
    return {"CN": 0, "Dementia": 3}[best.DIAGNOSIS], "OK", three


# ---------------------------------------------------------------- main
def build():
    nman = check_manifest()
    grp = groups()
    v = pd.read_csv(os.path.join(ROOT, "dataset", "mri_prep_patha", "visits_index.csv"))
    v["group"] = v.RID.map(grp).fillna("NOT_IN_COHORT")
    assert v.groupby("RID").group.nunique().max() == 1
    v["usable"] = v.present.astype(bool) & (v.prep_ok == True)            # noqa: E712
    v["fail_reason"] = np.where(v.usable, "", np.where(v.present & ~v.usable.astype(bool), "PREP_FAILED",
                                                       v.missing_reason.fillna("NOT_PRESENT")))
    v.loc[~v.usable & (v.missing_reason != "OK"), "fail_reason"] = v.missing_reason
    v["timepoint"] = np.where(v.slot == "bl", "baseline", "follow-up")
    v["scan_date"] = pd.to_datetime(v.scan_date, errors="coerce")

    dx, dx_bl = dx_tables()
    adsl = pd.read_csv(os.path.join(TAB, "ADSL.csv"), usecols=["SUBJID", "AGE", "ENRLDT", "SEX", "EDUC", "APOE"])
    adsl["RID"] = adsl.SUBJID.astype(str).str.split("_S_").str[-1].astype(int)
    adsl = adsl.set_index("RID")
    step2 = pd.read_csv(os.path.join(PROCESSED, "step2_features.csv"), low_memory=False).set_index("RID")
    tables, fs = visit_tables(), freesurfer()

    rows = []
    for r in v.itertuples():
        out = {"RID": r.RID, "PTID": r.PTID, "slot": r.slot, "group": r.group, "timepoint": r.timepoint,
               "usable": bool(r.usable), "fail_reason": r.fail_reason, "scan_date": r.scan_date,
               "image_id": r.image_id, "viscode2_scan": r.viscode2, "prep_key": r.prep_key,
               "corr_vs_mni": r.corr_vs_mni}
        for w in (90, 180):
            b = match_dx(dx, r.RID, r.scan_date, w)
            lab, why, three = truth(b, dx_bl.get(r.RID))
            out[f"y_{w}"], out[f"truth_{w}"], out[f"y3_{w}"] = lab, why, three
            out[f"dx_visit_{w}"] = None if b is None else b.VISCODE2
            out[f"dx_days_{w}"] = None if b is None else int(b._d)
        out["dx_raw_90"] = None
        b90 = match_dx(dx, r.RID, r.scan_date, 90)
        if b90 is not None:
            out["dx_raw_90"] = b90.DIAGNOSIS
        # clinical record of the same visit (+-90 match defines the visit; the +-180 run re-uses its own visit)
        for w in (90, 180):
            vis = out[f"dx_visit_{w}"]
            sfx = "" if w == 90 else "_180"
            if vis is None:
                out[f"clin_source{sfx}"] = "NO_VISIT"
                continue
            baseline = vis in BASELINE_CODES
            if baseline and r.RID in step2.index:
                rec = step2.loc[r.RID, VISIT_FEATURES].to_dict()
                out[f"clin_source{sfx}"] = "step2"
            else:
                rec = record(tables, r.RID, vis, baseline)
                out[f"clin_source{sfx}"] = "builder"
            if w == 90:
                out.update(rec)
            else:
                out.update({f"{k}_180": val for k, val in rec.items()})
        # subject-level constants and age at the visit
        a = adsl.loc[r.RID] if r.RID in adsl.index else None
        out["PTGENDER"] = None if a is None else a.SEX
        out["PTEDUCAT"] = None if a is None else a.EDUC
        out["APOE4"] = None if a is None else S0.derive_apoe4(a.APOE)
        out["age_at_scan"] = np.nan if a is None else a.AGE + (r.scan_date - pd.to_datetime(a.ENRLDT)).days / 365.25
        bl_cohort = r.slot == "bl" and r.RID in step2.index
        out["AGE"] = step2.loc[r.RID, "AGE"] if bl_cohort else out["age_at_scan"]
        if bl_cohort:
            out.update(step2.loc[r.RID, VOLUMES].to_dict())
            out["fs_source"] = "step2" if step2.loc[r.RID, VOLUMES].notna().all() else "step2_missing"
        else:
            row, src = fs_row(fs, r.RID, r.image_id, r.viscode2)
            out.update({c: (np.nan if row is None else row[c]) for c in VOLUMES})
            out["fs_source"] = src
        rows.append(out)
    s = pd.DataFrame(rows)
    for sfx in ("", "_180"):
        s[f"has_clin{sfx}"] = s[f"CDRSB{sfx}"].notna() & (s[f"MMSE{sfx}"].notna() | s[f"ADAS13{sfx}"].notna())
    s["fs_available"] = s[VOLUMES].notna().all(axis=1)
    os.makedirs(PRIVATE, exist_ok=True)
    s.to_csv(os.path.join(PRIVATE, "scans.csv"), index=False)
    return s, nman, step2


def check_builder():
    """G1c: builder at the baseline visit vs step2_features for the 814 cohort subjects."""
    step2 = pd.read_csv(os.path.join(PROCESSED, "step2_features.csv"), low_memory=False)
    tables = visit_tables()
    rows = [dict(RID=rid, **record(tables, rid, None, True)) for rid in step2.RID]
    b = pd.DataFrame(rows).set_index("RID")
    s2 = step2.set_index("RID")
    res = {}
    for f in VISIT_FEATURES:
        x, y = b[f].astype(float), s2[f].astype(float)
        agree = ((x - y).abs() <= 1e-6) | (x.isna() & y.isna())
        res[f] = (float(agree.mean()), int((~agree).sum()), int((x.isna() & y.notna()).sum()),
                  int((x.notna() & y.isna()).sum()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-builder", action="store_true")
    a = ap.parse_args()
    if a.check_builder:
        for f, (rate, n_bad, miss_b, miss_s) in check_builder().items():
            print(f"{f:22s} agree {rate:.4f}  differ {n_bad:3d}  (builder NaN only {miss_b}, step2 NaN only {miss_s})")
        return
    write_inventory(build()[0])



def write_inventory(s):
    """Aggregate counts only (committed): PLAN.md step 1 and the exclusion steps known before inference."""
    u = s[s.usable]
    L = ["# Inventory (step 1) and ground truth (step 2) — counts only, no predictions\n",
         f"Frozen checksum list: {check_manifest()} / 109 files OK.\n",
         "The frozen final system was fitted on **TRAIN + VAL** (691 subjects; `freeze.md`), so TRAIN and VAL "
         "scans are IN-SAMPLE. Out-of-sample groups: TEST and NOT_IN_COHORT.\n",
         f"- Grid rows (subject x slot): {len(s)}; usable scans: **{len(u)}** from **{u.RID.nunique()}** subjects",
         f"- Baseline scans: {int((u.timepoint == 'baseline').sum())}; follow-up: {int((u.timepoint == 'follow-up').sum())}",
         f"- Usable visits per subject: {u.groupby('RID').size().value_counts().sort_index().to_dict()}",
         f"- Usable scans with a clinical record at the same visit (+-90 d visit): {int(u.has_clin.sum())}\n",
         "## Scans and subjects per group\n", "| Group | in-sample | scans | subjects | baseline | follow-up |",
         "|---|---|---|---|---|---|"]
    for g in GROUPS:
        x = u[u.group == g]
        L.append(f"| {g} | {'IN-SAMPLE' if g in ('TRAIN', 'VAL') else 'out-of-sample'} | {len(x)} | "
                 f"{x.RID.nunique()} | {int((x.timepoint == 'baseline').sum())} | {int((x.timepoint == 'follow-up').sum())} |")
    L += ["\n## Unusable grid rows (preprocessing / ingestion failures, reported not dropped)\n",
          "| Reason | rows |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in s[~s.usable].fail_reason.value_counts().items()]
    for w in (90, 180):
        L += [f"\n## Ground truth, +-{w} days (usable scans)\n", "| Group | timepoint | " +
              " | ".join(["OK", "NO_DX_MATCH", "UNDEFINED_PLAIN_MCI", "UNDEFINED_BASELINE_*"]) + " | CN | EMCI | LMCI | AD |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for g in GROUPS:
            for t in ("baseline", "follow-up"):
                x = u[(u.group == g) & (u.timepoint == t)]
                if not len(x):
                    continue
                tr = x[f"truth_{w}"]
                yc = np.bincount(x[f"y_{w}"].dropna().astype(int), minlength=4)
                L.append(f"| {g} | {t} | {int((tr == 'OK').sum())} | {int((tr == 'NO_DX_MATCH').sum())} | "
                         f"{int((tr == 'UNDEFINED_PLAIN_MCI').sum())} | {int(tr.str.startswith('UNDEFINED_BASELINE').sum())} | "
                         + " | ".join(str(c) for c in yc) + " |")
    L += ["\n## Clinical record and FreeSurfer source (usable scans, +-90 d)\n",
          "| Group | timepoint | clinical record | no clinical record | FS step2 | FS IMAGEUID | FS VISCODE2 | FS unavailable |",
          "|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        for t in ("baseline", "follow-up"):
            x = u[(u.group == g) & (u.timepoint == t)]
            if not len(x):
                continue
            fsrc = x.fs_source.value_counts()
            L.append(f"| {g} | {t} | {int(x.has_clin.sum())} | {int((~x.has_clin).sum())} | "
                     f"{int(fsrc.get('step2', 0))} | {int(fsrc.get('IMAGEUID', 0))} | {int(fsrc.get('VISCODE2', 0))} | "
                     f"{int((~x.fs_available).sum())} |")
    L += ["\n## Per-feature missingness before frozen imputation (usable scans with a clinical record)\n",
          "| Feature | baseline | follow-up |", "|---|---|---|"]
    c = u[u.has_clin]
    for f in VISIT_FEATURES:
        L.append(f"| {f} | {c[c.timepoint == 'baseline'][f].isna().mean():.3f} | "
                 f"{c[c.timepoint == 'follow-up'][f].isna().mean():.3f} |")
    os.makedirs(RESULTS, exist_ok=True)
    open(os.path.join(RESULTS, "inventory.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
