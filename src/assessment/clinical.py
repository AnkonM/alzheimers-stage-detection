"""Clinical input validation and completeness bookkeeping.

The numeric imputation, z-scoring and CogIndex are applied later by the streams worker with the frozen
feature_stats.json (analysis/temporal_eval/infer_streams.frame), so this module only decides what the
user supplied, what is out of range, and what will be imputed (with the training median, as frozen).
"""
import json

from .paths import FROZEN, REPO
from .schema import InputItem

# Instrument limits. Values outside these are rejected, not clipped.
HARD = {
    "AGE": (18, 110), "PTEDUCAT": (0, 30), "APOE4": (0, 2), "CDRSB": (0, 18), "ADAS11": (0, 70),
    "ADAS13": (0, 85), "MMSE": (0, 30), "RAVLT_immediate": (0, 75), "RAVLT_learning": (-15, 15),
    "RAVLT_forgetting": (-15, 15), "RAVLT_perc_forgetting": (-1500, 100), "FAQ": (0, 30), "MOCA": (0, 30),
}
ECOG = (1.0, 4.0)
VOLUME_FIELDS = {"Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV"}
# Missing any of these is a FAIL for modes that use the clinical stream: they drive the KAN extras.
REQUIRED = ("CDRSB", "MMSE", "ADAS13")
MAX_MISSING_WARN = 4
SEXES = ("Male", "Female")


class ClinicalError(ValueError):
    """Hard-range violation or malformed input; the message names the field."""


def stats() -> dict:
    return json.load(open(FROZEN / "feature_stats.json"))


def input_fields() -> list[str]:
    s = stats()
    return [f for f in s["num"] + s["cat"] if f not in VOLUME_FIELDS]


def limits(field: str):
    if field.startswith("Ecog"):
        return ECOG
    return HARD.get(field)


def soft_limits(field: str):
    """[min, max] of the field over the 569 train subjects (committed aggregate), or None. Outside it: extrapolation."""
    r = json.load(open(REPO / "analysis" / "inference_refs" / "results" / "reference_ranges.json"))["train_range"]
    return r.get(field)


def core_fields() -> list[str]:
    """The fields the frozen clinical specialists and KAN extras read (the union of the per-seed top-12 plus extras)."""
    cols = json.load(open(FROZEN / "columns.json"))
    u = {c for v in cols["clinical_top12"].values() for c in v} | set(cols["extras"])
    u -= {"CogIndex"}                       # derived from MMSE, CDRSB and ADAS13
    return [f for f in input_fields() if f in u or f in ("AGE", "PTGENDER")]


def derive_ravlt(raw: dict) -> dict:
    """Fill the four RAVLT features from raw trials, as the ADNI tables define them.

    raw may carry ravlt_trials (five trial totals) and ravlt_delayed. perc_forgetting is rounded to 2 dp,
    as ADSL does. Explicit values the caller gave are never overwritten.
    """
    out = dict(raw)
    t, d = out.pop("ravlt_trials", None), out.pop("ravlt_delayed", None)
    if t is None:
        return out
    if len(t) != 5:
        raise ClinicalError("ravlt_trials must hold the five trial totals")
    derived = {"RAVLT_immediate": float(sum(t)), "RAVLT_learning": float(t[4] - t[0])}
    if d is not None:
        derived["RAVLT_forgetting"] = float(t[4] - d)
        if t[4] > 0:
            derived["RAVLT_perc_forgetting"] = round(100 * (t[4] - d) / t[4], 2)
    for k, v in derived.items():
        out.setdefault(k, v)
    return out


def validate(raw: dict) -> tuple[dict, list[str]]:
    """Return (clean values for known fields, warnings). Raises ClinicalError on a hard violation."""
    raw = derive_ravlt({k: v for k, v in raw.items() if v is not None})
    known = set(input_fields())
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ClinicalError(f"unknown field(s): {', '.join(unknown)}")
    clean = {}
    for f, v in raw.items():
        if f == "PTGENDER":
            if v not in SEXES:
                raise ClinicalError(f"PTGENDER must be one of {SEXES}")
            clean[f] = v
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ClinicalError(f"{f} must be a number") from None
        lo_hi = limits(f)
        if lo_hi and not lo_hi[0] <= v <= lo_hi[1]:
            raise ClinicalError(f"{f}={v} outside the allowed range {lo_hi[0]}..{lo_hi[1]}")
        clean[f] = v
    warnings = []
    for f, v in clean.items():
        sl = None if f == "PTGENDER" else soft_limits(f)
        if sl and not sl[0] <= v <= sl[1]:
            warnings.append(f"{f}={v:g} is outside the range of the training subjects ({sl[0]:g}..{sl[1]:g}): extrapolation")
    return clean, warnings


def completeness(clean: dict) -> tuple[list[InputItem], int, list[str]]:
    """Per-field provided/imputed record, the imputed count, and the reasons that make the input unusable."""
    s = stats()
    items = []
    for f in input_fields():
        if f in clean:
            items.append(InputItem(field=f, state="provided", value=clean[f]))
        else:
            fill = s["median"].get(f, s["mode"].get(f))
            items.append(InputItem(field=f, state="imputed", value=fill, note="training median"
                                   if f in s["median"] else "training mode"))
    n_imp = sum(i.state == "imputed" for i in items)
    fails = [f"required input missing: {f}" for f in REQUIRED if f not in clean]
    return items, n_imp, fails


def used_fields() -> set[str]:
    """The union of fields the frozen models read directly (for the 'more than 4 missing' warning)."""
    cols = json.load(open(FROZEN / "columns.json"))
    u = {c for v in cols["clinical_top12"].values() for c in v}
    u |= set(cols["extras"]) | set(cols["catboost_F_cols"]) | set(cols["catboost_R_cols"])
    return (u & set(input_fields())) | {"AGE", "PTGENDER"}


def n_missing_used(clean: dict) -> int:
    return len([f for f in used_fields() if f not in clean])
