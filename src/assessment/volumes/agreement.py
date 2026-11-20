"""Caveat text for reports that use SynthSeg volumes, quoting the committed agreement check
(analysis/volume_agreement/results/agreement.json; descriptive, n=25 train scans, not a gate)."""
import json

from ..paths import REPO
from ..schema import AGREEMENT_GATE, QCGate, SYNTHSEG_PERCENTILE_OK

RULES = REPO / "analysis" / "inference_refs" / "results" / "rules.json"
RESULT = REPO / "analysis" / "volume_agreement" / "results" / "agreement.json"
ORDER = ["Hippocampus", "Ventricles", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp"]


def load() -> dict:
    return json.loads(RESULT.read_text())


def summary(a: dict | None = None) -> str:
    """One line for meta.volume_agreement."""
    a = a or load()
    parts = [f"{s} {a['ratios'][f'{s} / ICV']['icc21']:.2f}" for s in ORDER]
    return (f"ICC(2,1) of SynthSeg vs training (FreeSurfer 5.1) volume/ICV ratios, n={a['meta']['n_ok']} train scans: "
            + ", ".join(parts))


def caveats(a: dict | None = None) -> list[dict]:
    a = a or load()
    out = [{"id": "volume_provider", "text": "Volumes from SynthSeg, not the FreeSurfer 5.1 volumes the MRI "
            "specialist was trained on. Agreement with training volumes: " + summary(a) + "."}]
    if a["cortical_below_floor"]:
        out.append({"id": "volume_unreliable", "text": "Agreement is low for " + ", ".join(a["cortical_below_floor"])
                    + f" (ICC(2,1) below {json.loads(RULES.read_text())['synthseg_display']['min_icc21']}): the MRI-specialist contribution may be unreliable for this scan."})
    return out


def percentile_ok(a: dict | None = None) -> set[str]:
    """Ratios that may show a percentile: ICC(2,1) >= 0.50 and |bias| <= 10% (PLAN.md amendment 2026-10-29)."""
    a = a or load()
    rule = json.loads(RULES.read_text())["synthseg_display"]
    return {s for s in ORDER if a["ratios"][f"{s} / ICV"]["icc21"] >= rule["min_icc21"]
            and abs(a["ratios"][f"{s} / ICV"]["bias_pct_of_fs_mean"]) <= rule["max_abs_bias_pct"]}


def _threshold_text() -> str:
    r = json.loads(RULES.read_text())["synthseg_display"]
    return f"percentile shown only if ICC(2,1) >= {r['min_icc21']} and |bias| <= {r['max_abs_bias_pct']}%"


def qc_gate(a: dict | None = None) -> QCGate:
    """WARN-level QC item quoting ICC and bias for the six ratios; first in the MRI QC list."""
    a = a or load()
    txt = "; ".join(f"{s} ICC {a['ratios'][f'{s} / ICV']['icc21']:.2f}, bias "
                    f"{a['ratios'][f'{s} / ICV']['bias_pct_of_fs_mean']:+.0f}%" for s in ORDER)
    return QCGate(name=AGREEMENT_GATE, value=f"n={a['meta']['n_ok']} train scans: {txt}",
                  threshold=_threshold_text(), status="WARN",
                  note="SynthSeg volumes differ from the FreeSurfer 5.1 volumes the MRI specialist was trained on")
