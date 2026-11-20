"""AssessmentReport: the single contract between the engine, the API and the report UI.

Every number in a report comes from the engine; the UI renders, it never computes.
Two content rules are enforced here rather than left to the renderer:
  * clinical-stream driver text and feature names must not refer to MRI;
  * the word "calibrated" must not appear in any free text (the frozen
    probabilities are shown as "model probability").
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"
STAGES = ("CN", "EMCI", "LMCI", "AD")
Stage = Literal["CN", "EMCI", "LMCI", "AD"]
Status = Literal["PASS", "WARN", "FAIL"]
Mode = Literal["FULL", "CLINICAL_ONLY", "MRI_ONLY"]

MRI_TERMS = re.compile(
    r"\b(mri|scan|brain[- ]?age|volume|ventric|hippocamp|entorhinal|fusiform|icv|cnn|layercam|"
    r"freesurfer|synthseg|atrophy)\w*|\b(alz|cam)\b", re.I)
BANNED_WORD = re.compile(r"calibrat", re.I)


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class Artifact(Model):
    name: str
    sha256: str


class ReferencePopulation(Model):
    description: str = "ADNI training subjects (train split only)"
    n: int = 569
    volume_percentiles: str = "pooled CN-train, not age/sex-adjusted"
    per_stage_ranges: str = "p25/p50/p75 by true stage, train split only"


MODE_LABELS = {
    "FULL": "fusion model with scan and clinical inputs",
    "CLINICAL_ONLY": "fusion model run without scan inputs",  # the frozen KAN with no-scan inputs
    "MRI_ONLY": "MRI specialist only (AD probability; no stage)",
}
ALZ_BATCH_CAVEAT = ("ALZ may differ from the published run by up to about 5e-5 because of GPU batch effects; "
                    "the predicted stage is unaffected.")


class Meta(Model):
    schema_version: str = SCHEMA_VERSION
    report_id: str
    created_at: str  # ISO 8601 UTC
    subject_label: str  # pseudonymous label chosen by the user, never an ADNI ID in committed files
    visit_label: str | None = None
    mode: Mode
    mode_label: str  # fixed text per mode, see MODE_LABELS
    requested_mode: Mode  # differs from mode when FULL was downgraded
    system_version: str = "fusion-v3"
    code_version: str | None = None
    artifacts: list[Artifact] = []
    input_provenance: Literal["adni_demo", "new_upload"]
    reference_population: ReferencePopulation = ReferencePopulation()
    volume_provider: Literal["adni_fs", "synthseg", "none"] = "none"
    volume_agreement: str | None = None  # quoted ICC summary when volume_provider == synthseg
    runtime_s: float | None = None
    has_brain_image: bool = False  # skull-stripped scan available to the viewer (served by the API)


class QCGate(Model):
    name: str
    value: float | str | None = None
    threshold: str
    status: Status
    note: str | None = None


class QC(Model):
    gates: list[QCGate] = []
    overall: Status = "PASS"


class InputItem(Model):
    field: str
    state: Literal["provided", "imputed", "missing"]
    value: float | str | None = None  # value used by the model
    note: str | None = None  # e.g. "training median"


class Completeness(Model):
    clinical: list[InputItem] = []
    mri_provided: bool
    n_clinical_imputed: int = 0


class Severity(Model):
    value: float = Field(ge=0, le=3)  # sum_k k * p(stage k)
    range: tuple[float, float] = (0.0, 3.0)
    stage_means: dict[Stage, float] | None = None  # reference class means (train)


class Ambiguity(Model):
    flag: bool
    top2: tuple[Stage, Stage]
    margin: float
    rule: str


class StageResult(Model):
    model_probability: dict[Stage, float]  # frozen softmax output, shown as "model probability"
    predicted: Stage
    clinical_predicted: Stage | None = None  # clinical-specialist stage, used only for the "fusion changed the stage" flag
    severity: Severity | None = None
    ambiguity: Ambiguity | None = None

    @model_validator(mode="after")
    def _sums(self):
        if abs(sum(self.model_probability.values()) - 1) > 1e-3:
            raise ValueError("stage probabilities must sum to 1")
        if self.predicted != max(self.model_probability, key=self.model_probability.get):
            raise ValueError("predicted stage must be the argmax")
        return self


class ADResult(Model):
    model_probability: float = Field(ge=0, le=1)
    source: Literal["KAN_ADD_HEAD", "MRI_SPECIALIST", "ALZ_ONLY"]
    alz: float | None = Field(default=None, ge=0, le=1)  # CNN AD probability


class BrainAge(Model):
    status: Literal["reliable", "unreliable", "not_available"]
    age_at_scan: float | None = None
    predicted: float | None = None
    raw_gap: float | None = None
    corrected_gap: float | None = None
    extreme: bool = False
    note: str | None = None


# SynthSeg display rule (analysis/volume_agreement/PLAN.md, amendment 2026-10-29): a percentile is shown only for
# ratios with ICC(2,1) >= 0.50 and |bias| <= 10% in the committed agreement check.
SYNTHSEG_PERCENTILE_OK = {"WholeBrain", "Hippocampus"}
LOW_AGREEMENT_NOTE = "low agreement with training volumes, no percentile"
AGREEMENT_GATE = "volume agreement (SynthSeg)"


class VolumeItem(Model):
    structure: str
    ratio_to_icv: float
    agreement: Literal["not_applicable", "ok", "low"] = "not_applicable"
    display_note: str | None = None
    percentile: float | None = Field(default=None, ge=0, le=100)
    percentile_basis: str = "pooled CN-train (n=133), not age/sex-adjusted"
    pct_of_cn_median: float | None = None


class StageRange(Model):
    p25: float
    p50: float
    p75: float


class Driver(Model):
    feature: str
    label: str
    value: float | str | None
    shap: float  # log-odds toward the predicted stage
    imputed: bool = False
    per_stage_ranges: dict[Stage, StageRange] | None = None  # train-569 only


class StreamContribution(Model):
    group: str
    mean: float
    consistent: bool


class Validation(Model):
    status: Literal["VALIDATED", "WEAK"]
    detail: str | None = None


class Drivers(Model):
    clinical_target: Stage | None = None  # stage whose log-odds the clinical SHAP explains (clinical specialist's own prediction)
    mri_target: Stage | None = None  # likewise for the MRI specialist
    stream_level: list[StreamContribution] = []  # KAN group Shapley, final predicted-stage logit
    clinical: list[Driver] = []
    clinical_validation: Validation | None = None
    mri: list[Driver] = []
    mri_validation: Validation | None = None

    @model_validator(mode="after")
    def _clinical_has_no_mri(self):
        for d in self.clinical:
            if MRI_TERMS.search(d.feature) or MRI_TERMS.search(d.label):
                raise ValueError(f"MRI term in clinical driver: {d.feature!r}")
        return self


class RegionDensity(Model):
    region: str
    density: float
    ad_minus_cn: float | None = None


class CNNPanel(Model):
    """Technical view only. Both pre-registered sanity checks FAIL, so no anatomical claim is made."""
    layer: str = "layer1 LayerCAM"
    n_models: int | None = None
    overlay_file: str | None = None
    sanity_cascade: Status = "FAIL"
    sanity_deletion: Status = "FAIL"
    regions: list[RegionDensity] = []
    wording: str = "attribution density, not evidence of disease location"


class Flag(Model):
    id: Literal["discordance", "fusion_changed_stage", "mri_changed_stage", "ambiguous"]
    text: str


class Caveat(Model):
    id: str
    text: str


class AssessmentReport(Model):
    meta: Meta
    qc: QC
    completeness: Completeness
    stage: StageResult | None = None  # None in MRI_ONLY
    ad: ADResult
    brain_age: BrainAge
    volumes: list[VolumeItem] = []
    drivers: Drivers = Drivers()
    cnn: CNNPanel | None = None
    flags: list[Flag] = []
    caveats: list[Caveat] = []
    research_use_notice: str = "Research use only. Not a diagnostic device."

    @model_validator(mode="after")
    def _rules(self):
        if self.meta.mode_label != MODE_LABELS[self.meta.mode]:
            raise ValueError(f"mode_label must be {MODE_LABELS[self.meta.mode]!r}")
        if self.ad.alz is not None and not any(c.id == "alz_batch_effect" and c.text == ALZ_BATCH_CAVEAT
                                               for c in self.caveats):
            raise ValueError("reports with an ALZ value require the alz_batch_effect caveat")
        if self.meta.mode == "MRI_ONLY" and self.stage is not None:
            raise ValueError("MRI_ONLY reports carry no 4-stage prediction")
        if self.meta.mode != "MRI_ONLY" and self.stage is None:
            raise ValueError("stage is required unless mode is MRI_ONLY")
        if self.meta.mode == "CLINICAL_ONLY" and (self.volumes or self.cnn or self.drivers.mri):
            raise ValueError("CLINICAL_ONLY reports carry no MRI content")
        if self.meta.mode == "CLINICAL_ONLY" and self.brain_age.status != "not_available":
            raise ValueError("CLINICAL_ONLY reports carry no brain age")
        if self.meta.volume_provider == "synthseg":
            if not self.meta.volume_agreement:
                raise ValueError("SynthSeg volumes require the quoted agreement caveat")
            if not self.qc.gates or self.qc.gates[0].name != AGREEMENT_GATE or self.qc.gates[0].status != "WARN":
                raise ValueError("SynthSeg reports need a WARN 'volume agreement' gate first in the QC list")
            for v in self.volumes:
                ok = v.structure in SYNTHSEG_PERCENTILE_OK
                if v.agreement != ("ok" if ok else "low"):
                    raise ValueError(f"{v.structure}: wrong agreement status for SynthSeg volumes")
                if not ok and (v.percentile is not None or v.pct_of_cn_median is not None
                               or v.display_note != LOW_AGREEMENT_NOTE):
                    raise ValueError(f"{v.structure}: low-agreement ratio must show no percentile")
            if any(f.id == "mri_changed_stage" for f in self.flags):
                raise ValueError("the 'MRI changed the stage' flag is suppressed on SynthSeg reports")
        elif any(v.agreement != "not_applicable" for v in self.volumes):
            raise ValueError("agreement status applies to SynthSeg volumes only")
        if self.meta.input_provenance == "new_upload" and "domain_shift" not in {c.id for c in self.caveats}:
            raise ValueError("new_upload reports require the domain_shift caveat")
        if self.meta.input_provenance == "adni_demo" and "domain_shift" in {c.id for c in self.caveats}:
            raise ValueError("domain_shift caveat applies to new uploads only")
        if self.meta.mode == "CLINICAL_ONLY":
            if any(MRI_TERMS.search(g.group) for g in self.drivers.stream_level):
                raise ValueError("CLINICAL_ONLY stream drivers must not include MRI streams")
            if self.stage.severity is not None:
                raise ValueError("CLINICAL_ONLY shows no severity by default")
            if "no_severity" not in {c.id for c in self.caveats}:
                raise ValueError("CLINICAL_ONLY requires the no_severity caveat")
        bad = _find_bad_strings(self.model_dump(mode="json"))
        if bad:
            raise ValueError(f"absolute path or subject ID in report body ({bad[0]})")
        texts = [c.text for c in self.caveats] + [f.text for f in self.flags]
        if any(BANNED_WORD.search(t) for t in texts):
            raise ValueError("the word 'calibrated' must not be used")
        return self


PTID = re.compile(r"\b\d{3}_S_\d{4}\b")
EXTERNAL_ID = re.compile(r"OAS[0-9]_[0-9]{4}", re.I)      # OASIS-1 subject/session id
ABS_PATH = re.compile(r"(^|\s)(/[A-Za-z0-9_.-]+/|[A-Za-z]:\\|~/)")


def _find_bad_strings(o):
    """Returns the KIND of problem, never the offending text: an error message must not echo an identifier."""
    if isinstance(o, str):
        if PTID.search(o) or EXTERNAL_ID.search(o):
            return ["subject ID"]
        return ["absolute path"] if ABS_PATH.search(o) else []
    items = o.values() if isinstance(o, dict) else o if isinstance(o, list) else []
    return [b for v in items for b in _find_bad_strings(v)]


def export_json_schema() -> dict:
    return AssessmentReport.model_json_schema(mode="serialization")  # defaulted fields are required in output


if __name__ == "__main__":
    import json
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "schemas/assessment_report.json"
    with open(out, "w") as f:
        json.dump(export_json_schema(), f, indent=2)
        f.write("\n")
