"""Mode routing: what the inputs allow, and what QC leaves usable."""
from .schema import Mode


def effective_mode(requested: Mode, has_clinical: bool, has_mri: bool, mri_qc_ok: bool,
                   clinical_ok: bool) -> tuple[Mode, list[str]]:
    """Return (mode to run, caveat ids). Raises ValueError when no usable input remains."""
    notes = []
    if requested == "FULL":
        if not (has_clinical and has_mri):
            raise ValueError("FULL needs both a T1 scan and clinical inputs")
        if not mri_qc_ok:
            notes.append("downgraded")
            requested = "CLINICAL_ONLY"
    if requested == "CLINICAL_ONLY":
        if not has_clinical:
            raise ValueError("CLINICAL_ONLY needs clinical inputs")
        if not clinical_ok:
            raise ValueError("required clinical inputs are missing (CDRSB, MMSE, ADAS13)")
        notes.append("no_severity")
    if requested == "MRI_ONLY":
        if not has_mri:
            raise ValueError("MRI_ONLY needs a T1 scan")
        if not mri_qc_ok:
            raise ValueError("MRI failed quality control and no clinical inputs are available")
    if requested == "FULL" and not clinical_ok:
        raise ValueError("required clinical inputs are missing (CDRSB, MMSE, ADAS13)")
    return requested, notes
