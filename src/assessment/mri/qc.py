"""MRI quality gates. Thresholds are declared in docs/eval_feedback_plan.md before any run.

Header/voxel-size gate and the registration gate are computed here; the intensity-reliability rule
(no zero voxels in the mask, or > 10% saturated) is the frozen rule from the brain-age work and is
evaluated by the MRI worker, which has the volume; its result is passed in as a boolean.
"""
from ..schema import QCGate, Status

CORR_PASS, CORR_FAIL = 0.30, 0.15
VOX_PASS, VOX_WARN = 1.6, 2.0
VOX_MIN, MIN_SLICES = 0.8, 100


def voxel_gate(spacing: tuple[float, ...], shape: tuple[int, ...]) -> QCGate:
    txt = "x".join(f"{s:.2f}" for s in spacing) + " mm, " + "x".join(map(str, shape))
    thr = f"PASS: every dim {VOX_MIN}-{VOX_PASS} mm and >={MIN_SLICES} slices; WARN <= {VOX_WARN} mm; FAIL otherwise"
    if len(shape) != 3 or len(spacing) != 3 or any(s <= 0 for s in spacing):
        return QCGate(name="voxel size / dimensions", value=txt, threshold=thr, status="FAIL",
                      note="not a 3D volume with a valid header")
    if max(spacing) > VOX_WARN or min(shape) < MIN_SLICES:
        st: Status = "FAIL"
    elif max(spacing) > VOX_PASS or min(spacing) < VOX_MIN:
        st = "WARN"
    else:
        st = "PASS"
    return QCGate(name="voxel size / dimensions", value=txt, threshold=thr, status=st)


def registration_gate(corr_vs_mni: float) -> QCGate:
    thr = f"PASS >= {CORR_PASS}; WARN {CORR_FAIL}-{CORR_PASS}; FAIL < {CORR_FAIL}"
    st: Status = "PASS" if corr_vs_mni >= CORR_PASS else "WARN" if corr_vs_mni >= CORR_FAIL else "FAIL"
    note = {"PASS": None, "WARN": "registration is suspect", "FAIL": "registration failed; MRI withheld"}[st]
    return QCGate(name="registration corr_vs_mni", value=round(float(corr_vs_mni), 3), threshold=thr,
                  status=st, note=note)


def reliability_gate(reliable: bool) -> QCGate:
    return QCGate(name="intensity reliability (frozen rule)", value="reliable" if reliable else "unreliable",
                  threshold="PASS: zero voxels present in mask and <= 10% saturated",
                  status="PASS" if reliable else "WARN",
                  note=None if reliable else "brain-age gap unreliable; imputed with training median, as frozen")


def overall(gates: list[QCGate]) -> Status:
    s = {g.status for g in gates}
    return "FAIL" if "FAIL" in s else "WARN" if "WARN" in s else "PASS"
