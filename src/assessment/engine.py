"""AssessmentEngine: one case in, one validated AssessmentReport out.

Steps: validate clinical inputs -> (native scan: m1 preprocessing and header QC) -> registration QC -> mode routing ->
volumes (adni_fs lookup for ADNI demo cases, SynthSeg for uploads) -> ALZ / reliability / brain age / streams / KAN
(the four venv workers) -> explanations -> report_builder. Every step is timed.
"""
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import time
import uuid

import numpy as np

from . import adni_demo, clinical, pathguard, registry
from .core import run_core
from .mri import qc as mqc
from .modes import effective_mode
from .paths import REPO
from .report_builder import BuildInput, NAMES, build
from .schema import AssessmentReport
from .volumes import adni_fs, agreement
from .workers.pool import WorkerPool

RUNS = REPO / "var" / "runs"


class Timer:
    def __init__(self):
        self.steps = {}

    def run(self, name, fn, *a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            self.steps[name] = self.steps.get(name, 0.0) + time.perf_counter() - t


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True).stdout.strip() or None
    except OSError:
        return None


class AssessmentEngine:
    def __init__(self, run_root=RUNS, *, guard_outside_git=False, release_around_synthseg=False, before_synthseg=None):
        """guard_outside_git: refuse a run_root inside the repository or any git work tree (required for external
        datasets, whose per-subject outputs must never be able to land in git). Checked before anything is created.
        release_around_synthseg: stop the mri and dbn workers before the SynthSeg step (they restart lazily afterwards),
        so SynthSeg does not share the machine with their models. before_synthseg: optional callable run after that
        release and immediately before SynthSeg (the batch runner's memory-floor check). Both default to off and
        change no report content."""
        if guard_outside_git:
            run_root = pathguard.assert_outside_git(run_root, "run root")
        self.run_root = pathlib.Path(run_root)
        self.pool = WorkerPool(str(self.run_root / "worker_logs"))
        self.release_around_synthseg = release_around_synthseg
        self.before_synthseg = before_synthseg

    def _volumes_synthseg(self, tm, native_path, out):
        """SynthSeg step: optionally release the mri and dbn workers, run the memory check, then segment."""
        from .volumes import synthseg
        if self.release_around_synthseg:
            tm.run("release_workers", self.pool.release, "mri", "dbn")
        if self.before_synthseg is not None:
            self.before_synthseg()
        return tm.run("volumes_synthseg", synthseg.run, str(native_path), str(out / "synthseg"))

    def close(self):
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def predict(self, mode="FULL", clinical_values=None, native_path=None, adni_rid=None, subject_label=None,
                report_id=None, verify_artifacts=True, progress=None, in_sample=False,
                extra_caveats=None) -> AssessmentReport:
        """clinical_values: raw form dict (None for MRI_ONLY). native_path: a T1 NIfTI upload. adni_rid: ADNI demo case
        (clinical values, volumes and stored m1 outputs are looked up; mutually exclusive with the two above)."""
        tm = Timer()
        t0 = time.perf_counter()
        report_id = pathguard.check_report_id(report_id or uuid.uuid4().hex[:12])
        out = self.run_root / report_id
        out.mkdir(parents=True, exist_ok=True)
        note = progress or (lambda step: None)
        demo = adni_demo.case(adni_rid) if adni_rid is not None else None
        provenance = "adni_demo" if demo else "new_upload"
        note("validate")
        arts = tm.run("verify_artifacts", registry.verify) if verify_artifacts else []

        # 1. clinical inputs
        raw_in = demo["clinical"] if demo and mode != "MRI_ONLY" else clinical_values
        if demo and mode == "MRI_ONLY":
            raw_in = {k: v for k, v in demo["clinical"].items() if k in ("AGE", "PTGENDER")}
        has_clin = raw_in is not None
        clean, items, n_imp, fails, warns = {}, [], 0, [], []
        if has_clin:
            clean, warns = clinical.validate(raw_in)
            items, n_imp, fails = clinical.completeness(clean)
        clinical_ok = has_clin and not fails
        if mode == "MRI_ONLY" and demo is None and native_path is not None and (
                clean.get("AGE") is None or clean.get("PTGENDER") is None):
            raise ValueError("MRI_ONLY needs AGE and PTGENDER: age at scan is required for the brain-age gap and the "
                             "MRI specialist")

        # 2. scan: stored (ADNI) or native upload
        gates, scan = [], None
        has_scan = bool(demo["has_scan"]) if demo else native_path is not None
        if mode == "CLINICAL_ONLY":
            has_scan = False
        mri_ok = False
        if has_scan:
            if demo:
                scan = {"mni": demo["mni"], "vol96": demo["vol96"], "corr": demo["corr_vs_mni"],
                        "age_at_scan": demo["age_at_scan"]}
            else:
                import nibabel as nib
                h = nib.load(native_path)
                vg = mqc.voxel_gate(tuple(float(x) for x in h.header.get_zooms()[:3]), tuple(h.shape))
                gates.append(vg)
                if vg.status != "FAIL":
                    note("preprocess")
                    pp = tm.run("m1_preprocess", self.pool.mri.call, "preprocess", native_path=str(native_path),
                                out_dir=str(out / "m1"))
                    scan = {"mni": pp["mni"], "vol96": pp["vol96"], "corr": pp["qc"]["corr_vs_mni"],
                            "age_at_scan": clean.get("AGE")}
            if scan:
                rg = mqc.registration_gate(scan["corr"])
                gates.append(rg)
                mri_ok = rg.status != "FAIL" and not any(g.status == "FAIL" for g in gates)
        mode_run, _ = effective_mode(mode, has_clin, has_scan, mri_ok, clinical_ok)
        use_scan = mode_run != "CLINICAL_ONLY" and scan is not None

        # 3. volumes
        vols, provider, extra, agree_txt = None, "none", [], None
        if use_scan:
            note("volumes")
            if demo:
                vols = tm.run("volumes_adni_fs", adni_fs.lookup, adni_rid)
                provider = "adni_fs"
            else:
                from .volumes import synthseg
                ss = self._volumes_synthseg(tm, native_path, out)
                vols, provider = ss["volumes"], "synthseg"
                a = agreement.load()
                gates.insert(0, agreement.qc_gate(a))
                gates.append(mqc.QCGate(name="SynthSeg QC (min over structure groups)", value=round(ss["qc_min"], 2),
                                        threshold=f"PASS >= {synthseg.QC_MIN}", status="PASS" if ss["qc_ok"] else "WARN"))
                extra, agree_txt = agreement.caveats(a), agreement.summary(a)

        # 4. frozen core
        note("model")
        raw = dict(clean)
        if use_scan:
            raw.update({k: v for k, v in vols.items()})
        case = {"key": "case", "raw": raw, "vol96": scan["vol96"] if use_scan else None,
                "mni": scan["mni"] if use_scan else None,
                "brain_age": bool(use_scan and scan["corr"] >= mqc.CORR_FAIL),
                "age_at_scan": scan["age_at_scan"] if use_scan else None}
        res = run_core(self.pool, [case], str(out / "job"), explain=True, layercam=use_scan)
        tm.steps.update({k: v for k, v in res["timings"].items()})
        if use_scan:
            gates.append(mqc.reliability_gate(not bool(res["unreliable"][0])))

        # 5. report
        note("report")
        z, ks = np.load(res["streams_npz"]), np.load(res["shapley_npz"])
        seeds = (42, 43, 44, 45, 46)
        core = {"p": res["p"][0], "add": res["add"][0], "alz": res["alz"][0], "pred_age": res["pred_age"][0],
                "unreliable": res["unreliable"][0], "age_at_scan": case["age_at_scan"],
                "clin_p": np.mean([z[f"clin_{s}"][0] for s in seeds], 0),
                "mri_p": np.mean([z[f"mri_{s}"][0] for s in seeds], 0) if use_scan else None,
                "cshap": z["cshap"][0], "clinical_features": z["clinical_features"], "cogindex": z["cogindex"][0],
                "corrected_gap": z["corrected_gap"][0], "mshap_phi": z["mshap_phi"][0],
                "mshap_imputed": z["mshap_imputed"][0], "mri_features": z["mri_features"],
                "kan": {"phi_group": ks["phi_group"][:, 0], "p_mri_masked": ks["p_mri_masked"][0],
                        "groups": ks["groups"], "targets": ks["targets"]}}
        cnn = res.get("layercam", {}).get("case") if use_scan else None
        if use_scan:
            shutil.copyfile(scan["mni"], out / "brain.nii.gz")
            if cnn:
                shutil.copyfile(cnn["overlay"], out / "overlay.nii.gz")
        bi = BuildInput(
            report_id=report_id, created_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            subject_label=subject_label or f"{provenance.replace('_', '-')}-{report_id[:6]}", requested_mode=mode,
            mode=mode_run, input_provenance=provenance, volume_provider=provider, clinical=clean, completeness=items,
            n_imputed=n_imp, qc_gates=gates, vols_mm3=vols, core=core, cnn=cnn, has_brain_image=use_scan,
            artifacts=arts, code_version=git_sha(), in_sample=in_sample or bool(demo and demo["split"] in ("train", "val")),
            volume_agreement=agree_txt,
            extra_caveats=extra + list(extra_caveats or []) + (
                [{"id": "extrapolation", "text": "Some inputs are outside the range of the training subjects: "
                  + "; ".join(w.split(" is outside")[0] for w in warns) + "."}] if warns else []))
        report = tm.run("report_builder", build, bi)
        total = time.perf_counter() - t0
        report = report.model_copy(update={"meta": report.meta.model_copy(update={"runtime_s": total})})
        AssessmentReport.model_validate(report.model_dump())          # re-validate after the update
        (out / "report.json").write_text(report.model_dump_json(indent=1))
        (out / "steps.json").write_text(json.dumps({"total_s": total, "steps_s": tm.steps}, indent=1))
        note("done")
        return report
