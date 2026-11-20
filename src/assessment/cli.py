"""assess: command line for the engine.

  python -m src.assessment.cli predict --adni-rid 1234 --mode FULL
  python -m src.assessment.cli predict --nifti scan.nii.gz --clinical form.json --mode FULL
  python -m src.assessment.cli predict --nifti scan.nii.gz --clinical form.json --run-root /data/outside/any/repo
  python -m src.assessment.cli verify-artifacts
"""
import json
import pathlib

import typer

from . import memguard, pathguard, registry
from .engine import RUNS, AssessmentEngine

app = typer.Typer(add_completion=False, help="Cognitive assessment engine (research use only).")


@app.command()
def predict(mode: str = typer.Option("FULL", help="FULL | CLINICAL_ONLY | MRI_ONLY"),
            nifti: pathlib.Path = typer.Option(None, help="T1 NIfTI upload"),
            clinical: pathlib.Path = typer.Option(None, help="JSON file of clinical form values"),
            adni_rid: int = typer.Option(None, help="ADNI demo case by RID (local data; stored m1 outputs, FS volumes)"),
            report_id: str = typer.Option(None), label: str = typer.Option(None),
            run_root: pathlib.Path = typer.Option(None, help="output folder (default: var/runs inside the repo). Required "
                                                  "for external datasets; must be outside every git work tree"),
            release_workers: bool = typer.Option(False, help="stop the mri and dbn workers before the SynthSeg step "
                                                 "(they restart afterwards); lowers peak memory, changes no output")):
    values = json.loads(clinical.read_text()) if clinical else None
    if run_root is None and nifti is not None and pathguard.EXTERNAL_ID.search(str(nifti)):
        raise typer.BadParameter("this scan looks like an external-dataset file: pass --run-root outside any git work tree")
    try:
        report_id = pathguard.check_report_id(report_id) if report_id else None      # before anything is created
        eng = (AssessmentEngine(run_root=run_root, guard_outside_git=True, release_around_synthseg=release_workers)
               if run_root else AssessmentEngine(release_around_synthseg=release_workers))
    except pathguard.GuardError as e:
        raise typer.BadParameter(str(e))
    with eng:
        r = eng.predict(mode=mode, clinical_values=values, native_path=str(nifti) if nifti else None,
                        adni_rid=adni_rid, subject_label=label, report_id=report_id,
                        progress=lambda s: typer.echo(f"[{s}]"))
    d = eng.run_root / r.meta.report_id
    typer.echo(f"report: {d / 'report.json'}  mode={r.meta.mode}  runtime={r.meta.runtime_s:.1f}s")
    typer.echo((d / "steps.json").read_text())


UPLOAD_DEMO_NOTE = ("This is an ADNI scan used to demonstrate the upload path. The subject was in the frozen system's "
                    "training data, so the output is in-sample. It is a demonstration, not a validation.")


@app.command()
def precompute(out: pathlib.Path = typer.Option("demo_cases", help="gitignored folder for the precomputed reports"),
               only: str = typer.Option("", help="comma-separated case names to (re)run, default all")):
    """Run the demo cases chosen by docs/demo_case_selection.md through the single-scan engine path (batch of 1)
    and keep each report with its skull-stripped image and overlay as the instant fallback set."""
    import shutil
    import sys
    sys.path.insert(0, ".")
    from tools.select_demo_cases import select
    sel = select()
    plan = [("cn", dict(adni_rid=sel["cn"], mode="FULL")), ("lmci", dict(adni_rid=sel["lmci"], mode="FULL")),
            ("ad", dict(adni_rid=sel["ad"], mode="FULL")),
            ("missing-modality", dict(adni_rid=sel["missing_modality"], mode="CLINICAL_ONLY")),
            ("upload-synthseg", dict(adni_rid=sel["new_upload"], mode="FULL", upload=sel["new_upload_native"]))]
    out.mkdir(parents=True, exist_ok=True)
    wanted = {x for x in only.split(",") if x}
    with AssessmentEngine() as eng:
        for name, kw in plan:
            if wanted and name not in wanted:
                continue
            typer.echo(f"== {name}")
            up = kw.pop("upload", None)
            if up:  # an ADNI scan through the upload path: native NIfTI + form values, SynthSeg volumes
                from . import adni_demo
                form = adni_demo.case(kw.pop("adni_rid"))["clinical"]
                r = eng.predict(mode=kw["mode"], clinical_values=form, native_path=up, subject_label=f"demo-{name}",
                                report_id=f"demo-{name}", in_sample=True,
                                extra_caveats=[{"id": "upload_demo", "text": UPLOAD_DEMO_NOTE}])
            else:
                r = eng.predict(subject_label=f"demo-{name}", report_id=f"demo-{name}", **kw)
            src, dst = RUNS / r.meta.report_id, out / r.meta.report_id
            shutil.rmtree(dst, ignore_errors=True)
            dst.mkdir(parents=True)
            for f in ("report.json", "steps.json", "brain.nii.gz", "overlay.nii.gz"):
                if (src / f).exists():
                    shutil.copyfile(src / f, dst / f)
    typer.echo(f"precomputed reports in {out}")


@app.command("predict-batch")
def predict_batch(manifest: pathlib.Path = typer.Option(..., help="folder holding inputs.csv (labels.csv is never opened)"),
                  run_root: pathlib.Path = typer.Option(..., help="output folder, outside every git work tree"),
                  label_map: str = typer.Option("cdr_3class", help="registered label map (name only; labels are not read)"),
                  mode: str = typer.Option("MRI_ONLY", help="only MRI_ONLY is supported"),
                  stage: str = typer.Option("full", help="plumbing | full (full needs a clean tree)"),
                  limit: int = typer.Option(None, help="first N case ids in sorted order"),
                  resume: bool = typer.Option(False), retry_failed: bool = typer.Option(False),
                  allow_dirty: bool = typer.Option(False, help="plumbing runs only"),
                  min_available_gb: float = typer.Option(memguard.DEFAULT_FLOOR_GB, help="before each SynthSeg call, stop "
                                                         "cleanly if MemAvailable (GiB) is below this; 0 disables"),
                  release_workers: bool = typer.Option(True, help="stop the mri and dbn workers before each SynthSeg call")):
    """Blind, resumable, sequential batch run through the unchanged engine. Prints counters only."""
    from . import batch
    try:
        res = batch.run(manifest, run_root, mode=mode, stage=stage, label_map=label_map, limit=limit, resume=resume,
                        retry_failed=retry_failed, allow_dirty=allow_dirty, out=typer.echo,
                        min_available_gb=min_available_gb, release_workers=release_workers)
    except batch.BatchHalt as e:
        typer.echo(f"halted: {e}", err=True)
        raise typer.Exit(3)
    except (batch.BatchError, pathguard.GuardError, KeyError) as e:
        typer.echo(f"refused: {str(e).strip(chr(39))}", err=True)
        raise typer.Exit(2)
    typer.echo(f"batch finished: ok={res['n_ok']} failed={res['n_failed']} cases={res['n_cases']}")


@app.command("batch-check")
def batch_check(run_root: pathlib.Path = typer.Option(...)):
    """Pass/fail plumbing checks for a finished plumbing run. No scores are read or printed."""
    from . import batch
    checks = batch.check_plumbing(pathguard.assert_outside_git(run_root, "run root"))
    for name, ok in checks:
        typer.echo(f"{'PASS' if ok else 'FAIL'}  {name}")
    raise typer.Exit(0 if all(ok for _, ok in checks) else 1)


@app.command("batch-purge")
def batch_purge(run_root: pathlib.Path = typer.Option(...)):
    """Delete a finished batch run root (Stage B outputs are removed before the full run)."""
    from . import batch
    try:
        batch.purge(run_root)
    except (batch.BatchError, pathguard.GuardError) as e:
        typer.echo(f"refused: {e}", err=True)
        raise typer.Exit(2)
    typer.echo("purged")


@app.command("verify-artifacts")
def verify_artifacts():
    typer.echo(f"{len(registry.verify())} artifacts verified")


if __name__ == "__main__":
    app()
