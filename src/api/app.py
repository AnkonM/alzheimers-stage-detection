"""FastAPI app: job store + one background worker thread + the built frontend. Local use only (127.0.0.1).

Safeguards (tests/test_api.py): loopback host only; report files served by validated report id from a fixed allow-list
(skull-stripped brain, overlay for the technical view only); NIfTI-only, size-capped, sniffed uploads; delete endpoint and a
retention sweep; responses carry no subject IDs and no absolute paths.
"""
import json
import queue
import re
import shutil
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from ..assessment import clinical
from ..assessment.report_builder import LABELS
from ..assessment.schema import MODE_LABELS, AssessmentReport
from . import config, uploads
from .store import Store

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
LABEL_RE = re.compile(r"^[A-Za-z0-9 _.-]{0,40}$")
PTID_RE = re.compile(r"\b\d{3}_S_\d{4}\b")
ABS_PATH_RE = re.compile(r"(/[A-Za-z0-9_.-]+){2,}|[A-Za-z]:\\\S+")
VIEWS = ("clin", "tech")
FILES = {"brain.nii.gz": "brain", "overlay.nii.gz": "overlay"}
MODES = ("FULL", "CLINICAL_ONLY", "MRI_ONLY")


def safe_text(s: str) -> str:
    return ABS_PATH_RE.sub("<path>", PTID_RE.sub("<id>", s))[:300]


def plan_for(has_scan: bool) -> list[str]:
    return ["validate"] + (["preprocess", "volumes"] if has_scan else []) + ["model", "report"]


class Runner:
    """One background thread; the engine (and its persistent stage workers) is created on the first job."""

    def __init__(self, store: Store, engine_factory):
        self.store, self.factory = store, engine_factory
        self.q: queue.Queue = queue.Queue()
        self.engine = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, job):
        self.q.put(job)

    def warm(self):
        """Start the four stage workers in the background so the first job does not pay their start-up."""
        def go():
            try:
                if self.engine is None:
                    self.engine = self.factory()
                for w in ("mri", "dbn", "streams", "kan"):
                    getattr(self.engine.pool, w).call("ping")
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=go, daemon=True).start()

    def _loop(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            jid = job["id"]
            try:
                if self.engine is None:
                    self.engine = self.factory()
                self.engine.predict(mode=job["mode"], clinical_values=job["clinical"], native_path=job["native_path"],
                                    subject_label=job["subject_label"], report_id=jid,
                                    progress=lambda step, j=jid: self.store.step(j, step))
                self.store.finish(jid)
            except (clinical.ClinicalError, ValueError) as e:
                self.store.finish(jid, safe_text(str(e)))
            except Exception:  # noqa: BLE001
                self.store.finish(jid, "Processing failed; see the server log")
            finally:
                shutil.rmtree(config.UPLOADS / jid, ignore_errors=True)      # inputs are never kept after processing

    def stop(self):
        self.q.put(None)
        if self.engine is not None:
            self.engine.close()


def create_app(engine_factory=None, store: Store | None = None, start_runner=True) -> FastAPI:
    config.RUNS.mkdir(parents=True, exist_ok=True)
    config.UPLOADS.mkdir(parents=True, exist_ok=True)
    store = store or Store()
    if engine_factory is None:
        def engine_factory():
            from ..assessment.engine import AssessmentEngine
            return AssessmentEngine(run_root=config.RUNS)
    app = FastAPI(title="Cognitive assessment (research use only)", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store
    app.state.runner = Runner(store, engine_factory) if start_runner else None

    @app.middleware("http")
    async def local_only(request: Request, call_next):          # also blocks DNS-rebinding to the loopback server
        host = (request.headers.get("host") or "").split(":")[0]
        if host not in config.ALLOWED_HOSTS:
            return JSONResponse({"detail": "Local access only"}, status_code=403)
        return await call_next(request)

    def job_dict(r):
        d = {"id": r["id"], "status": r["status"], "step": r["step"], "mode": r["mode"],
             "plan": json.loads(r["plan"]), "steps_done": [x["step"] for x in json.loads(r["steps"])],
             "timeline": [{"step": x["step"], "start_s": round(x["t"] - r["created_at"], 1)}
                          for x in json.loads(r["steps"])],
             "elapsed_s": round((r["finished_at"] or time.time()) - r["created_at"], 1),
             "precomputed": bool(r["precomputed"]), "error": r["error"],
             "report_id": r["id"] if r["status"] == "done" else None}
        return d

    def need_job(jid):
        if not ID_RE.match(jid):
            raise HTTPException(404, "Not found")
        r = store.get(jid)
        if r is None:
            raise HTTPException(404, "Not found")
        return r

    def run_dir(rid):
        if not ID_RE.match(rid):
            raise HTTPException(404, "Not found")
        d = (config.RUNS / rid).resolve()
        if config.RUNS.resolve() not in d.parents or not d.is_dir():
            raise HTTPException(404, "Not found")
        return d

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/form-spec")
    def form_spec():
        fields, stats, core = [], clinical.stats(), set(clinical.core_fields())
        for f in clinical.input_fields():
            lim = clinical.limits(f)
            soft = None if f == "PTGENDER" else clinical.soft_limits(f)
            med = stats.get("median", {}).get(f, stats.get("mode", {}).get(f))
            fields.append({"name": f, "label": LABELS.get(f, f), "kind": "choice" if f == "PTGENDER" else "number",
                           "choices": list(clinical.SEXES) if f == "PTGENDER" else None,
                           "min": lim[0] if lim else None, "max": lim[1] if lim else None,
                           "soft_min": soft[0] if soft else None, "soft_max": soft[1] if soft else None,
                           "default_if_missing": med, "core": f in core,
                           "required": f in clinical.REQUIRED or f in ("AGE", "PTGENDER")})
        return {"fields": fields, "required_for_clinical": list(clinical.REQUIRED),
                "modes": [{"value": m, "label": MODE_LABELS[m]} for m in MODES],
                "max_upload_mb": config.MAX_UPLOAD_BYTES // (1 << 20)}

    @app.post("/api/assessments", status_code=202)
    async def create_assessment(mode: str = Form("FULL"), clinical_json: str | None = Form(None),
                                subject_label: str = Form(""), scan: UploadFile | None = File(None)):
        if mode not in MODES:
            raise HTTPException(422, f"mode must be one of {MODES}")
        if not LABEL_RE.match(subject_label) or PTID_RE.search(subject_label):
            raise HTTPException(422, "Label: up to 40 letters, digits, spaces, dot, dash or underscore, and no subject ID")
        values, warns = None, []
        if clinical_json:
            try:
                values = json.loads(clinical_json)
                if not isinstance(values, dict):
                    raise ValueError
            except ValueError:
                raise HTTPException(422, "clinical_json must be a JSON object") from None
            try:
                clean, warns = clinical.validate(values)
            except clinical.ClinicalError as e:
                raise HTTPException(422, safe_text(str(e))) from None
            if mode in ("FULL", "CLINICAL_ONLY"):
                miss = [f for f in clinical.REQUIRED if f not in clean]
                if miss:
                    raise HTTPException(422, f"Required clinical inputs missing: {', '.join(miss)}")
            if mode == "MRI_ONLY" and not {"AGE", "PTGENDER"} <= set(clean):
                raise HTTPException(422, "MRI-only needs AGE and PTGENDER")
        has_scan = scan is not None and bool(scan.filename)
        if mode in ("FULL", "MRI_ONLY") and not has_scan:
            raise HTTPException(422, f"{mode} needs a T1 scan upload")
        if mode in ("FULL", "CLINICAL_ONLY") and values is None:
            raise HTTPException(422, f"{mode} needs clinical inputs")
        jid = uuid.uuid4().hex[:12]
        native = None
        if has_scan and mode != "CLINICAL_ONLY":
            native = str(await uploads.save(scan, config.UPLOADS / jid))
        store.create(jid, mode, subject_label or None, plan=plan_for(native is not None))
        app.state.runner.submit({"id": jid, "mode": mode, "clinical": values, "native_path": native,
                                 "subject_label": subject_label or None})
        return {"job_id": jid, "warnings": warns}

    @app.get("/api/jobs")
    def jobs():
        return [job_dict(r) for r in store.list()]

    @app.get("/api/jobs/{jid}")
    def job(jid: str):
        return job_dict(need_job(jid))

    @app.delete("/api/jobs/{jid}", status_code=204)
    def delete_job(jid: str):
        r = need_job(jid)
        if r["precomputed"]:
            raise HTTPException(403, "Precomputed demo reports cannot be deleted")
        remove_job(jid)
        return Response(status_code=204)

    def remove_job(jid):
        shutil.rmtree(config.UPLOADS / jid, ignore_errors=True)
        shutil.rmtree(config.RUNS / jid, ignore_errors=True)
        store.delete(jid)

    def sweep():
        for jid in store.expired(config.RETENTION_HOURS):
            remove_job(jid)

    app.state.sweep = sweep

    @app.get("/api/demo-cases")
    def demo_cases():
        out = []
        for r in store.list():
            if r["precomputed"]:
                rep = json.loads((config.RUNS / r["id"] / "report.json").read_text())
                m = rep["meta"]
                out.append({"id": r["id"], "label": m["subject_label"], "mode_label": m["mode_label"],
                            "provenance": m["input_provenance"], "volume_provider": m["volume_provider"]})
        return sorted(out, key=lambda x: x["id"])

    @app.get("/api/reports/{rid}.pdf")
    def report_pdf(rid: str, request: Request, view: str = Query("clin")):
        d = run_dir(rid)
        if view not in VIEWS:
            raise HTTPException(422, "view must be clin or tech")
        pdf = d / f"report_{view}.pdf"
        if not pdf.exists():
            from ..assessment.pdf import render_pdf
            render_pdf(str(request.base_url).rstrip("/"), rid, str(pdf), view)
        return FileResponse(pdf, media_type="application/pdf", filename=f"assessment_{view}.pdf")

    @app.get("/api/reports/{rid}")
    def report(rid: str):
        d = run_dir(rid)
        f = d / "report.json"
        if not f.exists():
            raise HTTPException(404, "Report not ready")
        return JSONResponse(AssessmentReport.model_validate_json(f.read_text()).model_dump(mode="json"))

    @app.get("/api/reports/{rid}/files/{name}")
    def report_file(rid: str, name: str, view: str = Query("clin")):
        d = run_dir(rid)
        if name not in FILES:                                   # allow-list: nothing but these two skull-stripped volumes
            raise HTTPException(404, "Not found")
        if name == "overlay.nii.gz" and view != "tech":
            raise HTTPException(403, "The overlay is only available in the technical view")
        f = (d / name).resolve()
        if d not in f.parents or not f.is_file():
            raise HTTPException(404, "Not found")
        return FileResponse(f, media_type="application/gzip", filename=name)

    @app.get("/api/model-card")
    def model_card():
        if not config.MODEL_CARD.exists():
            raise HTTPException(404, "Model card not found")
        return {"markdown": config.MODEL_CARD.read_text()}

    # --- precomputed demo reports -----------------------------------------------------------------------------
    def import_demo():
        if not config.DEMO.is_dir():
            return
        for src in sorted(p for p in config.DEMO.iterdir() if (p / "report.json").exists()):
            if not ID_RE.match(src.name):
                continue
            dst = config.RUNS / src.name
            if not dst.exists():
                shutil.copytree(src, dst)
            rep = json.loads((dst / "report.json").read_text())
            store.create(src.name, rep["meta"]["mode"], rep["meta"]["subject_label"], precomputed=True, status="done")

    import_demo()
    sweep()

    # --- built frontend (single-page app) -------------------------------------------------------------------------
    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "Not found")
        f = (config.DIST / path).resolve()
        if path and config.DIST.resolve() in f.parents and f.is_file():
            return FileResponse(f)
        index = config.DIST / "index.html"
        if not index.exists():
            raise HTTPException(404, "Frontend not built (run npm run build in frontend/)")
        return FileResponse(index)

    return app


def run():
    import uvicorn
    port = int(__import__("os").environ.get("AD_PORT", "8000"))
    app = create_app()
    app.state.runner.warm()

    def sweeper():
        while True:
            time.sleep(600)
            app.state.sweep()
    threading.Thread(target=sweeper, daemon=True).start()
    uvicorn.run(app, host=config.HOST, port=port, log_level="warning")


if __name__ == "__main__":
    run()
