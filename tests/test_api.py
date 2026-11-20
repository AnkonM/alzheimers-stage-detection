import gzip
import json
import pathlib
import re
import shutil
import struct
import time

import pytest
from fastapi.testclient import TestClient

from src.api import app as api_app
from src.api import config

FIX = pathlib.Path(__file__).parent / "fixtures" / "reports"
PTID = re.compile(r"\b\d{3}_S_\d{4}\b")
FORM = {"AGE": 70, "PTGENDER": "Male", "CDRSB": 1.5, "MMSE": 27, "ADAS13": 15}


def nifti_bytes(gz=True):
    hdr = struct.pack("<i", 348) + b"\0" * 344 + b"\0" * 4 + b"\0" * 16
    return gzip.compress(hdr) if gz else hdr


class FakeEngine:
    """Stands in for the real engine: writes a fixture report and dummy skull-stripped volumes."""
    fail = None

    def __init__(self, run_root=None):
        pass

    def predict(self, mode, clinical_values, native_path, subject_label, report_id, progress):
        if FakeEngine.fail:
            raise FakeEngine.fail
        for s in ("validate", "model", "report"):
            progress(s)
        d = config.RUNS / report_id
        d.mkdir(parents=True, exist_ok=True)
        fx = {"FULL": "ad", "CLINICAL_ONLY": "clinical_only", "MRI_ONLY": "mri_only"}[mode]
        shutil.copyfile(FIX / f"{fx}.json", d / "report.json")
        (d / "brain.nii.gz").write_bytes(b"brain")
        (d / "overlay.nii.gz").write_bytes(b"overlay")

    def close(self):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    for name, sub in (("VAR", ""), ("RUNS", "runs"), ("UPLOADS", "uploads"), ("DEMO", "demo"), ("DB", "app.db")):
        monkeypatch.setattr(config, name, tmp_path / sub if sub else tmp_path)
    monkeypatch.setattr(config, "DIST", tmp_path / "dist")
    FakeEngine.fail = None
    app = api_app.create_app(engine_factory=FakeEngine)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        c.app_ = app
        yield c


def post(client, mode="FULL", name="scan.nii.gz", data=None, clinical=FORM, label=""):
    files = {"scan": (name, data if data is not None else nifti_bytes(), "application/octet-stream")}
    form = {"mode": mode, "subject_label": label}
    if clinical is not None:
        form["clinical_json"] = json.dumps(clinical)
    return client.post("/api/assessments", data=form, files=files if mode != "CLINICAL_ONLY" else None)


def wait_done(client, jid):
    for _ in range(100):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_binds_loopback_and_rejects_foreign_host(client, monkeypatch):
    assert config.HOST == "127.0.0.1"
    assert client.get("/api/health", headers={"host": "evil.example"}).status_code == 403
    assert client.get("/api/health").status_code == 200


def test_full_flow_and_no_ids_or_paths_in_responses(client):
    r = post(client)
    assert r.status_code == 202
    jid = r.json()["job_id"]
    j = wait_done(client, jid)
    assert j["status"] == "done" and j["report_id"] == jid and j["plan"][0] == "validate"
    bodies = [client.get(u).text for u in (f"/api/jobs/{jid}", "/api/jobs", f"/api/reports/{jid}", "/api/form-spec",
                                           "/api/demo-cases")]
    for b in bodies:
        assert not PTID.search(b) and "/home/" not in b and "/tmp/" not in b and "\\" not in b.replace("\\n", "")
    assert not list((config.UPLOADS).glob("*"))                 # upload removed once processed


def test_upload_type_content_and_size_limits(client, monkeypatch):
    assert post(client, name="scan.dcm").status_code == 415
    assert post(client, name="scan.nii.gz", data=b"not a nifti at all").status_code == 415
    assert post(client, name="x.zip").status_code == 415
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    assert post(client, data=gzip.compress(b"\0" * 5000 + struct.pack("<i", 348))).status_code in (413, 415)
    big = struct.pack("<i", 348) + b"\0" * 5000
    assert post(client, name="a.nii", data=big).status_code == 413
    assert not list(config.UPLOADS.glob("*/*"))                  # rejected uploads leave nothing behind


def test_request_validation(client):
    assert post(client, clinical={**FORM, "MMSE": 31}).status_code == 422
    assert post(client, clinical={"AGE": 70, "PTGENDER": "Male"}).status_code == 422          # required inputs missing
    assert post(client, clinical={**FORM, "Hippocampus": 1}).status_code == 422
    assert post(client, label="subject " + "123" + "_S_" + "4567").status_code == 422
    assert post(client, label="../../etc").status_code == 422
    assert post(client, mode="NOPE").status_code == 422
    assert client.post("/api/assessments", data={"mode": "FULL", "clinical_json": json.dumps(FORM)}).status_code == 422
    assert post(client, mode="MRI_ONLY", clinical={"AGE": 70}).status_code == 422


def test_file_endpoints_allowlist_traversal_and_overlay_rule(client):
    jid = post(client).json()["job_id"]
    wait_done(client, jid)
    assert client.get(f"/api/reports/{jid}/files/brain.nii.gz").status_code == 200
    assert client.get(f"/api/reports/{jid}/files/overlay.nii.gz").status_code == 403
    assert client.get(f"/api/reports/{jid}/files/overlay.nii.gz?view=clin").status_code == 403
    assert client.get(f"/api/reports/{jid}/files/overlay.nii.gz?view=tech").status_code == 200
    for bad in ("report.json", "scan.nii.gz", "..%2freport.json", "%2e%2e/%2e%2e/etc/passwd", "brain.nii.gz%00"):
        assert client.get(f"/api/reports/{jid}/files/{bad}").status_code in (404, 422), bad
    for rid in ("..", "..%2f..%2fetc", "a.b", "x" * 80, "%2e%2e"):
        for u in (f"/api/reports/{rid}", f"/api/reports/{rid}/files/brain.nii.gz", f"/api/jobs/{rid}"):
            assert client.get(u).status_code in (404, 405), (rid, u)


def test_delete_job_and_retention(client):
    jid = post(client).json()["job_id"]
    wait_done(client, jid)
    assert (config.RUNS / jid).exists()
    assert client.delete(f"/api/jobs/{jid}").status_code == 204
    assert not (config.RUNS / jid).exists() and client.get(f"/api/jobs/{jid}").status_code == 404
    old = post(client).json()["job_id"]
    wait_done(client, old)
    store = client.app_.state.store
    store._run("update jobs set created_at=? where id=?", time.time() - 3600 * (config.RETENTION_HOURS + 1), old)
    keep = post(client).json()["job_id"]
    wait_done(client, keep)
    client.app_.state.sweep()
    assert not (config.RUNS / old).exists() and (config.RUNS / keep).exists()


def test_precomputed_demo_reports_are_listed_and_protected(tmp_path, monkeypatch):
    for name, sub in (("VAR", ""), ("RUNS", "runs"), ("UPLOADS", "uploads"), ("DEMO", "demo"), ("DB", "app.db")):
        monkeypatch.setattr(config, name, tmp_path / sub if sub else tmp_path)
    d = tmp_path / "demo" / "demo-cn"
    d.mkdir(parents=True)
    shutil.copyfile(FIX / "cn.json", d / "report.json")
    app = api_app.create_app(engine_factory=FakeEngine)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        assert [x["id"] for x in c.get("/api/demo-cases").json()] == ["demo-cn"]
        assert c.get("/api/reports/demo-cn").status_code == 200
        assert c.delete("/api/jobs/demo-cn").status_code == 403
        app.state.store._run("update jobs set created_at=0 where id='demo-cn'")
        app.state.sweep()
        assert (config.RUNS / "demo-cn").exists()               # retention never removes the precomputed set


def test_errors_are_sanitised(client):
    FakeEngine.fail = ValueError("failed at /home/user/data/scan_" + "123" + "_S_" + "4567" + ".nii")
    jid = post(client).json()["job_id"]
    j = wait_done(client, jid)
    assert j["status"] == "failed" and "/home" not in j["error"] and not PTID.search(j["error"])
    FakeEngine.fail = RuntimeError("secret internals /home/x")
    j = wait_done(client, post(client).json()["job_id"])
    assert j["error"] == "Processing failed; see the server log"


def test_clinical_only_needs_no_scan(client):
    r = post(client, mode="CLINICAL_ONLY")
    assert r.status_code == 202 and wait_done(client, r.json()["job_id"])["status"] == "done"
