import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "reports"
AXE = ROOT / "frontend" / "node_modules" / "axe-core" / "axe.min.js"

pytestmark = pytest.mark.skipif(not (ROOT / "frontend" / "dist" / "index.html").exists(), reason="frontend not built")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(tmp_path, demo_dir, extra_env=None):
    port = free_port()
    env = {**os.environ, "AD_VAR": str(tmp_path / "var"), "AD_DEMO_DIR": str(demo_dir), "AD_PORT": str(port),
           **(extra_env or {})}
    p = subprocess.Popen([sys.executable, "-m", "src.api.app"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(url + "/api/health", timeout=1)
            return p, url
        except OSError:
            time.sleep(0.2)
    p.kill()
    raise RuntimeError("server did not start: " + p.stderr.read().decode()[-500:])


@pytest.fixture(scope="module")
def synthetic_app(tmp_path_factory):
    """The real server with synthetic fixtures as the precomputed demo set (no engine, no ADNI data)."""
    tmp = tmp_path_factory.mktemp("e2e")
    demo = tmp / "demo"
    for name, fx in (("demo-cn", "cn"), ("demo-ad", "ad"), ("demo-missing-modality", "clinical_only"),
                     ("demo-upload-synthseg", "mri_only")):
        (demo / name).mkdir(parents=True)
        shutil.copyfile(FIX / f"{fx}.json", demo / name / "report.json")
    p, url = start_server(tmp, demo)
    yield url
    p.terminate()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        yield b
        b.close()


def axe_violations(page):
    page.add_script_tag(path=str(AXE))
    res = page.evaluate("async () => (await axe.run(document, {runOnly: ['wcag2a','wcag2aa']})).violations")
    return [(v["id"], v["impact"], [n["target"] for n in v["nodes"][:3]]) for v in res
            if v["impact"] in ("serious", "critical")]
