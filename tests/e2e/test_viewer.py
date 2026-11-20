"""NiiVue viewer in the report's MRI panel: no runaway zoom, no leaked instances, no overlay outside the technical view.

Uses a synthetic NIfTI pair (no ADNI data) served by the real server, in a software-WebGL Chromium. Regression tests for
two bugs: (1) a new NiiVue instance on every view toggle, never disposed, each resizing the shared canvas from its own
bitmap size (exponential growth at devicePixelRatio > 1); (2) stale instances redrawing the overlay in the clinician view.
"""
import json
import pathlib
import re
import shutil

import numpy as np
import pytest

from .conftest import FIX, ROOT, start_server

pytestmark = pytest.mark.skipif(not (ROOT / "frontend" / "dist" / "index.html").exists(), reason="frontend not built")

INIT = """(() => {
  const RO = window.ResizeObserver; window.__roActive = 0; window.__resizeListeners = 0;
  window.ResizeObserver = class extends RO {
    observe(t, o) { if (!this.__on) { this.__on = true; window.__roActive++; } super.observe(t, o); }
    disconnect() { if (this.__on) { this.__on = false; window.__roActive--; } super.disconnect(); } };
  const add = window.addEventListener.bind(window), rem = window.removeEventListener.bind(window), seen = new Set();
  window.addEventListener = (t, f, o) => { if (t === 'resize' && !seen.has(f)) { seen.add(f); window.__resizeListeners++; } return add(t, f, o); };
  window.removeEventListener = (t, f, o) => { if (t === 'resize' && seen.delete(f)) window.__resizeListeners--; return rem(t, f, o); };
})()"""

STATE = """() => {
  const c = document.querySelector('[data-viewer] canvas');
  if (!c || !c.__nv) return null;
  const nv = c.__nv, r = c.getBoundingClientRect(), box = c.parentElement.parentElement.getBoundingClientRect();
  return { w: Math.round(r.width), h: Math.round(r.height), cw: c.width, ch: c.height, boxH: Math.round(box.height),
           canvases: document.querySelectorAll('[data-viewer] canvas').length, nVolumes: nv.volumes.length,
           colormaps: nv.volumes.map(v => v.colormap), zoom: nv.volScaleMultiplier,
           pan: Array.from(nv.scene.pan2Dxyzmm), ro: window.__roActive, listeners: window.__resizeListeners,
           docH: document.documentElement.scrollHeight };
}"""


def synthetic_case(dest: pathlib.Path):
    import nibabel as nib
    rng = np.random.default_rng(0)
    g = np.mgrid[:40, :48, :40].astype(float)
    r = np.sqrt(((g[0] - 20) / 15) ** 2 + ((g[1] - 24) / 19) ** 2 + ((g[2] - 20) / 15) ** 2)
    brain = np.where(r < 1, 0.6 + 0.2 * rng.random(r.shape) - 0.3 * r, 0).astype(np.float32)
    blobs = np.zeros_like(brain)
    for x, y, z in ((12, 14, 20), (28, 14, 20), (20, 30, 14), (20, 30, 26)):
        blobs += np.exp(-(((g[0] - x) ** 2 + (g[1] - y) ** 2 + (g[2] - z) ** 2) / 20)).astype(np.float32)
    aff = np.diag([4.0, 4.0, 4.0, 1.0])
    dest.mkdir(parents=True)
    nib.save(nib.Nifti1Image(brain, aff), dest / "brain.nii.gz")
    nib.save(nib.Nifti1Image(blobs * (brain > 0), aff), dest / "overlay.nii.gz")
    rep = json.loads((FIX / "ad.json").read_text())
    rep["meta"]["report_id"] = "demo-syn"
    rep["meta"]["subject_label"] = "demo-syn"
    (dest / "report.json").write_text(json.dumps(rep))


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("viewer")
    synthetic_case(tmp / "demo" / "demo-syn")
    p, url = start_server(tmp, tmp / "demo")
    yield url
    p.terminate()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        yield b
        b.close()


def open_page(browser, url, dpr, path="/report/demo-syn?view=clin"):
    ctx = browser.new_context(viewport={"width": 1000, "height": 1000}, device_scale_factor=dpr)
    page = ctx.new_page()
    page.add_init_script(INIT)
    reqs = []
    page.on("request", lambda r: reqs.append(r.url.split("/files/")[-1]) if "/files/" in r.url else None)
    page.goto(url + path)
    page.wait_for_function("window.__reportReady === true", timeout=30000)
    return page, reqs


def settle(page, volumes):
    page.wait_for_function(f"document.querySelector('[data-viewer] canvas')?.__nv?.volumes.length === {volumes}", timeout=15000)
    page.wait_for_timeout(250)                                 # let any (buggy) resize feedback show itself
    return page.evaluate(STATE)


@pytest.mark.parametrize("dpr", [1, 2])
def test_ten_view_toggles_leave_canvas_and_zoom_unchanged(app, browser, dpr):
    page, reqs = open_page(browser, app, dpr)
    before = settle(page, 1)
    assert before["canvases"] == 1 and before["ro"] == 1 and before["listeners"] == 1
    assert before["boxH"] == 180 and before["h"] == 180                  # fixed container: canvas CSS size is the box
    for i in range(12):
        page.get_by_role("button", name="Technical").click()
        tech = settle(page, 2)
        assert tech["colormaps"][1] == "viridis" and tech["w"] == before["w"] and tech["h"] == before["h"], (i, tech)
        page.get_by_role("button", name="Clinician").click()
        clin = settle(page, 1)
        assert clin["nVolumes"] == 1 and clin["colormaps"] == ["gray"], (i, clin)       # no overlay layer left behind
    after = settle(page, 1)
    for k in ("w", "h", "cw", "ch", "boxH", "canvases", "colormaps", "zoom", "pan", "ro", "listeners", "docH"):
        assert after[k] == before[k], (k, before[k], after[k])
    assert after["ro"] == 1 and after["listeners"] == 1                    # one NiiVue instance, nothing leaked
    assert sum(r.startswith("brain.nii.gz") for r in reqs) == 1            # the base volume is never reloaded
    page.context.close()


def test_overlay_is_never_requested_in_the_clinician_view(app, browser):
    page, reqs = open_page(browser, app, 1)
    settle(page, 1)
    page.set_viewport_size({"width": 1100, "height": 900})                 # stale redraws would show here
    page.wait_for_timeout(500)
    assert not [r for r in reqs if r.startswith("overlay")]
    page.get_by_role("button", name="Technical").click()
    settle(page, 2)
    assert [r for r in reqs if r.startswith("overlay")] and all("view=tech" in r for r in reqs if r.startswith("overlay"))
    n = len(reqs)
    page.get_by_role("button", name="Clinician").click()
    settle(page, 1)
    page.set_viewport_size({"width": 1000, "height": 1000})
    page.wait_for_timeout(500)
    assert len(reqs) == n                                                 # nothing fetched when going back
    fresh, freqs = open_page(browser, app, 1, "/report/demo-syn?view=clin")   # a clinician-view load starts clean
    settle(fresh, 1)
    assert not [r for r in freqs if r.startswith("overlay")]
    page.context.close()
    fresh.context.close()


@pytest.mark.parametrize("view,volumes", [("clin", 1), ("tech", 2)])
def test_print_mode_freezes_one_image_and_leaks_nothing(app, browser, view, volumes):
    page, reqs = open_page(browser, app, 2, f"/report/demo-syn?print=1&view={view}")
    page.wait_for_selector("[data-viewer] img")
    st = page.evaluate("""() => { const i = document.querySelector('[data-viewer] img'), b = i.parentElement.getBoundingClientRect(),
        c = document.querySelector('[data-viewer] canvas'); return { imgs: document.querySelectorAll('[data-viewer] img').length,
        w: Math.round(b.width), h: Math.round(b.height), natural: [i.naturalWidth, i.naturalHeight], nVolumes: c.__nv.volumes.length,
        canvasVisible: getComputedStyle(c.parentElement).visibility, ro: window.__roActive, listeners: window.__resizeListeners }; }""")
    assert st["imgs"] == 1 and st["h"] == 180 and st["nVolumes"] == volumes and st["canvasVisible"] == "hidden"
    assert st["ro"] == 1 and st["listeners"] == 1 and st["natural"][0] > 0
    assert bool([r for r in reqs if r.startswith("overlay")]) == (view == "tech")
    page.context.close()


def test_reload_in_the_middle_of_toggling(app, browser):
    page, _ = open_page(browser, app, 2)
    base = settle(page, 1)
    for expected in ("tech", "clin", "tech"):
        page.get_by_role("button", name="Technical" if expected == "tech" else "Clinician").click()
        page.reload()                                                      # before the viewer has settled
        page.wait_for_function("window.__reportReady === true", timeout=30000)
        assert f"view={expected}" in page.url
        st = settle(page, 2 if expected == "tech" else 1)
        assert st["ro"] == 1 and st["listeners"] == 1 and st["canvases"] == 1
        assert (st["w"], st["h"], st["cw"], st["ch"], st["zoom"]) == (base["w"], base["h"], base["cw"], base["ch"], base["zoom"])
        if expected == "clin":
            assert st["colormaps"] == ["gray"]
    page.context.close()
