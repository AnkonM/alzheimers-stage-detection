"""Panel-content rules (Addendum 2 + amendment 2026-10-28), checked in the rendered React route.

Needs `npm run build` in frontend/ (skipped otherwise).
- "hippocamp" inside the CNN panel only within the region table;
- CNN panel text never refers to volumes or drivers;
- volumes and drivers panels never refer to the overlay / attribution map;
- no CNN panel (and no region table) in the clinician view.
"""
import re

import pytest

from tests import spa_server

pw = pytest.importorskip("playwright.sync_api")
if not (spa_server.DIST / "index.html").exists():
    pytest.skip("frontend not built", allow_module_level=True)

FIXTURES = ["cn", "lmci_ambiguous", "ad", "mri_only", "clinical_only", "qc_fail"]
VOLUME_WORDS = re.compile(r"volume|icv|percentile|driver|shap|push", re.I)
OVERLAY_WORDS = re.compile(r"overlay|attribution|layercam|\bcam\b|heat ?map|saliency", re.I)


@pytest.fixture(scope="module")
def app():
    srv, url = spa_server.start()
    with pw.sync_playwright() as p:
        b = p.chromium.launch()
        yield b.new_page(), url
        b.close()
    srv.shutdown()


def open_report(app, fx, view):
    page, url = app
    page.goto(f"{url}/report/fixture-{fx}?view={view}")
    page.wait_for_selector("main.sheet")
    return page


@pytest.mark.parametrize("fx", FIXTURES)
@pytest.mark.parametrize("view", ["clin", "tech"])
def test_panel_rules(app, fx, view):
    page = open_report(app, fx, view)
    cnn_outside_table = page.evaluate("""() => { const p = document.querySelector('[data-panel=cnn]');
        if (!p) return ''; const c = p.cloneNode(true); c.querySelectorAll('table.cnn-table').forEach(t => t.remove());
        return c.textContent; }""")
    assert "hippocamp" not in cnn_outside_table.lower()
    assert not VOLUME_WORDS.search(cnn_outside_table)
    for sel in ("[data-panel=volumes]", "[data-panel=drivers]"):
        el = page.query_selector(sel)
        if el:
            assert not OVERLAY_WORDS.search(el.inner_text()), sel
    if view == "clin":
        assert page.query_selector("[data-panel=cnn]") is None
        assert page.query_selector("table.cnn-table") is None
    elif fx in ("cn", "lmci_ambiguous", "ad", "mri_only"):
        assert page.query_selector("table.cnn-table") is not None


def test_banner_and_labels(app):
    page = open_report(app, "clinical_only", "clin")
    assert "Research use only" in page.inner_text("body")
    assert "fusion model run without scan inputs" in page.inner_text("main")
    assert "calibrated" not in page.inner_text("body").lower()


def test_synthseg_report_shows_no_percentile_for_low_agreement(app):
    page = open_report(app, "mri_only", "clin")
    text = page.inner_text("[data-panel=volumes]")
    assert text.count("low agreement with training volumes, no percentile") == 4
    assert "may be unreliable" in page.inner_text("main")


@pytest.mark.parametrize("fx", ["clinical_only", "qc_fail"])
def test_clinical_only_report_has_no_mri_or_brain_age_sections(app, fx):
    page = open_report(app, fx, "tech")
    assert page.query_selector("[data-section=mri]") is None and page.query_selector("[data-section=brain-age]") is None
    assert page.query_selector("[data-viewer]") is None
    keys = page.eval_on_selector_all("main section.box", "els => els.map(e => e.dataset.section)")
    assert keys == ["stage", "ad", "severity", "flags", "quality", "drivers", "caveats"]   # numbered 1..7 by the CSS counter
