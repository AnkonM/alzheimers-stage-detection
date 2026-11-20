import json
import os
import pathlib

import pytest

from .conftest import ROOT, axe_violations, start_server

pytestmark = pytest.mark.skipif(not (ROOT / "frontend" / "dist" / "index.html").exists(), reason="frontend not built")


def test_load_demo_toggle_views_and_download_pdf(synthetic_app, browser, tmp_path):
    page = browser.new_page(accept_downloads=True)
    page.goto(synthetic_app + "/")
    assert "Research use only" in page.inner_text("header")
    page.get_by_role("link", name="demo-cn").click()
    page.wait_for_selector("main.sheet")
    assert page.query_selector("table.cnn-table") is None                       # clinician view: no CNN table
    page.get_by_role("button", name="Technical").click()
    page.wait_for_selector("table.cnn-table")
    assert "attribution density, not evidence of disease location" in page.inner_text("[data-panel=cnn]").lower()
    page.get_by_role("button", name="Clinician").click()
    assert page.query_selector("table.cnn-table") is None
    with page.expect_download() as dl:
        page.get_by_role("link", name="Download PDF").click()
    path = tmp_path / "report.pdf"
    dl.value.save_as(path)
    data = path.read_bytes()
    assert data[:5] == b"%PDF-" and len(data) > 10_000


def test_model_card_page(synthetic_app, browser):
    page = browser.new_page()
    page.goto(synthetic_app + "/model-card")
    page.wait_for_selector("article")
    text = page.inner_text("article")
    assert "External validation (OASIS-1, cross-sectional)" in text and "pending" not in text.lower() and "calibrat" not in text.lower()
    assert "Research use only" in page.inner_text("header")


def test_form_validation_and_imputation_notes(synthetic_app, browser):
    page = browser.new_page()
    page.goto(synthetic_app + "/")
    page.wait_for_selector("#f-MMSE")
    page.fill("#f-MMSE", "31")
    assert "Must be a number from 0 to 30" in page.inner_text("#f-MMSE-hint")
    page.fill("#f-MMSE", "10")
    assert "extrapolation" in page.inner_text("#f-MMSE-hint")
    assert "will be imputed with the training median" in page.inner_text("form")
    assert page.query_selector("#f-Hippocampus") is None                         # volumes come from the scan, not the form
    page.check("input[value=CLINICAL_ONLY]")
    assert "fusion model run without scan inputs" in page.inner_text("fieldset").lower()
    assert page.query_selector("#scan") is None
    page.check("input[value=MRI_ONLY]")
    assert page.query_selector("#scan") is not None and page.query_selector("#f-MMSE") is None
    page.check("input[value=FULL]")
    page.set_input_files("#scan", files=[{"name": "notes.txt", "mimeType": "text/plain", "buffer": b"x"}])
    assert "Only NIfTI files are accepted" in page.inner_text("form")
    assert "up to 300 MB" in page.inner_text("form") or "MB" in page.inner_text("form")


@pytest.mark.parametrize("path", ["/", "/model-card", "/report/demo-cn", "/report/demo-cn?view=tech",
                                  "/report/demo-missing-modality", "/report/demo-upload-synthseg?view=tech"])
def test_axe_no_serious_violations(synthetic_app, browser, path):
    page = browser.new_page()
    page.goto(synthetic_app + path)
    page.wait_for_selector("main, article, form")
    page.wait_for_timeout(300)
    assert axe_violations(page) == []


def test_assets_are_self_hosted():
    dist = ROOT / "frontend" / "dist"
    text = "".join(p.read_text(errors="ignore") for p in dist.rglob("*") if p.suffix in (".html", ".css"))
    assert "fonts.googleapis" not in text and "fonts.gstatic" not in text
    assert list((dist / "assets").glob("*.woff2")), "font files are bundled"
    html = (dist / "index.html").read_text()
    assert "http://" not in html and "https://" not in html


@pytest.mark.skipif(not os.environ.get("E2E_NIFTI"), reason="set E2E_NIFTI (and E2E_FORM) to run the real upload e2e")
def test_real_upload_through_the_ui(browser, tmp_path):
    """Real engine, real scan (local data): fill the form, upload, watch progress, open the report, toggle, PDF."""
    form = json.loads(pathlib.Path(os.environ["E2E_FORM"]).read_text())
    p, url = start_server(tmp_path, tmp_path / "nodemo")
    try:
        page = browser.new_page(accept_downloads=True)
        page.goto(url + "/")
        page.wait_for_selector("#scan")
        page.set_input_files("#scan", os.environ["E2E_NIFTI"])
        page.fill("#label", "ui-e2e")
        page.get_by_role("button", name="Show more optional fields").click()
        for k, v in form.items():
            page.select_option(f"#f-{k}", str(v)) if k == "PTGENDER" else page.fill(f"#f-{k}", str(v))
        page.get_by_role("button", name="Run assessment").click()
        page.wait_for_url("**/jobs/*")
        page.wait_for_selector("[data-step=validate]")
        assert "about 2–3 minutes" in page.inner_text("body")
        page.wait_for_selector("[data-step=preprocess]")
        page.wait_for_url("**/report/*", timeout=420_000)
        page.wait_for_selector("main.sheet")
        page.wait_for_function("document.querySelector('[data-viewer] img, [data-viewer] canvas')")
        text = page.inner_text("main")
        assert "new upload" in text and "SynthSeg" in text
        assert page.query_selector("table.cnn-table") is None
        page.get_by_role("button", name="Technical").click()
        page.wait_for_selector("table.cnn-table")
        with page.expect_download(timeout=120_000) as dl:
            page.get_by_role("link", name="Download PDF").click()
        dl.value.save_as(tmp_path / "ui.pdf")
        assert (tmp_path / "ui.pdf").read_bytes()[:5] == b"%PDF-"
        (tmp_path / "ui_report.png").write_bytes(page.screenshot(full_page=True))
        os.makedirs("tests/parity/private/pdf", exist_ok=True)
        (ROOT / "tests/parity/private/pdf/ui_e2e_tech.png").write_bytes((tmp_path / "ui_report.png").read_bytes())
    finally:
        p.terminate()
