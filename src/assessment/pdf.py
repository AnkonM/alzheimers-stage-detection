"""PDF of a report: Playwright prints the same React route the browser shows (A4), so web and PDF share one source."""
import pathlib

FOOTER = ('<div style="font-size:8px;width:100%;text-align:center;color:#444;">'
          'Research use only — not a diagnostic device &nbsp;·&nbsp; page <span class="pageNumber"></span> of '
          '<span class="totalPages"></span></div>')


def render_pdf(base_url: str, report_id: str, out_path: str, view: str = "clin", timeout_ms: int = 60000) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        try:
            pg = b.new_page(viewport={"width": 794, "height": 1123})
            pg.goto(f"{base_url}/report/{report_id}?print=1&view={view}", wait_until="networkidle")
            pg.wait_for_function("window.__reportReady === true", timeout=timeout_ms)
            pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            pg.pdf(path=out_path, format="A4", print_background=True, display_header_footer=True,
                   header_template="<div></div>", footer_template=FOOTER,
                   margin={"top": "12mm", "bottom": "16mm", "left": "12mm", "right": "12mm"})
        finally:
            b.close()
    return out_path
