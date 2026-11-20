"""Figure 5: report card, clinician view, synthetic lmci_ambiguous case (re-render of F0031).

Fixture: tests/make_report_fixtures.py FIX["lmci_ambiguous"] (all values invented), validated against
src/assessment/schema.AssessmentReport and written only to figures/out/build/.
Renderer: the app's React report route, served from the main checkout's built frontend/dist (read-only) by a
local static server modelled on tests/spa_server.py; printed by Chromium (playwright) as in src/assessment/pdf.py.
Render options (DOM edits after the page is ready; no app code changed on any branch):
  1. header provenance "ADNI demo case" -> "synthetic case";
  2. the header date (meta.created_at) is removed (no dates in paper figures);
  3. "Percentile among CN training subjects (n=569)" -> the true reference n, read from
     analysis/inference_refs/results/reference_ranges.json cn_volume_quantiles[*].n (percentiles are computed
     from CN training subjects only: analysis/inference_refs/build_refs.py, cn = lab == 0).
Crop: if the card is taller than 1.4x its width, sections 6, 7 and 9 are collapsed (numbering kept) and section 8
is moved into the emptied right column (DOM move at render time; CSS counters keep it numbered 8). Run with LYP/.venv-app.
"""
import hashlib
import http.server
import json
import os
import re
import subprocess
import sys
import threading

from _prov import MAIN, OUT, WT, Prov

sys.path.insert(0, WT)
P = Prov("fig5_report_card", __file__)
BUILD = os.path.join(OUT, "build")
os.makedirs(BUILD, exist_ok=True)
DIST = os.path.join(MAIN, "frontend", "dist")
WIDTH_IN, DPI, MIN_PT = 7.16, 300, 8.0

# ---- fixture (invented values) -----------------------------------------------------------------
P.file("tests/make_report_fixtures.py")
src_fx = open(os.path.join(WT, "tests/make_report_fixtures.py")).read()
P.check("fixture generator declares invented values", "All values are invented" in src_fx, True, True)
from tests.make_report_fixtures import FIX  # noqa: E402  (module import writes nothing; writing is under __main__)
from src.assessment.schema import AssessmentReport  # noqa: E402

rep = AssessmentReport.model_validate(FIX["lmci_ambiguous"])
fx = rep.model_dump(mode="json")
fx_path = os.path.join(BUILD, "lmci_ambiguous.json")
open(fx_path, "w").write(json.dumps(fx, indent=2) + "\n")
date = fx["meta"]["created_at"][:10]

# ---- percentile reference population -----------------------------------------------------------
refs = json.loads(P.file("analysis/inference_refs/results/reference_ranges.json"))
ns = {k: v["n"] for k, v in refs["cn_volume_quantiles"].items()}
P.check("percentile reference n is the same for all 6 ratios", len(set(ns.values())) == 1 and len(ns) == 6, 6, ns)
N_CN = next(iter(ns.values()))
P.check("percentile reference population = CN training subjects, n = 133", N_CN == 133, 133, N_CN)
br = P.file("analysis/inference_refs/build_refs.py").decode().splitlines()
P.check("build_refs.py:66 selects CN only (cn = lab == 0)", br[65].strip() == "cn = lab == 0", "cn = lab == 0", br[65].strip())
P.check("build_refs.py:70 stores cn_volume_quantiles with the CN n", 'out["cn_volume_quantiles"][r] = {"n": int(len(x))' in br[69],
        True, br[69].strip()[:80])
rb = P.file("src/assessment/report_builder.py").decode().splitlines()
P.check("report_builder.py:89-91 _percentile reads cn_volume_quantiles", "def _percentile" in rb[88] and "cn_volume_quantiles" in rb[89],
        True, [rb[88].strip(), rb[89].strip()[:60]])
sch = P.file("src/assessment/schema.py").decode().splitlines()
ln = [i + 1 for i, s in enumerate(sch) if s.strip() == "n: int = 569"]
P.check("schema.py ReferencePopulation default n = 569 (the source of the wrong label)", len(ln) == 1, "one line", ln)
P.d["percentile_answer"] = {
    "computed_from": "CN training subjects only", "n": N_CN,
    "code": ["analysis/inference_refs/build_refs.py:66 (cn = lab == 0)", "analysis/inference_refs/build_refs.py:70 (cn_volume_quantiles)",
             "src/assessment/report_builder.py:89-91 (_percentile reads cn_volume_quantiles)"],
    "wrong_label_source": [f"src/assessment/schema.py:{ln[0]} (ReferencePopulation.n = 569, all training subjects)",
                           "frontend/src/report/MriPanel.tsx:24 (renders meta.reference_population.n)"]}
P.check("fixture label n is the schema default 569", fx["meta"]["reference_population"]["n"] == 569, 569,
        fx["meta"]["reference_population"]["n"])

# ---- serve dist + our fixture ------------------------------------------------------------------
h = hashlib.sha256()
for root, _, files in sorted(os.walk(DIST)):
    for f in sorted(files):
        h.update(f.encode()); h.update(open(os.path.join(root, f), "rb").read())
P.d["inputs"].append({"path": "frontend/dist/** (main checkout, gitignored build of 2026-11-03)", "sha256": h.hexdigest()})


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=DIST, **k)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/fixtures/lmci_ambiguous.json":
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(open(fx_path, "rb").read()); return
        if path.startswith("/fixtures/") or path.startswith("/api/"):
            self.send_error(404); return
        if not os.path.isfile(os.path.join(DIST, path.lstrip("/"))):
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, *a):
        pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{srv.server_address[1]}/report/fixture-lmci_ambiguous?view=clin&print=1"

EDIT_JS = r"""([date, ncn]) => {
  const main = document.querySelector('main.sheet');
  const leaves = [...main.querySelectorAll('*')].filter(e => e.children.length === 0);
  let hdr = 0, pct = 0;
  for (const e of leaves) {
    const t = e.textContent;
    if (t.includes('ADNI demo case')) {
      e.textContent = t.replace('ADNI demo case', 'synthetic case').replace(' · ' + date, '');
      hdr++;
    }
    if (t.startsWith('Percentile among CN training subjects (n=')) {
      e.textContent = t.replace(/\(n=\d+\)/, '(n=' + ncn + ')');
      pct++;
    }
  }
  return {hdr, pct, text: main.innerText};
}"""
CROP_CSS = """
  section[data-section="mri"], section[data-section="quality"], section[data-section="caveats"] {
    visibility: hidden !important; height: 0 !important; min-height: 0 !important; margin: 0 !important;
    padding: 0 !important; border: 0 !important; overflow: hidden !important; }
  .sheet { width: 700px !important; }
  .cols2 { grid-template-columns: 5fr 6fr !important; }"""  # stacked single column: 700 CSS px at 7.16 in keeps 11 px text >= 8 pt
BASE_CSS = "body{margin:0!important;background:#fff!important} .sheet{margin:0!important;box-shadow:none!important;min-height:0!important}"
MOVE_JS = """() => { const d = document.querySelector('section[data-section="drivers"]');
  const right = document.querySelector('.cols2').children[1]; right.appendChild(d);
  return d.parentElement === right; }"""
FONT_JS = """() => { let m = 1e9; const main = document.querySelector('main.sheet');
  for (const e of main.querySelectorAll('*')) { const st = getComputedStyle(e);
    if (st.visibility === 'hidden' || st.display === 'none') continue;
    const own = [...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim());
    if (own && e.getClientRects().length) m = Math.min(m, parseFloat(st.fontSize)); } return m; }"""

from playwright.sync_api import sync_playwright  # noqa: E402

with sync_playwright() as pw:
    b = pw.chromium.launch()

    def load(ctx):
        pg = ctx.new_page()
        pg.emulate_media(media="screen")
        pg.goto(URL, wait_until="networkidle")
        try:
            pg.wait_for_function("window.__reportReady === true", timeout=20000)
            ready = "__reportReady"
        except Exception:
            pg.wait_for_selector('section[data-section="drivers"]', timeout=20000)
            ready = "drivers section present (__reportReady not set for fixtures)"
        pg.add_style_tag(content=BASE_CSS)
        res = pg.evaluate(EDIT_JS, [date, N_CN])
        return pg, ready, res

    ctx = b.new_context(viewport={"width": 1000, "height": 1400})
    pg, ready, res = load(ctx)
    P.note(f"page readiness: {ready}")
    P.check("header 'ADNI demo case' replaced exactly once", res["hdr"] == 1, 1, res["hdr"])
    P.check("percentile label replaced exactly once", res["pct"] == 1, 1, res["pct"])
    P.check("'synthetic case' in header", "synthetic case" in res["text"], True, "synthetic case" in res["text"])
    P.check("no 'ADNI demo case' left", "ADNI demo case" not in res["text"], True, "ADNI demo case" not in res["text"])
    P.check("no date (YYYY-MM-DD) in the card text", not re.search(r"\d{4}-\d{2}-\d{2}", res["text"]), None,
            re.findall(r"\d{4}-\d{2}-\d{2}", res["text"]))
    P.check(f"percentile label reads n={N_CN}", f"Percentile among CN training subjects (n={N_CN})" in res["text"], True, True)
    box = pg.locator("main.sheet").bounding_box()
    ratio = box["height"] / box["width"]
    crop = ratio > 1.4
    P.d["crop"] = {"full_height_over_width": round(ratio, 3), "cropped": crop,
                   "kept_sections": "1-5 and 8 (6 MRI, 7 Input quality, 9 Caveats collapsed; numbering kept; 8 moved to the right column)" if crop else "all"}
    if crop:
        pg.add_style_tag(content=CROP_CSS)
        P.check("section 8 moved into the right column", pg.evaluate(MOVE_JS), True, True)
        txt = pg.locator("main.sheet").inner_text()
        over = pg.evaluate("""() => { const s = document.querySelector('section[data-section="drivers"]').getBoundingClientRect();
          let w = 0; for (const e of document.querySelectorAll('section[data-section="drivers"] *')) {
            const r = e.getBoundingClientRect(); if (r.width) w = Math.max(w, r.right - s.right); } return w; }""")
        P.check("no element overflows the section 8 box (px)", over <= 0.5, "<= 0.5", round(over, 1))
        P.check("section 8 still numbered 8 after the move",
                pg.evaluate("""() => getComputedStyle(document.querySelector('section[data-section="drivers"] h2'), '::before').content""") != "none",
                True, True)
    box = pg.locator("main.sheet").bounding_box()
    css_w, css_h = box["width"], box["height"]
    min_px = pg.evaluate(FONT_JS)
    min_pt = min_px * WIDTH_IN * 72 / css_w
    P.d["font"] = {"min_css_px": min_px, "card_css_width_px": css_w, "min_pt_at_7.16in": round(min_pt, 2),
                   "meets_8pt": min_pt >= MIN_PT}
    pdf_raw = os.path.join(BUILD, "fig5_raw.pdf")
    pg.pdf(path=pdf_raw, width=f"{css_w}px", height=f"{css_h + 2}px", print_background=True,
           margin={"top": "0", "bottom": "0", "left": "0", "right": "0"}, page_ranges="1")
    ctx.close()

    scale = WIDTH_IN * DPI / css_w
    ctx2 = b.new_context(viewport={"width": 1000, "height": 1400}, device_scale_factor=scale)
    pg2, _, res2 = load(ctx2)
    P.check("PNG pass: same edits applied", res2["hdr"] == 1 and res2["pct"] == 1, [1, 1], [res2["hdr"], res2["pct"]])
    if crop:
        pg2.add_style_tag(content=CROP_CSS)
        pg2.evaluate(MOVE_JS)
    png = os.path.join(OUT, "fig5_report_card.png")
    pg2.locator("main.sheet").screenshot(path=png)
    ctx2.close()
    b.close()
srv.shutdown()

pdf = os.path.join(OUT, "fig5_report_card.pdf")
subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite", "-dOmitInfoDate", "-dOmitID", "-dOmitXMP",
                f"-sOutputFile={pdf}", pdf_raw], check=True)
pb = open(pdf, "rb").read()
P.check("PDF has no CreationDate/ModDate", b"CreationDate" not in pb and b"ModDate" not in pb, True,
        [b"CreationDate" in pb, b"ModDate" in pb])
P.check("PDF is vector (text kept as fonts)", b"/Font" in pb, True, b"/Font" in pb)
P.note("PDF: Chromium print (vector) re-written by Ghostscript pdfwrite with -dOmitInfoDate -dOmitID -dOmitXMP.")
P.write([pdf, png])
print(json.dumps({"crop": P.d["crop"], "font": P.d["font"]}))
