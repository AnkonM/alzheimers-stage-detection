"""Shared house style for every paper figure (matplotlib).

- Widths: single column 3.5 in, double column 7.16 in.
- One font family everywhere (DejaVu Sans), no font smaller than 8 pt at final size (enforced by check_fonts).
- Okabe-Ito colourblind-safe palette; no titles inside figures (captions live in the paper).
- save(): vector PDF (TrueType fonts embedded, no CreationDate) + 300 dpi PNG preview (no date metadata).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

SINGLE = 3.5
DOUBLE = 7.16
MIN_PT = 8.0
DPI = 300

OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
# Order used for categorical series.
CYCLE = [OKABE_ITO[k] for k in ("blue", "orange", "green", "vermillion", "purple", "skyblue", "yellow", "black")]
# 12 categories (atlas regions): Okabe-Ito (7, black excluded) + 5 Paul Tol 'muted' colours; 'other' is grey.
ATLAS_12 = [OKABE_ITO[k] for k in ("vermillion", "orange", "skyblue", "purple", "green", "yellow", "blue")] + \
           ["#882255", "#44AA99", "#DDCC77", "#117733", "#AA4499"]
OTHER_GREY = "#BBBBBB"
SEQ_CMAP = "Blues"  # sequential, colourblind-safe

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.title_fontsize": 8, "figure.titlesize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": DPI, "figure.dpi": 100, "legend.frameon": False,
}
plt.rcParams.update(RC)


def panel_label(ax, letter, x=-0.02, y=1.0):
    """Small bold panel letter in axes coordinates (the only text allowed on image panels)."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="right")


def check_fonts(fig):
    """Fail if any visible, non-empty text is smaller than MIN_PT at final size."""
    bad = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-9:
            bad.append((t.get_text()[:30], t.get_fontsize()))
    if bad:
        raise AssertionError(f"text smaller than {MIN_PT} pt: {bad[:5]}")
    return True


def save(fig, out_dir, name):
    """Write <name>.pdf (vector) and <name>.png (300 dpi). Returns the two paths."""
    check_fonts(fig)
    os.makedirs(out_dir, exist_ok=True)
    pdf, png = os.path.join(out_dir, f"{name}.pdf"), os.path.join(out_dir, f"{name}.png")
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    fig.savefig(png, dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    return pdf, png
