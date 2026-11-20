#!/usr/bin/env python3
"""Write docs/validation/oasis_results_tables.md from the committed aggregate JSON (no number typed by hand).

    python tools/build_oasis_tables.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import oasis_render  # noqa: E402


def main():
    (ROOT / "docs" / "validation" / "oasis_results_tables.md").write_text(oasis_render.tables_markdown())
    print("wrote docs/validation/oasis_results_tables.md")


if __name__ == "__main__":
    main()
