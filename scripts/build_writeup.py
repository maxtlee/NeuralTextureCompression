"""Render the progression writeup HTML to PDF.

Usage:  .venv/bin/python scripts/build_writeup.py
"""

import sys
from pathlib import Path

from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "writeup" / "progression.html"
OUTPUT = ROOT / "writeup" / "progression.pdf"


def main() -> int:
    if not SOURCE.exists():
        print(f"missing source: {SOURCE}")
        return 1
    HTML(str(SOURCE)).write_pdf(str(OUTPUT))
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
