"""Render the writeup HTML files to PDF.

Renders both the running progression notes and the assignment submission.

Usage:  .venv/bin/python scripts/build_writeup.py [name ...]
        .venv/bin/python scripts/build_writeup.py submission
"""

import sys
from pathlib import Path

from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "progression": ROOT / "writeup" / "progression.html",
    "submission": ROOT / "writeup" / "submission.html",
}


def main() -> int:
    names = sys.argv[1:] or list(SOURCES)
    for name in names:
        source = SOURCES.get(name)
        if source is None:
            print(f"unknown writeup {name!r}; choose from {list(SOURCES)}")
            return 1
        if not source.exists():
            print(f"missing source: {source}")
            return 1
        output = source.with_suffix(".pdf")
        HTML(str(source)).write_pdf(str(output))
        print(f"wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
