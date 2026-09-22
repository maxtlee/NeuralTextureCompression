"""Fetch the textures sourced for the "own textures" experiment.

Downloads CC0 material color maps from ambientCG, extracts the 1K color map,
resizes it to 512x512, and writes it to assets/own/<name>.png plus a CREDITS
file. Dry-run by default; pass --execute to actually download.

Source: https://ambientcg.com  (all assets are CC0, no attribution required;
credit is still recorded in assets/own/CREDITS.md).

Usage:
  python scripts/fetch_own_textures.py             # show the plan
  python scripts/fetch_own_textures.py --execute   # download
"""

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "own"
GET = "https://ambientcg.com/get?file={asset}_1K-JPG.zip"
SOURCE = "https://ambientcg.com/view?id={asset}"

# A spread of frequency content and structure, on purpose.
TEXTURES = [
    ("gravel", "Ground037", "fine stochastic high-frequency detail"),
    ("grass", "Grass004", "fine high-frequency organic detail"),
    ("concrete", "Concrete034", "smooth, low-frequency, low contrast"),
    ("wood", "Wood062", "directional mid-frequency grain"),
    ("marble", "Marble016", "smooth with strong multi-scale veins"),
    ("fabric", "Fabric066", "fine repetitive weave"),
    ("rock", "Rock030", "multi-scale natural detail"),
    ("paving", "PavingStones070", "structured, sharp edges"),
]


def fetch_color_map(asset: str, size: int = 512) -> Image.Image:
    url = GET.format(asset=asset)
    request = urllib.request.Request(url, headers={"User-Agent": "ntc-texture-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = next(n for n in archive.namelist()
                      if n.endswith("_Color.jpg") or n.endswith("_Color.png"))
        with archive.open(member) as handle:
            image = Image.open(io.BytesIO(handle.read())).convert("RGB")
    return image.resize((size, size), Image.LANCZOS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="actually download")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    print(f"{'name':10s} {'asset':16s} description")
    for name, asset, note in TEXTURES:
        print(f"{name:10s} {asset:16s} {note}")
    if not args.execute:
        print("\ndry run; pass --execute to download")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    for name, asset, note in TEXTURES:
        image = fetch_color_map(asset, args.size)
        image.save(OUT / f"{name}.png")
        print(f"saved assets/own/{name}.png  ({SOURCE.format(asset=asset)})")

    credits = [
        "# Texture credits",
        "",
        "Textures in this folder are color maps from ambientCG, released under CC0",
        "(public domain dedication, https://ambientcg.com/license). No attribution is",
        "required; sources are listed for provenance.",
        "",
        "| file | asset | source | description |",
        "|---|---|---|---|",
    ]
    for name, asset, note in TEXTURES:
        credits.append(f"| `{name}.png` | {asset} | {SOURCE.format(asset=asset)} | {note} |")
    (OUT / "CREDITS.md").write_text("\n".join(credits) + "\n")
    print(f"wrote {OUT / 'CREDITS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
