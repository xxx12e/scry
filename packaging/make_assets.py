"""Generate the MSIX logo assets (no binaries in git): python make_assets.py OUT_DIR"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

SIZES = {"Square44x44Logo.png": (44, 44), "Square150x150Logo.png": (150, 150),
         "Wide310x150Logo.png": (310, 150), "StoreLogo.png": (50, 50)}


def logo(w, h):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = min(w, h)
    x0, y0 = (w - s) // 2, (h - s) // 2
    pad = max(2, s // 10)
    d.ellipse((x0 + pad, y0 + pad, x0 + s - pad, y0 + s - pad), fill=(37, 99, 235, 255))
    r = s // 6
    cx, cy = w // 2, h // 2
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, 255))
    return img


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (w, h) in SIZES.items():
        logo(w, h).save(out / name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Assets")
