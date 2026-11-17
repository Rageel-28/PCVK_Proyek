"""Buat lembar kontak: ringkasan visual dataset struk thermal.

Menampilkan beberapa citra dalam satu gambar agar isi dataset bisa dinilai
sekilas: apakah strukturnya seragam, bahasanya apa, kualitas cetaknya
bagaimana. Ini penting sebelum memutuskan mana yang dilabelkan.

Pemakaian:
    python make_contact_sheet.py [jumlah] [split]
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]          # diagnostic -> ocr_classical -> ML -> repo root
COCO = ROOT / "ML" / "OCR Struk Belanja.v2i.coco"
OUT = HERE / "contact_sheet.jpg"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    split = sys.argv[2] if len(sys.argv) > 2 else "train"

    files = sorted((COCO / split).glob("*.jpg"))
    if not files:
        raise SystemExit(f"tidak ada citra di {COCO / split}")

    step = max(1, len(files) // n)
    picked = files[::step][:n]

    cols, cell = 4, 320
    rows = (len(picked) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), "white")

    for i, p in enumerate(picked):
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell - 8, cell - 8))
        x = (i % cols) * cell + (cell - im.width) // 2
        y = (i // cols) * cell + (cell - im.height) // 2
        sheet.paste(im, (x, y))

    sheet.save(OUT, quality=88)
    print(f"total citra {split}: {len(files)}")
    print(f"lembar kontak: {OUT}  ({len(picked)} citra)")


if __name__ == "__main__":
    main()
