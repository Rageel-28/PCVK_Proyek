"""Simpan gambar tiap tahap agar penyebab kegagalan bisa dilihat langsung.

Keluaran di data/tmp_diag/stages/:
  01_gray.png      citra setelah upscale
  02_sauvola.png   mask tinta Sauvola
  03_otsu.png      mask tinta Otsu
  04_lines.png     kotak baris yang terdeteksi (merah)
  05_glyphs.png    kotak glyph yang terdeteksi (biru)

Pemakaian:
    python dump_stages.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paper as PPR  # noqa: E402
import preprocess as PP  # noqa: E402
import segment as SG  # noqa: E402

DIAG = HERE / "data" / "tmp_diag"
OUT = DIAG / "stages"


def dump(name: str, path: Path, upscale: int = 3) -> None:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    gray, _ = PPR.crop_paper(gray, enabled=True)
    if upscale > 1:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)

    (OUT / name).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / name / "01_gray.png"), gray)

    sauvola = PP.prepare(gray, "sauvola")
    otsu = PP.prepare(gray, "otsu")
    cv2.imwrite(str(OUT / name / "02_sauvola.png"), sauvola)
    cv2.imwrite(str(OUT / name / "03_otsu.png"), otsu)

    ink = sauvola
    lines = SG.segment_lines(ink)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for lb in lines:
        cv2.rectangle(vis, (0, lb[0]), (vis.shape[1] - 1, lb[1]), (0, 0, 255), 2)
    cv2.imwrite(str(OUT / name / "04_lines.png"), vis)

    vis2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for lb in lines:
        for (l, t, r, b) in SG.segment_characters(ink, lb):
            cv2.rectangle(vis2, (l, t), (r, b), (255, 0, 0), 1)
    cv2.imwrite(str(OUT / name / "05_glyphs.png"), vis2)

    print(f"{name}: {gray.shape[1]}x{gray.shape[0]}  baris={len(lines)}  "
          f"-> {OUT / name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, tag in (("r1_scan_datar.jpg", "r1"), ("r2_foto_tangan.jpg", "r2")):
        p = DIAG / fname
        if p.exists():
            dump(tag, p)


if __name__ == "__main__":
    main()
