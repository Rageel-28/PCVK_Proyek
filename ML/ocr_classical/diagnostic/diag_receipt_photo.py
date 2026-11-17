"""Diagnosa: kenapa OCR gagal pada foto struk (bukan scan datar)?

Menjalankan pipeline klasik bertahap pada beberapa citra dan mencetak
kondisi tiap tahap, sehingga penyebab kegagalan terlihat dari angka, bukan
dugaan.

Pemakaian:
    python diag_receipt_photo.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import deskew as DK  # noqa: E402
import paper as PPR  # noqa: E402
import preprocess as PP  # noqa: E402
import segment as SG  # noqa: E402
from ocr_pipeline import ClassicalOCR  # noqa: E402

SRC = Path(os.path.expanduser("~")) / "Downloads"
OUT = HERE / "data" / "tmp_diag"

CASES = [
    ("WhatsApp Image 2026-11-17 at 21.50.11.jpeg", "r1_scan_datar.jpg"),
    ("WhatsApp Image 2026-11-17 at 21.49.43 (1).jpeg", "r2_foto_tangan.jpg"),
    ("WhatsApp Image 2026-11-17 at 21.49.49.jpeg", "r4_foto_latar.jpg"),
]


def prepare() -> list[tuple[str, Path]]:
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for src_name, dst_name in CASES:
        p = SRC / src_name
        if not p.exists():
            continue
        dst = OUT / dst_name
        shutil.copy(p, dst)
        out.append((dst_name, dst))
    return out


def stage_report(name: str, path: Path, ocr: ClassicalOCR) -> dict:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    # 1. Deteksi kertas
    box = PPR.find_paper_region(img)
    cropped, _ = PPR.crop_paper(img, enabled=True)

    # 2. Binarisasi pada citra (setelah crop kertas bila berhasil)
    ink_sauvola = PP.prepare(cropped, method="sauvola")
    ink_otsu = PP.prepare(cropped, method="otsu")
    d_sau = float((ink_sauvola > 0).mean())
    d_otsu = float((ink_otsu > 0).mean())

    # 3. Skala adaptif yang dipakai pipeline
    factor = ocr._adaptive_scale(ink_sauvola)

    # 4. Segmentasi baris pada kedua metode
    lines_sau = SG.segment_lines(ink_sauvola)
    lines_otsu = SG.segment_lines(ink_otsu)

    # 5. Glyph pada beberapa baris pertama
    def glyph_count(ink, lines):
        total = 0
        for lb in lines[:6]:
            total += len(SG.segment_characters(ink, lb))
        return total

    return {
        "file": name,
        "size": (w, h),
        "paper_box": box,
        "ink_ratio_sauvola": round(d_sau, 4),
        "ink_ratio_otsu": round(d_otsu, 4),
        "upscale": factor,
        "lines_sauvola": len(lines_sau),
        "lines_otsu": len(lines_otsu),
        "glyph_sauvola_6lines": glyph_count(ink_sauvola, lines_sau),
        "glyph_otsu_6lines": glyph_count(ink_otsu, lines_otsu),
    }


def main() -> None:
    cases = prepare()
    if not cases:
        raise SystemExit("citra tidak ditemukan di Downloads")

    ocr = ClassicalOCR(method="sauvola", use_deskew=True, use_lexicon=False)

    print("=" * 78)
    for name, path in cases:
        r = stage_report(name, path, ocr)
        print(f"\n[{r['file']}]  {r['size'][0]}x{r['size'][1]}")
        print(f"  deteksi kertas : {r['paper_box']}")
        print(f"  rasio tinta    : sauvola={r['ink_ratio_sauvola']:.4f}  "
              f"otsu={r['ink_ratio_otsu']:.4f}")
        print(f"  upscale        : {r['upscale']}x")
        print(f"  baris          : sauvola={r['lines_sauvola']}  otsu={r['lines_otsu']}")
        print(f"  glyph (6 baris): sauvola={r['glyph_sauvola_6lines']}  "
              f"otsu={r['glyph_otsu_6lines']}")

        # Hasil teks nyata
        for method in ("sauvola", "otsu"):
            ocr.method = method
            res = ocr.run_file(path)
            preview = " | ".join(l.text for l in res.lines[:3])
            print(f"  -> {method:7s}: {len(res.lines):3d} baris, "
                  f"conf={res.mean_confidence * 100:5.1f}%  \"{preview[:56]}\"")
    print("=" * 78)


if __name__ == "__main__":
    main()
