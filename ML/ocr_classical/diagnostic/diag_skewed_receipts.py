"""Uji apakah kemiringan & distorsi bisa diperbaiki SEBELUM pelabelan.

Dua kritik yang dijawab di sini:
  1. "Struk miring / tertekuk, bagaimana melabelinya?"
  2. "Bukankah COCO sudah ada labelnya?"

Untuk (1): diukur apakah tahap pra-proses (deskew + koreksi perspektif)
mampu memulihkan citra sulit sampai layak dilabeli. Bila gagal, citra itu
harus DITOLAK, bukan dipaksakan.

Untuk (2): anotasi COCO hanya berisi bbox per wilayah (line_item, nama_toko,
tanggal_waktu, total_belanja) TANPA teks. Model OCR memerlukan label PER
KARAKTER, jadi anotasi COCO tidak bisa dipakai langsung.

Pemakaian:
    python diag_skewed_receipts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OCR = ROOT / "ML" / "ocr_classical"
sys.path.insert(0, str(OCR))
sys.path.insert(0, str(HERE))

import dataset_source as DS  # noqa: E402
import deskew as DK  # noqa: E402
import paper as PPR  # noqa: E402
import preprocess as PP  # noqa: E402
import segment as SG  # noqa: E402


def coco_annotation_report() -> None:
    """Tunjukkan bahwa anotasi COCO tidak memuat teks."""
    print("=" * 74)
    print("BUKTI: ANOTASI COCO TIDAK BERISI TEKS")
    print("=" * 74)
    for split in ("train", "valid", "test"):
        p = DS.DATASET / split / "_annotations.coco.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        keys = set()
        for a in d["annotations"][:50]:
            keys |= set(a.keys())
        print(f"\n{split}: {len(d['images'])} gambar, "
              f"{len(d['annotations'])} anotasi")
        print(f"  kategori : {[c['name'] for c in d['categories']]}")
        print(f"  kunci    : {sorted(keys)}")
        contoh = d["annotations"][0]
        print(f"  contoh   : bbox={contoh['bbox']} "
              f"kategori={contoh['category_id']}")
    print("\n  -> Tidak ada kunci 'text'/'transcription'/'label'.")
    print("  -> Anotasi hanya KOTAK AREA. Untuk OCR per karakter, teks")
    print("     setiap baris tetap harus diketik manusia.\n")


def measure(path: Path) -> dict:
    """Ukur kualitas citra sebelum & sesudah pra-proses."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"file": path.name, "error": "gagal baca"}

    raw_h, raw_w = img.shape
    cropped, _ = PPR.crop_paper(img, enabled=True)

    ink0 = PP.prepare(cropped, "sauvola")
    angle = DK.estimate_skew(ink0, max_angle=25.0)
    if abs(angle) >= 0.3:
        gray, ink, used = DK.auto_deskew_pair(cropped, ink0, max_angle=25.0)
    else:
        gray, ink, used = cropped, ink0, 0.0

    lines = SG.segment_lines(ink)
    glyphs = sum(len(SG.segment_characters(ink, lb)) for lb in lines)

    heights = []
    for lb in lines:
        for (l, t, r, b) in SG.segment_characters(ink, lb):
            heights.append(b - t + 1)
    med_h = float(np.median(heights)) if heights else 0.0

    # Skor kelayakan: butuh cukup baris & glyph yang tidak terlalu kecil.
    layak = len(lines) >= 6 and glyphs >= 80 and med_h >= 8

    return {
        "file": path.name,
        "size": f"{raw_w}x{raw_h}",
        "skew_before": round(angle, 1),
        "skew_used": round(used, 1),
        "lines": len(lines),
        "glyphs": glyphs,
        "median_h": round(med_h, 1),
        "layak": "YA" if layak else "TIDAK",
    }


def main() -> None:
    coco_annotation_report()

    print("=" * 74)
    print("UJI: KEMIRINGAN & KUALITAS CITRA INDONESIA")
    print("=" * 74)
    print("%-42s %6s %6s %6s %7s %6s %s"
          % ("file", "skew", "dipakai", "baris", "glyph", "h", "layak"))
    print("-" * 74)

    rows = []
    for p in DS.all_indonesian_files():
        if "struk-14" in p.name or "trashbag" in p.name or "HVS" in p.name:
            r = measure(p)
            rows.append(r)
            if "error" in r:
                continue
            print("%-42s %5.1f° %6.1f° %6d %7d %6.1f %s"
                  % (r["file"][:42], r["skew_before"], r["skew_used"],
                     r["lines"], r["glyphs"], r["median_h"], r["layak"]))

    print("\nCATATAN:")
    print("- skew > 5° biasanya berarti struk tertekuk/miring di foto.")
    print("- 'layak = TIDAK' berarti citra itu SEBAIKNYA DILEWATI, bukan")
    print("  dipaksa dilabeli: label yang salah lebih merusak daripada")
    print("  kekurangan data.")


if __name__ == "__main__":
    main()
