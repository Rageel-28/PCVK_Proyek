"""Nilai kelayakan tiap citra dataset untuk pelabelan glyph.

Untuk setiap citra dihitung:
  - tinggi glyph median (semakin besar semakin baik; < 12 px sulit dilabeli)
  - rasio aspek glyph median (font thermal ~0.30-0.40, laser ~0.50-0.65)
  - jumlah baris & glyph yang bisa diekstrak
  - skor kemiripan dengan target (struk Indomaret: glyph sempit & banyak)

Hasil: CSV terurut, sehingga bisa memilih citra MANA yang layak dilabeli
lebih dulu, bukan melabeli 771 citra secara membabi buta.

Pemakaian:
    python rank_labeling_candidates.py [--split train] [--limit 0]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OCR = ROOT / "ML" / "ocr_classical"
OUT = HERE / "labeling_candidates.csv"

sys.path.insert(0, str(OCR))
sys.path.insert(0, str(HERE))
import dataset_source as DS  # noqa: E402
import paper as PPR  # noqa: E402
import preprocess as PP  # noqa: E402
import segment as SG  # noqa: E402

# Target: glyph Indomaret hasil ukur (rasio aspek ~0.33, tinggi ~50 px
# setelah upscale 4x dari citra 310x320).
TARGET_ASPECT = 0.33
MIN_GLYPH_H = 12.0


def analyze(path: Path) -> dict | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img, _ = PPR.crop_paper(img, enabled=True)
    if img.size == 0:
        return None

    # Perbesar agar glyph setara target bila citra kecil.
    h, w = img.shape
    scale = 1.0
    if max(h, w) < 900:
        scale = 900 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    ink = PP.prepare(img, "sauvola")
    lines = SG.segment_lines(ink)

    heights, aspects = [], []
    n_glyph = 0
    for lb in lines:
        boxes = SG.segment_characters(ink, lb)
        n_glyph += len(boxes)
        for (l, t, r, b) in boxes:
            gh, gw = b - t + 1, r - l + 1
            if gh < 6:
                continue
            heights.append(gh)
            aspects.append(gw / gh)

    if not heights:
        return None

    med_h = float(np.median(heights))
    med_a = float(np.median(aspects))
    # Skor: dekat dengan target aspek, glyph cukup tinggi, banyak glyph.
    aspect_score = 1.0 - min(abs(med_a - TARGET_ASPECT) / 0.35, 1.0)
    height_score = min(med_h / MIN_GLYPH_H, 1.0)
    volume_score = min(n_glyph / 300.0, 1.0)
    score = 0.45 * aspect_score + 0.30 * height_score + 0.25 * volume_score

    quality = "A" if (med_h >= MIN_GLYPH_H and med_a < 0.45) else \
              "B" if (med_h >= MIN_GLYPH_H * 0.7) else "C"

    return {
        "file": path.name,
        "w": w, "h": h, "upscale": round(scale, 2),
        "lines": len(lines),
        "glyphs": n_glyph,
        "median_glyph_h": round(med_h, 1),
        "median_aspect": round(med_a, 3),
        "score": round(score, 3),
        "quality": quality,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Nilai kelayakan citra struk Indonesia untuk pelabelan glyph"
    )
    ap.add_argument("--split", default="all",
                    choices=["all", "train", "valid", "test"],
                    help="split yang dinilai (bawaan: all = ketiganya)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.split == "all":
        files = DS.all_indonesian_files()
    else:
        files = DS.indonesian_files(args.split)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(
            f"tidak ada citra struk Indonesia di {DS.DATASET.name} "
            f"(split={args.split})"
        )

    print(f"dataset: {DS.DATASET.name}")
    print(f"menganalisis {len(files)} citra struk Indonesia...")
    rows = []
    for i, p in enumerate(files, 1):
        r = analyze(p)
        if r:
            rows.append(r)
        if i % 50 == 0:
            print(f"  {i}/{len(files)}")

    rows.sort(key=lambda r: -r["score"])
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    a = [r for r in rows if r["quality"] == "A"]
    b = [r for r in rows if r["quality"] == "B"]
    c = [r for r in rows if r["quality"] == "C"]
    print(f"\nhasil: {len(rows)} citra bisa dianalisis")
    print(f"  kualitas A (layak dilabeli)  : {len(a)}")
    print(f"  kualitas B (mungkin, glyph kecil): {len(b)}")
    print(f"  kualitas C (tidak layak)     : {len(c)}")
    print(f"\nCSV: {OUT}")
    print("\n10 citra teratas:")
    print("%-46s %6s %6s %7s %7s %5s" % ("file", "glyph", "tinggi", "aspek", "skor", "mutu"))
    for r in rows[:10]:
        print("%-46s %6d %6.1f %7.2f %7.3f %5s"
              % (r["file"][:46], r["glyphs"], r["median_glyph_h"],
                 r["median_aspect"], r["score"], r["quality"]))


if __name__ == "__main__":
    main()
