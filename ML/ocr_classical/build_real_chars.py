"""Bangun data latih karakter dari struk **asli** yang sudah ditranskripsi.

Berbeda dengan `harvest_real_chars.py` yang lama, modul ini tidak memakai
self-training. Label diambil dari transkripsi manual
(`transcripts_real.json`), sehingga model tidak mungkin mengunci
kesalahannya sendiri.

Sumber lokalisasi baris adalah anotasi COCO (`product_item`), bukan
segmentasi otomatis. Alasannya: pada foto struk kusut, segmentasi baris
otomatis masih menghasilkan blok yang menyatu (logo, QR, lipatan), jadi
tidak bisa dipakai untuk melabeli.

Aturan penerimaan sampel sengaja ketat:

1. Hanya baris yang punya bbox COCO **dan** transkripsi yang dipakai.
2. Segmentasi karakter pada crop baris harus menghasilkan jumlah
   karakter yang sama dengan jumlah karakter transkripsi (spasi
   diabaikan). Kalau tidak, baris itu dilewati.
3. Hanya karakter di `CHARSET` yang disimpan.

Konsekuensinya data yang dihasilkan sedikit, tetapi setiap label benar.
Data sintetis tetap dipakai sebagai pelengkap untuk kelas yang tidak
terwakili (lihat `train_char.py`).

Jalankan:
    python build_real_chars.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import cv2
import numpy as np

import config as C
import real_data as RD
import segment as SG
from ocr_pipeline import ClassicalOCR
from render_synthetic import normalize_char

TRANSCRIPTS = C.ROOT / "transcripts_real.json"


def _truth_chars(line: str) -> str:
    """Buang spasi dan karakter di luar charset."""
    return "".join(ch for ch in line.upper() if ch != " " and ch in C.CHAR_TO_IDX)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="chars_real_gold")
    args = ap.parse_args()

    data = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    images = {im.stem: im for im in RD.load_real_images()}

    ocr = ClassicalOCR(method="sauvola", use_deskew=False, use_lexicon=False)

    X: list[np.ndarray] = []
    y: list[int] = []
    used_lines = 0
    skipped = 0

    for stem, entry in data.items():
        im = images.get(stem)
        if im is None:
            print(f"  ! tidak ada di indeks: {stem}")
            continue
        truths = entry.get("items", [])
        if len(truths) != len(im.lines):
            print(
                f"  ! {stem}: {len(im.lines)} bbox vs {len(truths)} transkripsi, dilewati"
            )
            continue

        gray = cv2.imread(str(im.path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        for box, truth in zip(im.lines, truths):
            target = _truth_chars(truth)
            if not target:
                continue

            crop = RD.crop_line(gray, box, pad=2)
            if crop.size == 0:
                continue

            g, ink, _angle = ocr.binarize(crop)
            line_boxes = SG.segment_lines(ink)
            if not line_boxes:
                skipped += 1
                continue

            # Gabungkan seluruh baris di crop: crop idealnya hanya satu baris.
            char_boxes = SG.segment_characters(ink, line_boxes[0])
            if len(char_boxes) != len(target):
                skipped += 1
                continue

            for cb, ch in zip(char_boxes, target):
                patch = normalize_char(g, cb)
                if patch.shape != (C.CHAR_SIZE, C.CHAR_SIZE):
                    continue
                X.append(patch)
                y.append(C.CHAR_TO_IDX[ch])
            used_lines += 1

    Xa = (
        np.stack(X).astype(np.uint8)
        if X
        else np.zeros((0, C.CHAR_SIZE, C.CHAR_SIZE), np.uint8)
    )
    ya = np.array(y, dtype=np.int32)
    out_path = C.DATA_DIR / f"{args.out}.npz"
    np.savez_compressed(out_path, X=Xa, y=ya)

    counts = Counter(C.IDX_TO_CHAR[int(v)] for v in ya)
    print(f"\nTersimpan: {out_path.name}  X={Xa.shape}  kelas={len(counts)}")
    print(f"  baris dipakai: {used_lines}   baris dilewati: {skipped}")
    print("  distribusi:", " ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    digits = sum(v for k, v in counts.items() if k.isdigit())
    letters = sum(v for k, v in counts.items() if k.isalpha())
    print(f"  angka={digits}  huruf={letters}")


if __name__ == "__main__":
    main()
