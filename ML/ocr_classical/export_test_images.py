"""Ekspor gambar baris uji untuk verifikasi end-to-end pipeline Dart.

Gambar dipilih dengan syarat: **versi Python berhasil mengenalinya dengan
benar**. Tujuannya bukan menguji ketahanan terhadap noise (itu cakupan
evaluate.py), melainkan memastikan port Dart menghasilkan keluaran yang
sama dengan Python pada masukan yang sama.

Jalankan:
    python export_test_images.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config as C
from ocr_pipeline import ClassicalOCR
from render_synthetic import degrade, load_font, render_line

OUT_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "smart_receipt"
    / "test"
    / "fixtures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXTS = [
    "INDOMIE GORENG",
    "AQUA 600ML",
    "PARACETAMOL",
    "PERTAMAX",
    "TKT TRANSJAKARTA",
    "BSKT COKLAT",
    "KARTU SIMPATI",
    "MINUMAN ISOTONIK",
]


def main() -> None:
    ocr = ClassicalOCR(method="sauvola", use_deskew=True, use_lexicon=True)

    fonts = [
        (n, s, load_font(n, s))
        for n in C.FONTS
        if (C.FONT_DIR / n).exists()
        for s in C.RENDER_SIZES
    ]

    entries = []
    attempts = 0
    for text in TEXTS:
        found = False
        for _ in range(40):
            attempts += 1
            rng = random.Random(attempts * 977 + len(text))
            _, _, font = rng.choice(fonts)
            image, _boxes = render_line(text, font)
            # Degradasi tanpa rotasi agar teks tetap sebaris.
            degraded = degrade(image, rng, allow_rotate=False)
            arr = np.asarray(degraded)

            result = ocr.run(arr)
            if result.text.strip() != text:
                continue

            name = f"line_{len(entries)}.png"
            Image.fromarray(arr).save(OUT_DIR / name)
            entries.append(
                {
                    "file": name,
                    "text": text,
                    "python_text": result.text.strip(),
                    "confidence": round(result.mean_confidence, 4),
                }
            )
            found = True
            break

        if not found:
            print(f"  ! tidak ada varian yang berhasil untuk: {text}")

    payload = {"images": entries}
    (OUT_DIR / "line_images.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"tersimpan {len(entries)} gambar di {OUT_DIR} (dari {attempts} percobaan)")
    for e in entries:
        print(f"   {e['file']} -> {e['text']!r} (conf {e['confidence']})")


if __name__ == "__main__":
    main()
