"""Ekspor golden vectors untuk menguji paritas HOG + SVM di Dart.

Menghasilkan JSON berisi beberapa patch karakter 32x32 beserta vektor HOG
dan hasil prediksi Python. Unit test Dart membandingkan hasilnya; bila
berbeda, port ke Dart salah.

Jalankan:
    python export_golden_vectors.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import config as C
import features as F
from ocr_pipeline import CharClassifier
from render_synthetic import load_font, normalize_char

OUT = C.DATA_DIR / "golden_vectors.json"


def main() -> None:
    rng = random.Random(7)

    df = pd.read_csv(C.DATASET_CSV)
    texts = [str(t).upper() for t in df["receipt_text"].dropna().unique()]
    rng.shuffle(texts)

    fonts = [
        (n, s, load_font(n, s))
        for n in C.FONTS
        if (C.FONT_DIR / n).exists()
        for s in (26, 30)
    ]

    # Ambil patch karakter dari beberapa baris.
    patches: list[np.ndarray] = []
    labels: list[str] = []
    for text in texts:
        if len(patches) >= 16:
            break
        _, _, font = rng.choice(fonts)
        probe = Image.new("L", (8, 8), 255)
        draw = ImageDraw.Draw(probe)
        ascent, descent = font.getmetrics()
        widths = [draw.textlength(ch, font=font) for ch in text]
        total_w = int(sum(widths)) + 2 * C.RENDER_MARGIN
        total_h = ascent + descent + 2 * C.RENDER_MARGIN
        if total_w <= 2 * C.RENDER_MARGIN or total_h <= 0:
            continue
        image = Image.new("L", (total_w, total_h), 255)
        draw = ImageDraw.Draw(image)
        x = float(C.RENDER_MARGIN)
        boxes = []
        for ch, w in zip(text, widths):
            if ch != " ":
                draw.text((x, C.RENDER_MARGIN), ch, font=font, fill=0)
                cell_l = max(0, int(np.floor(x)))
                cell_r = min(total_w, int(np.ceil(x + w)) + 1)
                if cell_r > cell_l:
                    cell = np.asarray(image)[:, cell_l:cell_r]
                    ys, xs = np.nonzero(cell < 128)
                    if len(xs) > 0:
                        boxes.append(
                            (
                                ch,
                                (
                                    cell_l + int(xs.min()),
                                    int(ys.min()),
                                    cell_l + int(xs.max()),
                                    int(ys.max()),
                                ),
                            )
                        )
            x += w

        gray = np.asarray(image)
        for ch, bb in boxes:
            if len(patches) >= 16:
                break
            if ch not in C.CHAR_TO_IDX:
                continue
            patch = normalize_char(gray, bb)
            patches.append(patch)
            labels.append(ch)

    classifier = CharClassifier()
    samples = []
    for patch, label in zip(patches, labels):
        vec = F.hog(patch)
        char, conf, margin = classifier.predict(vec)
        samples.append(
            {
                "expected_char": char,
                "truth_char": label,
                "pixels": [int(v) for v in patch.ravel()],
                "hog": [round(float(v), 7) for v in vec],
                "confidence": round(float(conf), 7),
                "margin": round(float(margin), 5),
            }
        )

    payload = {
        "char_size": C.CHAR_SIZE,
        "hog_dim": F.hog_dims(),
        "samples": samples,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"tersimpan: {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")
    print(f"sampel: {len(samples)}  cocok: {sum(s['expected_char'] == s['truth_char'] for s in samples)}/{len(samples)}")


if __name__ == "__main__":
    main()
