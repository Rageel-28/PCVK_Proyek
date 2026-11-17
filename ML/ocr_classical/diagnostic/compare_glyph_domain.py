"""Bandingkan glyph NYATA (thermal) vs glyph SINTETIS untuk huruf yang sama.

Mencetak bentuk ASCII dan prediksi model untuk keduanya. Tujuannya
menjawab satu pertanyaan: apakah model gagal karena bentuk glyph thermal
memang berbeda dari font latih (domain gap), atau karena segmentasi salah.

Pemakaian:
    python compare_glyph_domain.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import features as F  # noqa: E402
import preprocess as PP  # noqa: E402
import segment as SG  # noqa: E402
from ocr_pipeline import CharClassifier  # noqa: E402
from render_synthetic import load_font  # noqa: E402

DIAG = HERE / "data" / "tmp_diag"


def ascii_art(img: np.ndarray, width: int = 46) -> str:
    h, w = img.shape
    if w > width:
        scale = width / w
        img = cv2.resize(img, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        h, w = img.shape
    chars = " .:-=+*#%@"
    out = []
    for y in range(h):
        row = "".join(chars[min(len(chars) - 1, int((255 - img[y, x]) / 255 * len(chars)))]
                      for x in range(w))
        out.append(row)
    return "\n".join(out)


def render_text(text: str, font_name: str = "consola.ttf", size: int = 28) -> np.ndarray:
    font = load_font(font_name, size)
    probe = Image.new("L", (10, 10), 255)
    draw = ImageDraw.Draw(probe)
    widths = [draw.textlength(ch, font=font) for ch in text]
    total_w = int(sum(widths)) + 16
    ascent, descent = font.getmetrics()
    total_h = ascent + descent + 16
    image = Image.new("L", (total_w, total_h), 255)
    draw = ImageDraw.Draw(image)
    x = 8.0
    for ch, w in zip(text, widths):
        draw.text((x, 8), ch, font=font, fill=0)
        x += w
    return np.asarray(image)


def predict(clf: CharClassifier, patch: np.ndarray) -> tuple[str, float]:
    ch, cf, _ = clf.predict(F.combined(patch))
    return ch, cf


def normalize_like_pipeline(gray: np.ndarray, bbox) -> np.ndarray:
    from render_synthetic import normalize_char
    return normalize_char(gray, bbox)


def main() -> None:
    clf = CharClassifier()
    p = DIAG / "r1_scan_datar.jpg"
    if not p.exists():
        raise SystemExit(f"tidak ada: {p}")

    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    # Perbesar seperti yang dilakukan pipeline pada citra kecil.
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ink = PP.prepare(gray, "sauvola")

    print("=" * 78)
    print("GLYPH NYATA dari struk thermal (r1_scan_datar, 4x)")
    print("=" * 78)
    lines = SG.segment_lines(ink)
    shown = 0
    for lb in lines:
        boxes = SG.segment_characters(ink, lb)
        if len(boxes) < 12:
            continue
        t = lb[0]
        for (l, tt, r, b) in boxes[:10]:
            patch = gray[tt:b + 1, l:r + 1]
            if patch.size == 0:
                continue
            norm = normalize_like_pipeline(gray, (l, tt, r, b))
            ch, cf = predict(clf, norm)
            print(f"\n  glyph w={r - l + 1} h={b - tt + 1} -> model: "
                  f"'{ch}' (conf {cf * 100:.0f}%)")
            print(ascii_art(patch, width=24))
            shown += 1
            if shown >= 4:
                break
        if shown >= 4:
            break

    print("\n" + "=" * 78)
    print("GLYPH SINTETIS dari font latih (Consolas 28px)")
    print("=" * 78)
    for ch in "JAVAN":
        img = render_text(ch)
        ys, xs = np.nonzero(img < 128)
        if len(xs) == 0:
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        norm = normalize_like_pipeline(img, bbox)
        pred, cf = predict(clf, norm)
        ok = "OK " if pred == ch else "SALAH"
        print(f"\n  '{ch}' sintetis -> model: '{pred}' (conf {cf * 100:.0f}%)  [{ok}]")
        print(ascii_art(img[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1], width=24))

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
