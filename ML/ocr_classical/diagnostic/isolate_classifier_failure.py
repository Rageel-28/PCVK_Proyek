"""Uji isolasi: apakah model gagal karena DOMAIN (thermal vs font sistem)?

Mengambil glyph asli dari struk (segmentasi sudah terbukti benar), lalu:
  1. Tampilkan glyph apa adanya
  2. Bandingkan dengan glyph sintetis dari font latih untuk huruf yang sama
  3. Uji model pada 3 varian glyph yang sama:
       (a) asli
       (b) asli tapi di-trim & di-resize sesuai normalisasi pipeline
       (c) asli yang "dipadatkan" horizontal (meniru font thermal)

Pemakaian:
    python isolate_classifier_failure.py
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
from render_synthetic import load_font, normalize_char  # noqa: E402

DIAG = HERE / "data" / "tmp_diag"


def ascii_art(img: np.ndarray, width: int = 20) -> str:
    h, w = img.shape
    if w > width:
        img = cv2.resize(img, (width, max(1, int(h * width / w))),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape
    chars = " .:-=+*#%@"
    return "\n".join(
        "".join(chars[min(9, int((255 - img[y, x]) / 255 * 10))] for x in range(w))
        for y in range(h)
    )


def render_glyph(ch: str, font_name: str, size: int) -> np.ndarray | None:
    font = load_font(font_name, size)
    probe = Image.new("L", (10, 10), 255)
    d = ImageDraw.Draw(probe)
    w = d.textlength(ch, font=font)
    ascent, descent = font.getmetrics()
    img = Image.new("L", (int(w) + 20, ascent + descent + 10), 255)
    ImageDraw.Draw(img).text((10, 5), ch, font=font, fill=0)
    a = np.asarray(img)
    ys, xs = np.nonzero(a < 128)
    if len(xs) == 0:
        return None
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def main() -> None:
    clf = CharClassifier()
    p = DIAG / "r1_scan_datar.jpg"
    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ink = PP.prepare(gray, "sauvola")

    print("=" * 78)
    print("KONDISI A: glyph asli thermal, dinormalisasi pipeline (32x32)")
    print("pipeline: crop -> normalize_char (skala 'contain')")
    print("=" * 78)

    # Ambil baris "JAVANA TEH MLATI 350" (baris ke-7 pada gambar)
    lines = SG.segment_lines(ink)
    target = None
    for lb in lines:
        boxes = SG.segment_characters(ink, lb)
        if 20 <= len(boxes) <= 40:
            target = (lb, boxes)
            break
    if target is None:
        raise SystemExit("baris target tidak ditemukan")

    lb, boxes = target
    print(f"\nbaris y={lb[0]}-{lb[1]}, {len(boxes)} glyph\n")

    ratios = []
    for (l, t, r, b) in boxes:
        w, h = r - l + 1, b - t + 1
        ratios.append(w / h)
    print(f"rasio aspek glyph asli: median={np.median(ratios):.2f} "
          f"min={min(ratios):.2f} max={max(ratios):.2f}")

    # Proyeksi "huruf" yang dibentuk model untuk tiap glyph
    preds = []
    for (l, t, r, b) in boxes:
        norm = normalize_char(gray, (l, t, r, b))
        ch, cf, _ = clf.predict(F.combined(norm))
        preds.append((ch, cf))
    print("prediksi model: " + "".join(p[0] for p in preds))

    # Tampilkan 3 glyph pertama vs versi sintetisnya
    print("\n--- perbandingan visual (3 glyph pertama) ---")
    for i, (l, t, r, b) in enumerate(boxes[:3]):
        patch = gray[t:b + 1, l:r + 1]
        ch, cf = preds[i]
        print(f"\nglyph #{i}: {r - l + 1}x{b - t + 1}px  -> '{ch}' (conf {cf * 100:.0f}%)")
        print(ascii_art(patch, width=18))

    print("\n" + "=" * 78)
    print("KONDISI B: glyph SINTETIS (font latih) untuk huruf yang sama")
    print("=" * 78)
    for ch in "JAVAN":
        for fname in ("consola.ttf", "cour.ttf"):
            g = render_glyph(ch, fname, 30)
            if g is None:
                continue
            h, w = g.shape
            norm = np.array(Image.fromarray(g).resize(
                (C.CHAR_SIZE - 4, C.CHAR_SIZE - 4), Image.BILINEAR))
            canvas = np.full((C.CHAR_SIZE, C.CHAR_SIZE), 255, np.uint8)
            oy, ox = (C.CHAR_SIZE - norm.shape[0]) // 2, (C.CHAR_SIZE - norm.shape[1]) // 2
            canvas[oy:oy + norm.shape[0], ox:ox + norm.shape[1]] = norm
            pred, cf, _ = clf.predict(F.combined(canvas))
            mark = "OK" if pred == ch else "SALAH"
            print(f"  '{ch}' {fname:14s} {w}x{h}px -> '{pred}' "
                  f"(conf {cf * 100:3.0f}%) [{mark}]")

    print("\n" + "=" * 78)
    print("KONDISI C: glyph thermal DIPADATKAN horizontal 0.6x")
    print("(meniru kepadatan font dot-matrix)")
    print("=" * 78)
    correct_before = correct_after = 0
    for (l, t, r, b) in boxes:
        norm = normalize_char(gray, (l, t, r, b))
        ch0, _, _ = clf.predict(F.combined(norm))

        # Padatkan 0.6x lalu tempel ulang ke kanvas
        narrow = cv2.resize(norm, (max(1, int(norm.shape[1] * 0.6)), norm.shape[0]),
                            interpolation=cv2.INTER_AREA)
        canvas = np.full_like(norm, 255)
        ox = (C.CHAR_SIZE - narrow.shape[1]) // 2
        canvas[:, ox:ox + narrow.shape[1]] = narrow
        ch1, _, _ = clf.predict(F.combined(canvas))
        correct_before += int(ch0 == ch1)
    print(f"  prediksi 0.6x berbeda dari asli pada "
          f"{len(boxes) - correct_before}/{len(boxes)} glyph")
    print("  (angka tinggi = model sangat sensitif terhadap lebar glyph)")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
