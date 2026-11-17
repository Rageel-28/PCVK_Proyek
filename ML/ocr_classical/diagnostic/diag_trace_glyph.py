"""Bandingkan bentuk glyph 32x32 hasil normalisasi: thermal vs font latih.

Tujuan: melihat APA yang sebenarnya masuk ke model. Diagnosis sebelumnya
(rasio aspek, kontras, resolusi) tidak menjelaskan kegagalan, jadi tahap
ini memeriksa bentuk akhirnya.

Pemakaian:
    python diag_trace_glyph.py
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


def art(img: np.ndarray) -> list[str]:
    """ASCII 32x32 dengan 2 karakter per piksel."""
    chars = " .:-=+*#%@"
    rows = []
    for y in range(0, img.shape[0], 1):
        line = "".join(chars[min(9, int((255 - img[y, x]) / 255 * 10))]
                       for x in range(img.shape[1]))
        rows.append(line)
    return rows


def side_by_side(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> None:
    ra, rb = art(a), art(b)
    h = max(len(ra), len(rb))
    print(f"\n{label_a:<34}{label_b}")
    print("-" * 70)
    for i in range(h):
        left = ra[i] if i < len(ra) else " " * a.shape[1]
        right = rb[i] if i < len(rb) else " " * b.shape[1]
        print(f"{left}  {right}")


def synth(ch: str, font: str = "consola.ttf", size: int = 30):
    f = load_font(font, size)
    probe = Image.new("L", (10, 10), 255)
    d = ImageDraw.Draw(probe)
    w = d.textlength(ch, font=f)
    ascent, descent = f.getmetrics()
    img = Image.new("L", (int(w) + 20, ascent + descent + 10), 255)
    ImageDraw.Draw(img).text((10, 5), ch, font=f, fill=0)
    a = np.asarray(img)
    ys, xs = np.nonzero(a < 128)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return normalize_char(a, bbox), bbox


def main() -> None:
    clf = CharClassifier()
    gray = cv2.imread(str(DIAG / "r1_scan_datar.jpg"), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ink = PP.prepare(gray, "sauvola")

    # Baris y=622-668 memuat "JAVANA TEH MLATI 350"
    box = (622, 668)
    boxes = SG.segment_characters(ink, box)
    print(f"baris y={box[0]}-{box[1]}: {len(boxes)} glyph")
    print("kotak glyph pertama:", boxes[:6])

    # Ambil 2 glyph pertama dan bandingkan dengan 'J' sintetis
    for i, bb in enumerate(boxes[:2]):
        l, t, r, b = bb
        w, h = r - l + 1, b - t + 1
        norm = normalize_char(gray, bb)
        pred, conf, _ = clf.predict(F.combined(norm))

        print("\n" + "=" * 70)
        print(f"GLYPH #{i}: bbox={bb}  {w}x{h}px  aspek={w / h:.2f}")
        print(f"model menebak: '{pred}'  conf={conf * 100:.1f}%")
        print("=" * 70)

        # Bandingkan dengan 'J' (huruf pertama teks acuan)
        s_norm, s_box = synth("J")
        sw, sh = s_box[2] - s_box[0] + 1, s_box[3] - s_box[1] + 1
        print(f"\n'J' sintetis Consolas: {sw}x{sh}px aspek={sw / sh:.2f}")
        side_by_side(norm, s_norm,
                     f"THERMAL -> '{pred}'", "'J' Consolas (model: benar?)")

    # Distribusi prediksi untuk seluruh baris
    print("\n" + "=" * 70)
    print("Semua glyph pada baris 'JAVANA TEH MLATI 350':")
    out = []
    for bb in boxes:
        norm = normalize_char(gray, bb)
        ch, cf, _ = clf.predict(F.combined(norm))
        out.append(ch)
    print("  " + "".join(out))
    print("  " + "JAVANATEHMLATI350 (acuan, tanpa spasi)")


if __name__ == "__main__":
    main()
