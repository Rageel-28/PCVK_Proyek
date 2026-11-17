"""Uji diagnostik terfokus: satu baris teks asli vs teks yang sama (sintetis).

Mencari baris yang mengandung teks struk nyata (bukan logo), lalu
membandingkan hasil prediksi model pada:
  A. glyph dari citra struk asli
  B. teks yang sama dirender dengan font latih

Jika B benar dan A salah, penyebabnya DOMAIN GAP (font thermal vs font
sistem), bukan preprocessing maupun segmentasi.

Pemakaian:
    python diag_domain_gap.py
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
TRUTH = "JAVANA TEH MLATI 350"


def render_line(text: str, font_name: str, size: int) -> tuple[np.ndarray, list]:
    font = load_font(font_name, size)
    probe = Image.new("L", (10, 10), 255)
    d = ImageDraw.Draw(probe)
    widths = [d.textlength(ch, font=font) for ch in text]
    ascent, descent = font.getmetrics()
    W = int(sum(widths)) + 20
    H = ascent + descent + 10
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)
    x = 10.0
    boxes = []
    for ch, w in zip(text, widths):
        if ch != " ":
            draw.text((x, 5), ch, font=font, fill=0)
            cl, cr = max(0, int(x)), min(W, int(np.ceil(x + w)) + 1)
            if cr > cl:
                cell = np.asarray(img)[:, cl:cr]
                ys, xs = np.nonzero(cell < 128)
                if len(xs):
                    boxes.append((ch, (cl + int(xs.min()), int(ys.min()),
                                       cl + int(xs.max()), int(ys.max()))))
        x += w
    return np.asarray(img), boxes


def find_text_line(ink: np.ndarray, gray: np.ndarray):
    """Cari baris yang punya banyak glyph dengan tinggi seragam (baris teks)."""
    best = None
    for lb in SG.segment_lines(ink):
        h = lb[1] - lb[0] + 1
        if h < 20 or h > 80:          # baris teks struk: 20-80 px
            continue
        boxes = SG.segment_characters(ink, lb)
        if len(boxes) < 15:
            continue
        heights = [b[3] - b[2] + 1 for b in boxes]
        spread = float(np.std(heights) / max(np.mean(heights), 1))
        if spread > 0.35:             # tinggi tidak seragam -> bukan satu baris
            continue
        score = len(boxes)
        if best is None or score > best[0]:
            best = (score, lb, boxes)
    return best


def main() -> None:
    clf = CharClassifier()
    gray = cv2.imread(str(DIAG / "r1_scan_datar.jpg"), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ink = PP.prepare(gray, "sauvola")

    found = find_text_line(ink, gray)
    if not found:
        raise SystemExit("tidak menemukan baris teks")

    n, lb, boxes = found
    print("=" * 78)
    print(f"baris teks terpilih: y={lb[0]}-{lb[1]} (h={lb[1] - lb[0] + 1}), "
          f"{n} glyph")
    print("=" * 78)

    ratios = [(b[2] - b[0] + 1) / (b[3] - b[1] + 1) for b in boxes]
    print(f"rasio aspek glyph: median={np.median(ratios):.2f} "
          f"min={min(ratios):.2f} max={max(ratios):.2f}")

    # ---- A: glyph asli ----
    print("\n[A] GLYPH ASLI dari struk thermal")
    print("-" * 78)
    preds_a, confs_a = [], []
    for (l, t, r, b) in boxes:
        norm = normalize_char(gray, (l, t, r, b))
        ch, cf, _ = clf.predict(F.combined(norm))
        preds_a.append(ch)
        confs_a.append(cf)
    text_a = "".join(preds_a)
    print(f"  prediksi : {text_a}")
    print(f"  rata2 conf: {np.mean(confs_a) * 100:.1f}%  "
          f"(glyph conf>55%: {sum(1 for c in confs_a if c > 0.55)}/{len(confs_a)})")

    # ---- B: teks sama, font latih ----
    print("\n[B] TEKS SAMA dirender dengan font latih")
    print("-" * 78)
    for fname in ("consola.ttf", "cour.ttf", "OCRAEXT.TTF"):
        try:
            img, bxs = render_line(TRUTH, fname, 30)
        except Exception as e:  # noqa: BLE001
            print(f"  {fname}: gagal ({e})")
            continue
        preds_b, confs_b = [], []
        for ch_t, bb in bxs:
            norm = normalize_char(img, bb)
            ch, cf, _ = clf.predict(F.combined(norm))
            preds_b.append(ch)
            confs_b.append(cf)
        acc = sum(1 for (t_cl, _), p in zip(bxs, preds_b)
                  if t_cl.upper() == p) / max(len(bxs), 1)
        print(f"  {fname:14s} akurasi={acc * 100:5.1f}%  "
              f"conf={np.mean(confs_b) * 100:5.1f}%")
        print(f"                  {''.join(preds_b)}")

    # ---- C: glyph asli dipertebal / dipertegas ----
    print("\n[C] GLYPH ASLI dengan stroke dipertebal (dilasi 1px)")
    print("-" * 78)
    preds_c, confs_c = [], []
    for (l, t, r, b) in boxes:
        norm = normalize_char(gray, (l, t, r, b))
        binm = (norm < 128).astype(np.uint8) * 255
        binm = cv2.dilate(binm, np.ones((2, 2), np.uint8), iterations=1)
        bold = 255 - binm
        ch, cf, _ = clf.predict(F.combined(bold))
        preds_c.append(ch)
        confs_c.append(cf)
    print(f"  prediksi : {''.join(preds_c)}")
    print(f"  rata2 conf: {np.mean(confs_c) * 100:.1f}%")

    print("\n" + "=" * 78)
    print("KESIMPULAN:")
    print(f"  glyph thermal : conf {np.mean(confs_a) * 100:.1f}%")
    print(f"  font sistem   : conf lihat tabel B di atas")
    print("  Bila font sistem ~99% dan thermal ~20%, penyebabnya DOMAIN GAP.")
    print("=" * 78)


if __name__ == "__main__":
    main()
