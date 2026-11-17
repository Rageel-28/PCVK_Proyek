"""Cari font Windows yang paling menyerupai font thermal struk.

Membandingkan glyph thermal nyata dengan glyph dari font-font sistem, memakai
korelasi piksel pada kanvas 32x32. Hasilnya dipakai untuk memutuskan font apa
yang harus ditambahkan ke data latih.

Pemakaian:
    python find_thermal_font.py
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
import preprocess as PP  # noqa: E402
import segment as SG  # noqa: E402
from render_synthetic import load_font, normalize_char  # noqa: E402

DIAG = HERE / "data" / "tmp_diag"

# Font kandidat: monospace + dot-matrix + gaya struk
CANDIDATES = [
    "consola.ttf", "consolab.ttf", "cour.ttf", "courbd.ttf",
    "lucon.ttf", "OCRAEXT.TTF", "CascadiaMono.ttf",
    "arial.ttf", "arialbd.ttf",
    "tahoma.ttf", "tahomabd.ttf",
    "verdana.ttf", "verdanab.ttf",
    "micross.ttf",   # Microsoft Sans Serif (dipakai UI lama)
    "msgothic.ttc",  # MS Gothic - bergaya dot matrix
    "simsun.ttc",
    "mingliu.ttc",
    "msmincho.ttc",
    "simhei.ttf",
    "batang.ttc",
    "gulim.ttc",
    "NSimSun.ttf",
    "FSEX302.ttf",   # Fixedsys Excelsior (dot matrix)
    "Volter.ttf",    # pixel font
]

FONT_DIR = C.FONT_DIR


def render(ch: str, fname: str, size: int) -> np.ndarray | None:
    try:
        f = load_font(fname, size)
    except Exception:  # noqa: BLE001
        return None
    probe = Image.new("L", (10, 10), 255)
    w = ImageDraw.Draw(probe).textlength(ch, font=f)
    if w <= 0:
        return None
    ascent, descent = f.getmetrics()
    img = Image.new("L", (int(w) + 20, ascent + descent + 10), 255)
    ImageDraw.Draw(img).text((10, 5), ch, font=f, fill=0)
    a = np.asarray(img)
    ys, xs = np.nonzero(a < 128)
    if len(xs) == 0:
        return None
    return normalize_char(a, (int(xs.min()), int(ys.min()),
                              int(xs.max()), int(ys.max())))


def ink_of(norm: np.ndarray) -> np.ndarray:
    """Vektor 32x32 kepadatan tinta, ternormalisasi 0..1."""
    v = (255.0 - norm.astype(np.float32)) / 255.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main() -> None:
    gray = cv2.imread(str(DIAG / "r1_scan_datar.jpg"), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ink = PP.prepare(gray, "sauvola")

    # Baris "JAVANA TEH MLATI 350" mulai y=622
    boxes = SG.segment_characters(ink, (622, 668))
    truth = "JAVANATEHMLATI350"

    # Ambil glyph yang bisa dipetakan ke huruf acuan (16 glyph pertama)
    pairs = list(zip(truth, boxes[:len(truth)]))

    print("=" * 80)
    print("MEMBANDINGKAN GLYPH THERMAL DENGAN FONT SISTEM")
    print("(korelasi kosinus pada kanvas 32x32; 1.0 = identik)")
    print("=" * 80)

    available = [f for f in CANDIDATES if (FONT_DIR / f).exists()]
    print(f"font tersedia: {len(available)} dari {len(CANDIDATES)}")
    missing = [f for f in CANDIDATES if not (FONT_DIR / f).exists()]
    if missing:
        print(f"tidak ada   : {', '.join(missing[:8])}")

    scores: dict[str, list[float]] = {}
    for fname in available:
        vals = []
        for ch, bb in pairs:
            if ch == " ":
                continue
            t_norm = ink_of(normalize_char(gray, bb))
            s_norm = render(ch, fname, 30)
            if s_norm is None:
                continue
            vals.append(float(np.dot(t_norm.ravel(), ink_of(s_norm).ravel())))
        if vals:
            scores[fname] = vals

    print("\n%-18s %10s %10s" % ("font", "korelasi", "n"))
    print("-" * 42)
    for fname, vals in sorted(scores.items(), key=lambda kv: -np.mean(kv[1])):
        print("%-18s %9.3f %10d" % (fname, float(np.mean(vals)), len(vals)))

    # Baseline: font yang dipakai latih
    train_fonts = [f for f in C.FONTS if (FONT_DIR / f).exists()]
    print(f"\nfont latih sekarang: {', '.join(train_fonts)}")
    if train_fonts:
        best_train = max(
            ((f, float(np.mean(scores[f]))) for f in train_fonts if f in scores),
            default=(None, 0.0),
            key=lambda kv: kv[1],
        )
        print(f"korelasi terbaik di font latih: {best_train[0]} "
              f"{best_train[1]:.3f}")
        best_any = max(scores.items(), key=lambda kv: np.mean(kv[1]))
        print(f"korelasi terbaik keseluruhan  : {best_any[0]} "
              f"{np.mean(best_any[1]):.3f}")
        if best_any[0] not in train_fonts:
            print(f"\n>>> {best_any[0]} lebih mirip glyph thermal tetapi "
                  f"TIDAK dipakai saat latih.")
    print("=" * 80)


if __name__ == "__main__":
    main()
