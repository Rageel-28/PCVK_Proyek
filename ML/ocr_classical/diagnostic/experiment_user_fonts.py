"""Uji: apakah font dot-matrix yang diunduh pengguna bisa memperbaiki OCR?

Latar: glyph struk thermal nyata tidak cocok dengan font latih lama
(korelasi <= 0.54). Pengguna mengunduh font bergaya thermal ke `ML/Font/`.
Uji ini mengukur dua hal:

  1. Seberapa mirip font baru dengan glyph thermal nyata (korelasi)
  2. Apakah model yang dilatih dengan font baru membaca struk nyata lebih baik

Yang diuji: Volter, FSEX302 (Fixedsys), dan keluarga DotMatrix.

Pemakaian:
    python experiment_user_fonts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OCR = ROOT / "ML" / "ocr_classical"
FONT_DIR = ROOT / "ML" / "Font"

sys.path.insert(0, str(OCR))
import config as C  # noqa: E402
import features as F  # noqa: E402
import preprocess as PP  # noqa: E402
import segment as SG  # noqa: E402
from render_synthetic import normalize_char  # noqa: E402

# Citra struk Indonesia milik pengguna (dipakai sebagai uji akhir).
USER_RECEIPTS = HERE / "testdata"

# Teks acuan yang dibaca manual dari struk Javana (baris demi baris).
JAVANA_TRUTH = {
    (291, 333): "CIDATARKABGARUT",
    (341, 393): "J1RayaCidatarCikajangKelCidatarKec",
    (622, 668): "JAVANATEHMLATI350",
    (915, 958): "HARGAJUAL",
    (1036, 1064): "TOTAL",
}


def collect_fonts() -> list[Path]:
    out = []
    if not FONT_DIR.exists():
        return out
    for ext in ("*.ttf", "*.otf", "*.ttc"):
        out.extend(FONT_DIR.rglob(ext))
    return sorted(set(out))


def render_char(ch: str, font_path: Path, size: int) -> np.ndarray | None:
    try:
        f = ImageFont_wrap(font_path, size)
    except Exception:  # noqa: BLE001
        return None
    try:
        probe = Image.new("L", (10, 10), 255)
        w = ImageDraw.Draw(probe).textlength(ch, font=f)
        ascent, descent = f.getmetrics()
        if w <= 0 or ascent <= 0:
            return None
        img = Image.new("L", (int(w) + 16, ascent + descent + 8), 255)
        ImageDraw.Draw(img).text((8, 4), ch, font=f, fill=0)
        a = np.asarray(img)
        ys, xs = np.nonzero(a < 128)
        if len(xs) == 0:
            return None
        return normalize_char(a, (int(xs.min()), int(ys.min()),
                                  int(xs.max()), int(ys.max())))
    except Exception:  # noqa: BLE001
        return None


def ImageFont_wrap(path: Path, size: int):
    from PIL import ImageFont
    return ImageFont.truetype(str(path), size)


def ink_vec(norm: np.ndarray) -> np.ndarray:
    v = (255.0 - norm.astype(np.float32)) / 255.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def load_thermal_glyphs():
    """Glyph thermal nyata + huruf acuannya dari struk Javana."""
    p = USER_RECEIPTS / "javana.jpg"
    if not p.exists():
        return None, None, None
    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ink = PP.prepare(gray, "sauvola")
    return gray, ink, JAVANA_TRUTH


def main() -> None:
    fonts = collect_fonts()
    print(f"font ditemukan: {len(fonts)}")
    for f in fonts[:6]:
        print("  ", f.relative_to(FONT_DIR))
    if len(fonts) > 6:
        print(f"   ... dan {len(fonts) - 6} lainnya")

    gray, ink, truth = load_thermal_glyphs()
    if gray is None:
        raise SystemExit("citra struk uji tidak ada")

    # --- 1. Korelasi font baru dengan glyph thermal nyata ---
    print("\n" + "=" * 76)
    print("1. KEMIRIPAN FONT BARU DENGAN GLYPH THERMAL NYATA")
    print("=" * 76)

    # Ambil pasangan (huruf, glyph) dari baris JAVANA TEH MLATI 350
    box = (622, 668)
    boxes = SG.segment_characters(ink, box)
    pairs = list(zip("JAVANATEHMLATI350", boxes))

    results = []
    for fp in fonts:
        vals = []
        for ch, bb in pairs:
            t = ink_vec(normalize_char(gray, bb))
            s = None
            for size in (28, 32, 36):
                g = render_char(ch, fp, size)
                if g is not None:
                    s = ink_vec(g)
                    break
            if s is None:
                continue
            vals.append(float(np.dot(t.ravel(), s.ravel())))
        if vals:
            results.append((fp.name, float(np.mean(vals)), len(vals)))

    results.sort(key=lambda r: -r[1])
    print("%-34s %10s %5s" % ("font", "korelasi", "n"))
    print("-" * 52)
    for name, corr, n in results[:12]:
        print("%-34s %10.3f %5d" % (name[:34], corr, n))
    print("-" * 52)
    print("font latih lama (Consolas) korelasi = 0.456")
    best = results[0] if results else None
    if best:
        gain = best[1] - 0.456
        print(f"terbaik: {best[0]} = {best[1]:.3f} ({gain:+.3f} vs Consolas)")

    # --- 2. Latih model dengan font baru, uji pada struk nyata ---
    print("\n" + "=" * 76)
    print("2. APAKAH MODEL BARU LEBIH BAIK PADA STRUK NYATA?")
    print("=" * 76)

    def build(font_list, per_char=40, seed=0):
        rng = np.random.default_rng(seed)
        X, y = [], []
        for ch in C.CHARSET:
            got = 0
            for _ in range(per_char * 4):
                if got >= per_char:
                    break
                fp = font_list[int(rng.integers(0, len(font_list)))]
                g = render_char(ch, fp, int(rng.choice([26, 30, 34])))
                if g is None:
                    continue
                X.append(g)
                y.append(C.CHAR_TO_IDX[ch])
                got += 1
        return np.stack(X).astype(np.uint8), np.array(y, np.int32)

    old_fonts = [C.FONT_DIR / f for f in C.FONTS if (C.FONT_DIR / f).exists()]
    # Jumlah sampel per kelas harus sama agar perbandingan adil. Model "font
    # baru" memakai semua font (variasi bentuk luas) dengan jumlah sama.
    N_PER_CHAR = 250
    X_old, y_old = build(old_fonts, per_char=N_PER_CHAR, seed=1)
    X_new, y_new = build(fonts, per_char=N_PER_CHAR, seed=1)
    print(f"  font lama : {len(y_old)} sampel ({len(old_fonts)} font)")
    print(f"  font baru : {len(y_new)} sampel ({len(fonts)} font)")

    def train(X_img, y):
        X = F.combined_batch(X_img)
        sc = StandardScaler().fit(X)
        m = LinearSVC(C=C.SVM_C, max_iter=5000, dual=True, random_state=42)
        m.fit(sc.transform(X), y)
        return m, sc

    m_old, sc_old = train(X_old, y_old)
    m_new, sc_new = train(X_new, y_new)

    def eval_line(m, sc, line_box, label):
        bxs = SG.segment_characters(ink, line_box)
        truth_txt = label.replace(" ", "")
        chars, confs = [], []
        for bb in bxs[:len(truth_txt)]:
            vec = F.combined(normalize_char(gray, bb))
            vec = (vec - sc.mean_) / sc.scale_
            sc_ = m.coef_ @ vec + m.intercept_
            best = int(np.argmax(sc_))
            chars.append(C.IDX_TO_CHAR[int(m.classes_[best])])
            z = sc_ - sc_.max()
            e = np.exp(z)
            confs.append(float(e[best] / e.sum()))
        pred = "".join(chars)
        n = min(len(pred), len(truth_txt))
        acc = sum(1 for i in range(n) if pred[i] == truth_txt[i]) / max(len(truth_txt), 1)
        return pred, float(np.mean(confs)) if confs else 0, acc

    print("\n%-22s %-13s %-26s %-7s %s"
          % ("teks acuan", "model", "prediksi", "conf", "akurasi"))
    print("-" * 92)
    agg = {"font lama": [], "font baru": []}
    for (y0, y1), label in JAVANA_TRUTH.items():
        for name, (m, sc) in (("font lama", (m_old, sc_old)),
                              ("font baru", (m_new, sc_new))):
            pred, conf, acc = eval_line(m, sc, (y0, y1), label)
            agg[name].append((conf, acc))
            print("%-22s %-13s %-26s %5.1f%% %6.1f%%"
                  % (label[:22], name, pred[:26], conf * 100, acc * 100))
        print("-" * 92)

    print("\nRATA-RATA:")
    for name, rows in agg.items():
        print("  %-12s conf=%5.1f%%  akurasi karakter=%5.1f%%"
              % (name, np.mean([r[0] for r in rows]) * 100,
                 np.mean([r[1] for r in rows]) * 100))
    print("=" * 76)


if __name__ == "__main__":
    main()
