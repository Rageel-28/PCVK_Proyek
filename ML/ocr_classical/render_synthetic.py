"""Render teks struk sintetis menjadi gambar + label karakter otomatis.

Ide utama: kita sudah punya 6.026 teks struk berlabel. Dengan merender teks
itu memakai font monospace dan mencatat posisi setiap huruf, kita mendapat
data latih karakter dalam jumlah besar TANPA anotasi manual.

Degradasi yang diterapkan meniru cetakan struk thermal:
  - blur halus (tinta menyebar di kertas)
  - noise titik (salt & pepper)
  - kontras rendah (kertas pudar)
  - kemiringan kecil (foto tidak tegak)
  - pencahayaan tidak rata (bayangan tangan)

Output:
  data/chars_train.npz   X (N,32,32) uint8, y (N,) int
  data/chars_val.npz
  data/lines_test.npz    gambar baris + teks asli untuk uji end-to-end
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import config as C


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = C.FONT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Font tidak ditemukan: {path}")
    return ImageFont.truetype(str(path), size)


def render_line(text: str, font: ImageFont.FreeTypeFont) -> tuple[Image.Image, list]:
    """Render satu baris dan kembalikan gambar + bbox tinta tiap karakter.

    Bbox diambil dari piksel tinta sebenarnya (tight bbox), bukan dari
    metrik font. Ini penting: saat inferensi, segmentasi menghasilkan
    tight bbox juga. Kalau pelatihan memakai bbox metrik yang lebih
    longgar, proporsi huruf saat latih dan saat uji akan berbeda dan
    akurasi jatuh (mis. "T" terbaca "Y").
    """
    probe = Image.new("L", (8, 8), 255)
    draw = ImageDraw.Draw(probe)
    ascent, descent = font.getmetrics()

    widths = [draw.textlength(ch, font=font) for ch in text]
    total_w = int(sum(widths)) + 2 * C.RENDER_MARGIN
    total_h = ascent + descent + 2 * C.RENDER_MARGIN
    if total_w <= 2 * C.RENDER_MARGIN or total_h <= 0:
        return Image.new("L", (8, 8), 255), []

    img = Image.new("L", (total_w, total_h), 255)
    draw = ImageDraw.Draw(img)

    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    x = float(C.RENDER_MARGIN)
    for ch, w in zip(text, widths):
        if ch != " ":
            draw.text((x, C.RENDER_MARGIN), ch, font=font, fill=0)
            # Tight bbox: cari piksel tinta di dalam sel karakter ini.
            cell_l = max(0, int(np.floor(x)))
            cell_r = min(total_w, int(np.ceil(x + w)) + 1)
            if cell_r > cell_l:
                cell = np.asarray(img)[:, cell_l:cell_r]
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

    return img, boxes


# ---------------------------------------------------------------------------
# Degradasi
# ---------------------------------------------------------------------------
def degrade(img: Image.Image, rng: random.Random, allow_rotate: bool = True) -> Image.Image:
    """Terapkan degradasi mirip struk thermal.

    ``allow_rotate=False`` dipakai saat bbox karakter masih akan dipakai:
    rotasi mengubah koordinat sehingga label menjadi tidak valid.
    """
    arr = np.asarray(img).astype(np.float32)

    # Pencahayaan tidak rata: gradien acak arah.
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx / max(w, 1) + np.sin(ang) * yy / max(h, 1))
    grad = (grad - grad.min()) / max(grad.max() - grad.min(), 1e-6)
    strength = rng.uniform(0.0, 0.18)
    arr = arr * (1.0 - strength * (1.0 - grad)) + 255.0 * strength * (1.0 - grad)

    # Kontras rendah (kertas pudar).
    lo = rng.uniform(0, 60)
    hi = rng.uniform(190, 255)
    arr = lo + (arr / 255.0) * (hi - lo)

    # Blur halus.
    if rng.random() < 0.7:
        radius = rng.uniform(0.4, 1.1)
        arr = np.asarray(
            Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(radius)
            )
        ).astype(np.float32)

    # Salt & pepper.
    if rng.random() < 0.6:
        p = rng.uniform(0.0005, 0.006)
        mask = np.random.random(arr.shape)
        arr[mask < p] = 0
        arr[mask > 1 - p] = 255

    out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    # Kemiringan kecil.
    if allow_rotate and rng.random() < 0.5:
        angle = rng.uniform(-3.0, 3.0)
        out = out.rotate(
            angle, resample=Image.BILINEAR, expand=True, fillcolor=int(rng.uniform(180, 255))
        )

    return out


# ---------------------------------------------------------------------------
# Normalisasi karakter
# ---------------------------------------------------------------------------
def crop_and_normalize(
    gray: np.ndarray, bbox: tuple[int, int, int, int], rng: random.Random
) -> np.ndarray | None:
    """Potong satu karakter lalu normalkan ke kanvas 32x32.

    Padding dan pergeseran acak kecil dipakai agar model tahan terhadap
    ketidaksempurnaan segmentasi saat inferensi.

    Augmentasi ``thin`` menipiskan stroke. Ini mensimulasikan font tipis
    pada struk thermal, yang setelah binarisasi bisa terpecah menjadi
    beberapa komponen. Tanpa augmentasi ini model gagal pada huruf tipis.
    """
    l, t, r, b = bbox
    h, w = gray.shape

    pad = rng.randint(0, 2)
    l2 = max(0, l - pad)
    t2 = max(0, t - pad)
    r2 = min(w - 1, r + pad)
    b2 = min(h - 1, b + pad)
    if r2 - l2 < 2 or b2 - t2 < 2:
        return None

    crop = gray[t2 : b2 + 1, l2 : r2 + 1]
    ch, cw = crop.shape
    scale = (C.CHAR_SIZE - 4) / max(ch, cw)
    nh, nw = max(1, int(round(ch * scale))), max(1, int(round(cw * scale)))

    pil = Image.fromarray(crop).resize((nw, nh), Image.BILINEAR)

    # Penipisan stroke: menaikkan kecerahan tepi sehingga stroke 1-2 px
    # bisa terputus, sama seperti yang terjadi pada font tipis.
    if rng.random() < 0.35:
        arr = np.asarray(pil).astype(np.float32)
        gamma = rng.uniform(0.55, 0.9)
        arr = 255.0 * (arr / 255.0) ** gamma
        pil = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    canvas = Image.new("L", (C.CHAR_SIZE, C.CHAR_SIZE), 255)
    ox = (C.CHAR_SIZE - nw) // 2 + rng.randint(-1, 1)
    oy = (C.CHAR_SIZE - nh) // 2 + rng.randint(-1, 1)
    canvas.paste(pil, (ox, oy))
    return np.asarray(canvas)


def normalize_char(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Versi deterministik untuk inferensi (tanpa padding/geser acak).

    Harus identik dengan crop_and_normalize tanpa komponen acak, supaya
    distribusi fitur saat latih dan saat uji sama.
    """
    l, t, r, b = bbox
    h, w = gray.shape
    l, t = max(0, l), max(0, t)
    r, b = min(w - 1, r), min(h - 1, b)
    crop = gray[t : b + 1, l : r + 1]
    if crop.size == 0:
        return np.full((C.CHAR_SIZE, C.CHAR_SIZE), 255, np.uint8)
    ch, cw = crop.shape
    scale = (C.CHAR_SIZE - 4) / max(ch, cw)
    nh, nw = max(1, int(round(ch * scale))), max(1, int(round(cw * scale)))
    pil = Image.fromarray(crop).resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (C.CHAR_SIZE, C.CHAR_SIZE), 255)
    canvas.paste(pil, ((C.CHAR_SIZE - nw) // 2, (C.CHAR_SIZE - nh) // 2))
    return np.asarray(canvas)


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------
def build(
    max_lines: int,
    samples_per_line: int,
    seed: int,
    out_prefix: str,
) -> dict:
    rng = random.Random(seed)
    np.random.seed(seed)

    df = pd.read_csv(C.DATASET_CSV)
    texts = [str(t) for t in df["receipt_text"].dropna().unique()]
    rng.shuffle(texts)
    texts = texts[:max_lines]

    fonts = []
    for name in C.FONTS:
        if not (C.FONT_DIR / name).exists():
            print(f"  ! font dilewati (tidak ada): {name}")
            continue
        for size in C.RENDER_SIZES:
            try:
                fonts.append((name, size, load_font(name, size)))
            except Exception as e:  # noqa: BLE001
                print(f"  ! font gagal dimuat {name} {size}: {e}")
    if not fonts:
        raise RuntimeError("Tidak ada font yang bisa dipakai.")
    print(f"  font aktif: {len(fonts)} varian")

    X: list[np.ndarray] = []
    y: list[int] = []
    skipped = 0

    for li, text in enumerate(texts):
        text = text.upper()
        for _ in range(samples_per_line):
            _, _, font = rng.choice(fonts)
            img, boxes = render_line(text, font)
            if not boxes:
                continue
            # Tanpa rotasi: bbox harus tetap valid untuk pelabelan.
            degraded = degrade(img, rng, allow_rotate=False)
            gray = np.asarray(degraded)
            for ch, bb in boxes:
                if ch not in C.CHAR_TO_IDX:
                    skipped += 1
                    continue
                patch = crop_and_normalize(gray, bb, rng)
                if patch is None:
                    skipped += 1
                    continue
                X.append(patch)
                y.append(C.CHAR_TO_IDX[ch])

        if (li + 1) % 200 == 0:
            print(f"    {li + 1}/{len(texts)} baris, {len(X)} sampel karakter")

    Xa = np.stack(X).astype(np.uint8) if X else np.zeros((0, C.CHAR_SIZE, C.CHAR_SIZE), np.uint8)
    ya = np.array(y, dtype=np.int32)

    path = C.DATA_DIR / f"{out_prefix}.npz"
    np.savez_compressed(path, X=Xa, y=ya)

    counts = np.bincount(ya, minlength=len(C.CHARSET))
    present = [(C.IDX_TO_CHAR[i], int(c)) for i, c in enumerate(counts) if c > 0]
    missing = [C.IDX_TO_CHAR[i] for i, c in enumerate(counts) if c == 0]

    print(f"  tersimpan: {path.name}  X={Xa.shape}  y={ya.shape}")
    print(f"  kelas hadir: {len(present)}/{len(C.CHARSET)}  dilewati: {skipped}")
    if missing:
        print(f"  kelas kosong: {''.join(missing)}")
    print(f"  sampel paling sedikit: {sorted(present, key=lambda t: t[1])[:6]}")

    return {"path": str(path), "n": int(len(ya)), "classes": len(present)}


def build_line_tests(n: int, seed: int) -> str:
    """Render beberapa baris utuh untuk uji end-to-end (dengan teks asli)."""
    rng = random.Random(seed + 1)
    df = pd.read_csv(C.DATASET_CSV)
    texts = [str(t).upper() for t in df["receipt_text"].dropna().unique()]
    rng.shuffle(texts)
    texts = texts[:n]

    imgs, labels = [], []
    for text in texts:
        name, size, font = rng.choice(
            [(n_, s_, load_font(n_, s_)) for n_ in C.FONTS if (C.FONT_DIR / n_).exists()
             for s_ in C.RENDER_SIZES]
        )
        img, _ = render_line(text, font)
        degraded = degrade(img, rng)
        imgs.append(np.asarray(degraded))
        labels.append(text)

    # Simpan sebagai array objek karena tinggi tiap baris berbeda.
    path = C.DATA_DIR / "lines_test.npz"
    np.savez_compressed(path, images=np.array(imgs, dtype=object), labels=np.array(labels))
    print(f"  tersimpan: {path.name}  n={len(labels)}")
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lines", type=int, default=1500,
                    help="jumlah baris unik yang dirender")
    ap.add_argument("--samples-per-line", type=int, default=2,
                    help="berapa kali tiap baris dirender (font acak)")
    ap.add_argument("--test-lines", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("[1/2] Membangun data latih karakter")
    info = build(args.max_lines, args.samples_per_line, args.seed, "chars_train")

    print("[2/2] Membangun data uji baris utuh")
    build_line_tests(args.test_lines, args.seed)

    print("\nRingkasan:", info)


if __name__ == "__main__":
    main()
