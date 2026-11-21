"""Bangkitkan struk sintetis untuk **evaluasi held-out**.

Himpunan fixture ``receipts_v2`` dipakai terus selama pengembangan, jadi
angkanya bukan lagi perkiraan generalisasi. Modul ini membuat struk baru
dari korpus dengan renderer yang berbeda (font, ukuran, kemiringan, noise
acak), sehingga hasil di atasnya lebih jujur.

Yang divariasikan:

- font monospace dan proporsional,
- ukuran huruf 14-22 px,
- kemiringan -2.5..2.5 derajat,
- noise sensor dan blur ringan,
- pencahayaan tidak rata.

Keluaran: direktori berisi ``*.png`` dan ``ground_truth.json`` dengan format
yang sama seperti fixture, sehingga bisa dibaca langsung oleh
:mod:`ocr.evaluate`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

import ocr.config as C
from ocr.corpus import build_corpus
from ocr.render_dataset import find_fonts


# Struk thermal dicetak dengan font monospace. Font proporsional tetap
# disertakan sebagai uji ketahanan, tetapi porsi utamanya monospace supaya
# hasil evaluasi mewakili domain yang sebenarnya.
_MONOSPACE_HINTS = ("mono", "courier", "consol", "menlo", "monaco", "andale")


def available_fonts() -> tuple[list[str], list[str]]:
    """``(font monospace, font proporsional)`` yang ada di mesin ini."""
    monospace: list[str] = []
    proportional: list[str] = []
    for path in find_fonts():
        text = str(path).lower()
        if any(hint in text for hint in _MONOSPACE_HINTS):
            monospace.append(str(path))
        else:
            proportional.append(str(path))
    return monospace, proportional


def _load_font(path: str, size: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, size)


def render_lines(
    lines: list[str],
    font_path: str,
    font_size: int,
    width: int = 560,
    margin: int = 24,
) -> np.ndarray:
    """Render daftar baris menjadi citra grayscale struk."""
    from PIL import Image, ImageDraw

    font = _load_font(font_path, font_size)
    line_height = int(font_size * 1.9)
    height = margin * 2 + line_height * len(lines)

    image = Image.new("L", (width, height), color=245)
    draw = ImageDraw.Draw(image)
    for index, text in enumerate(lines):
        draw.text((margin, margin + index * line_height), text, fill=30, font=font)

    arr = np.array(image, dtype=np.uint8)
    return arr


def degrade(
    gray: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    """Tambahkan kemiringan, noise, blur, dan pencahayaan tidak rata."""
    angle = rng.uniform(-2.5, 2.5)
    center = (gray.shape[1] / 2 - 0.5, gray.shape[0] / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    out = cv2.warpAffine(
        gray, matrix, (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )

    sigma = rng.uniform(0.0, 0.6)
    if sigma > 0.05:
        out = cv2.GaussianBlur(out, (3, 3), sigma)

    # Pencahayaan tidak rata: gradien linier lembut.
    rows, cols = out.shape
    gradient = np.linspace(
        rng.uniform(0.85, 1.0), rng.uniform(0.85, 1.0), cols
    )[None, :]
    out = np.clip(out.astype(np.float32) * gradient, 0, 255)

    noise = np.random.normal(0, rng.uniform(2.0, 7.0), out.shape)
    out = np.clip(out + noise, 0, 255).astype(np.uint8)
    return out


def build_set(
    count: int,
    seed: int = 12345,
    lines_per_receipt: int = 12,
) -> list[tuple[np.ndarray, list[str]]]:
    """Bangkitkan ``count`` struk sintetis beserta teks acuannya."""
    monospace, proportional = available_fonts()
    fonts = monospace or proportional
    if not fonts:
        raise RuntimeError("tidak ada font yang tersedia untuk render")

    rng = random.Random(seed)
    corpus = build_corpus(max(500, count * 40), seed=seed)

    out: list[tuple[np.ndarray, list[str]]] = []
    for _ in range(count):
        start = rng.randrange(0, max(1, len(corpus) - lines_per_receipt))
        lines = corpus[start : start + lines_per_receipt]
        # 85% monospace (domain struk thermal), sisanya proporsional.
        if proportional and rng.random() > 0.85:
            font_path = rng.choice(proportional)
        else:
            font_path = rng.choice(fonts)
        gray = render_lines(
            lines,
            font_path=font_path,
            font_size=rng.randint(14, 22),
        )
        # Baris pemisah (``-----``) sengaja dirender tetapi tidak dimasukkan
        # ke ground truth: pipeline memang membuangnya, dan menilainya
        # sebagai "tidak terbaca" akan menghukum perilaku yang benar.
        expected = [line for line in lines if line.strip("-").strip() != ""]
        out.append((degrade(gray, rng), expected))
    return out


def write_set(items: list[tuple[np.ndarray, list[str]]], out_dir: Path) -> Path:
    """Tulis citra dan ``ground_truth.json`` ke direktori."""
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = []
    for index, (image, lines) in enumerate(items, start=1):
        name = f"synth_{index:03d}.png"
        cv2.imwrite(str(out_dir / name), image)
        truth.append({"image": name, "lines": lines})
    path = out_dir / "ground_truth.json"
    path.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Bangkitkan struk sintetis held-out.")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument(
        "--out", type=str, default=str(C.OUTPUT_DIR / "synth_receipts")
    )
    args = ap.parse_args()

    items = build_set(args.count, seed=args.seed)
    path = write_set(items, Path(args.out))
    print(f"{len(items)} struk -> {path.parent}")


if __name__ == "__main__":
    main()
