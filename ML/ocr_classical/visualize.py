"""Buat visualisasi tahap pipeline untuk laporan.

Menghasilkan gambar tiap tahap pada satu contoh baris:
  asli -> grayscale -> biner -> segmentasi -> crop karakter
plus confusion matrix dan grafik ablasi.
"""

from __future__ import annotations

import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
import deskew as DK
import paper as PPR
import preprocess as PP
import segment as SG
from evaluate import load_lines
from ocr_pipeline import ClassicalOCR

BIRU = "#2E75B6"
ORANYE = "#C55A11"
OUT = C.OUTPUT_DIR


def fig_stages(ocr: ClassicalOCR, image: np.ndarray, truth: str, tag: str) -> str:
    """Gambar 6 panel: tiap tahap pemrosesan satu baris."""
    gray = PP.to_gray(image)
    gray_p, _ = PPR.crop_paper(gray)
    den = PP.denoise(gray_p, 3)
    ink_otsu = PP.binarize(den, "otsu")
    ink_sauvola = PP.binarize(den, "sauvola")
    ink_clean = PP.remove_small_components(ink_sauvola, C.CHAR_MIN_INK)
    desk, angle = DK.auto_deskew(ink_clean)
    lines = SG.segment_lines(desk)
    boxes = SG.segment_characters(desk, lines[0]) if lines else []

    vis = cv2.cvtColor(desk, cv2.COLOR_GRAY2BGR)
    for l, t, r, b in boxes:
        cv2.rectangle(vis, (l, t), (r, b), (0, 140, 255), 1)

    panels = [
        ("(a) Citra asli", gray, None),
        ("(b) Denoise", den, None),
        ("(c) Binarisasi Otsu", ink_otsu, None),
        ("(d) Binarisasi Sauvola", ink_sauvola, None),
        (f"(e) Deskew ({angle:+.1f}°)", desk, None),
        (f"(f) Segmentasi ({len(boxes)} karakter)", vis, None),
    ]

    fig, axes = plt.subplots(6, 1, figsize=(8.2, 9.0))
    for ax, (title, img, _) in zip(axes, panels):
        if img.ndim == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9, loc="left", color="#1F3864", fontweight="bold")
        ax.axis("off")
    fig.suptitle(f'Tahapan pipeline OCR — teks acuan: "{truth}"',
                 fontsize=11, fontweight="bold", color="#1F3864", y=0.995)
    fig.tight_layout()
    p = OUT / f"stages_{tag}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(p)


def fig_confusion() -> str | None:
    cm_path = OUT / "char_confusion.npy"
    lab_path = OUT / "char_confusion_labels.json"
    if not cm_path.exists() or not lab_path.exists():
        return None
    cm = np.load(cm_path)
    labels = json.loads(lab_path.read_text(encoding="utf-8"))
    n = len(labels)

    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

    size = max(7.0, n * 0.42)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if v >= 5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if v > 60 else "#1F3864")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion matrix karakter (persen per baris)",
                 fontweight="bold", color="#1F3864")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Persen (%)")
    p = OUT / "char_confusion.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(p)


def fig_ablation() -> str | None:
    path = OUT / "ablation.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))

    names = [d["name"] for d in data]
    cer = [d["cer"] * 100 for d in data]
    wer = [d["wer"] * 100 for d in data]
    exact = [d["exact_match"] * 100 for d in data]

    x = np.arange(len(names))
    w = 0.27

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    b1 = ax.bar(x - w, cer, w, label="CER", color=ORANYE)
    b2 = ax.bar(x, wer, w, label="WER", color="#9DC3E6")
    b3 = ax.bar(x + w, exact, w, label="Exact match", color=BIRU)
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                    f"{b.get_height():.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=16, ha="right", fontsize=8)
    ax.set_ylabel("Persen (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Ablasi: kontribusi setiap tahap pipeline",
                 fontweight="bold", color="#1F3864")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = OUT / "ablation.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(p)


def main() -> None:
    print("Memuat model dan data uji...")
    images, labels = load_lines()

    ocr = ClassicalOCR(method="sauvola", use_deskew=True, use_lexicon=True)

    # Pilih contoh yang berhasil dan yang gagal.
    from evaluate import cer
    results = []
    for i in range(min(40, len(images))):
        r = ocr.run(images[i])
        results.append((cer(r.text, labels[i]), i, r))

    results.sort()
    best = results[0]
    worst = results[-1]

    print(f"  contoh terbaik: {labels[best[1]]!r} CER={best[0]*100:.1f}%")
    print(f"  contoh terburuk: {labels[worst[1]]!r} CER={worst[0]*100:.1f}%")

    print(fig_stages(ocr, images[best[1]], labels[best[1]], "best"))
    print(fig_stages(ocr, images[worst[1]], labels[worst[1]], "worst"))
    print(fig_confusion())
    print(fig_ablation())


if __name__ == "__main__":
    main()
