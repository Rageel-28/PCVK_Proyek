"""Buat visualisasi tahap pipeline untuk laporan.

Menghasilkan gambar tiap tahap pada satu contoh baris:
  asli -> grayscale -> biner -> segmentasi -> crop karakter
plus confusion matrix dan grafik ablasi.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
import deskew as DK
import paper as PPR
import preprocess as PP
import real_data as RD
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


def fig_real_dataset() -> tuple[str, str, str] | None:
    """Simpan ringkasan visual & metadata dataset karakter asli permanen."""
    path = C.DATA_DIR / "chars_real_gold.npz"
    if not path.exists():
        return None
    d = np.load(path)
    X, y = d["X"], d["y"]
    counts = Counter(C.IDX_TO_CHAR[int(v)] for v in y)

    # 1. Sheet contoh potongan karakter 32x32 per kelas (maks 12 contoh/kelas)
    rng = np.random.default_rng(42)
    ordered = [ch for ch in C.CHARSET if counts.get(ch, 0) > 0]
    n_cols = 12
    tile = 34
    row_h = tile + 4
    label_w = 96
    canvas_h = len(ordered) * row_h + 12
    canvas_w = label_w + n_cols * tile + 12
    sheet = np.full((canvas_h, canvas_w), 245, dtype=np.uint8)
    for ri, ch in enumerate(ordered):
        y0 = 6 + ri * row_h
        idxs = np.nonzero(y == C.CHAR_TO_IDX[ch])[0]
        picks = idxs if len(idxs) <= n_cols else rng.choice(idxs, n_cols, replace=False)
        cv2.putText(sheet, f"{ch} ({len(idxs)})", (8, y0 + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, 20, 1, cv2.LINE_AA)
        for ci, idx in enumerate(picks):
            x0 = label_w + ci * tile
            sheet[y0 + 1 : y0 + 33, x0 + 1 : x0 + 33] = X[idx]
    samples_png = OUT / "real_chars_samples.png"
    cv2.imwrite(str(samples_png), sheet)

    # 2. Grafik distribusi per kelas untuk laporan
    labels = ordered
    vals = [counts[ch] for ch in labels]
    fig, ax = plt.subplots(figsize=(13.5, 4.2))
    ax.bar(range(len(labels)), vals, color=BIRU, width=0.72)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Jumlah sampel asli")
    ax.set_title(
        f"Distribusi dataset karakter asli ({len(y)} potongan 32x32, {len(labels)} kelas, 0 sintetis)",
        fontweight="bold", color="#1F3864",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    dist_png = OUT / "real_chars_dist.png"
    fig.savefig(dist_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 3. Metadata lengkap sumber citra & label
    tr_path = C.ROOT / "transcripts_real.json"
    transcripts = (
        {k: v for k, v in json.loads(tr_path.read_text(encoding="utf-8")).items() if not k.startswith("_")}
        if tr_path.exists() else {}
    )
    imgs = {im.stem: im for im in RD.load_real_images(require_lines=False, stems=set(transcripts))}
    split_counts: dict[str, int] = {}
    sources_list = []
    for stem in sorted(transcripts):
        im = imgs.get(stem)
        split = im.split if im is not None else "local"
        split_counts[split] = split_counts.get(split, 0) + 1
        sources_list.append({
            "stem": stem,
            "split": split,
            "source": transcripts[stem].get("source", im.source if im else "?"),
            "kind": transcripts[stem].get("kind", "?"),
            "n_items_labels": len(transcripts[stem].get("items", [])),
            "n_full_labels": len(transcripts[stem].get("full", [])),
        })

    meta = {
        "dataset_file": str(path),
        "transcripts_file": str(tr_path),
        "total_chars": int(len(y)),
        "synthetic_chars": 0,
        "patch_size": [int(X.shape[1]), int(X.shape[2])],
        "active_classes": len(ordered),
        "class_counts": {ch: int(counts[ch]) for ch in ordered},
        "transcript_stems_total": len(transcripts),
        "transcript_splits": split_counts,
        "sources": sources_list,
    }
    meta_json = OUT / "real_chars_metadata.json"
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(samples_png), str(dist_png), str(meta_json)


def fig_real_receipt_stages(ocr: ClassicalOCR) -> str | None:
    """Visualisasi langkah preprocessing & segmentasi pada struk foto asli."""
    imgs = RD.load_real_images(splits=("valid",), require_lines=True)
    photo = next((im for im in imgs if im.is_photo and len(im.lines) >= 3), None)
    if photo is None:
        return None
    raw = cv2.imread(str(photo.path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None
    cropped, _ = PPR.crop_paper(raw, enabled=True)
    gray, ink, angle = ocr.binarize(raw)
    lboxes = SG.segment_lines(ink)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for lb in lboxes:
        cv2.rectangle(vis, (0, lb[0]), (vis.shape[1] - 1, lb[1]), (210, 140, 40), 1)
        for l, t, r, b in SG.segment_characters(ink, lb):
            cv2.rectangle(vis, (l, t), (r, b), (0, 110, 255), 1)

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 5.5))
    steps = [
        ("(a) Foto asli", raw),
        ("(b) Crop kertas + bersihkan tepi", cropped),
        (f"(c) Sauvola + deskew ({angle:+.1f}°)", 255 - ink),
        (f"(d) Segmentasi ({len(lboxes)} baris)", cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)),
    ]
    for ax, (title, im_step) in zip(axes, steps):
        if im_step.ndim == 3:
            ax.imshow(im_step)
        else:
            ax.imshow(im_step, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9, fontweight="bold", color="#1F3864")
        ax.axis("off")
    fig.suptitle(f"Pipeline PCVK pada struk foto asli ({photo.stem})",
                 fontsize=11, fontweight="bold", color="#1F3864")
    fig.tight_layout()
    out_p = OUT / "real_stages_photo.png"
    fig.savefig(out_p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out_p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-only", action="store_true",
                    help="hanya buat visualisasi dataset & foto nyata")
    args = ap.parse_args()

    ocr = ClassicalOCR(method="sauvola", use_deskew=True, use_lexicon=True)
    print(fig_confusion())
    print(fig_ablation())
    print(fig_real_dataset())
    print(fig_real_receipt_stages(ocr))

    if args.real_only:
        return

    lines_path = C.DATA_DIR / "lines_test.npz"
    if lines_path.exists():
        print("Memuat contoh baris uji...")
        images, labels = load_lines()
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


if __name__ == "__main__":
    main()
