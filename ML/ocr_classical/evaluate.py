"""Evaluasi pipeline OCR: CER, WER, dan ablasi per tahap.

Dua jenis evaluasi:
  1. End-to-end pada baris sintetis (data/lines_test.npz) -- ada teks acuan
  2. Ablasi: matikan/nyalakan tiap tahap untuk melihat kontribusinya

Metrik:
  CER = jarak edit karakter / jumlah karakter acuan
  WER = jarak edit kata      / jumlah kata acuan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config as C
from lexicon import levenshtein
from ocr_pipeline import ClassicalOCR


# ---------------------------------------------------------------------------
# Metrik
# ---------------------------------------------------------------------------
def cer(pred: str, truth: str) -> float:
    if not truth:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, truth) / len(truth)


def wer(pred: str, truth: str) -> float:
    p = pred.split()
    t = truth.split()
    if not t:
        return 0.0 if not p else 1.0
    return levenshtein(p, t) / len(t)


# ---------------------------------------------------------------------------
# Dataset uji
# ---------------------------------------------------------------------------
def load_lines(path: Path | None = None):
    p = path or (C.DATA_DIR / "lines_test.npz")
    if not p.exists():
        raise FileNotFoundError(f"{p} belum ada. Jalankan: python render_synthetic.py")
    d = np.load(p, allow_pickle=True)
    return list(d["images"]), [str(s) for s in d["labels"]]


# ---------------------------------------------------------------------------
# Ablasi
# ---------------------------------------------------------------------------
CONFIGS = [
    {"name": "otsu", "method": "otsu", "deskew": False, "lexicon": False},
    {"name": "sauvola", "method": "sauvola", "deskew": False, "lexicon": False},
    {"name": "otsu+deskew", "method": "otsu", "deskew": True, "lexicon": False},
    {"name": "otsu+deskew+lexicon", "method": "otsu", "deskew": True, "lexicon": True},
    {"name": "sauvola+deskew+lexicon", "method": "sauvola", "deskew": True, "lexicon": True},
]


def run_config(ocr: ClassicalOCR, cfg: dict, images, labels, verbose: bool) -> dict:
    ocr.method = cfg["method"]
    ocr.use_deskew = cfg["deskew"]
    ocr.use_lexicon = cfg["lexicon"]

    cers, wers, confs, exact = [], [], [], 0
    samples = []

    for img, truth in zip(images, labels):
        res = ocr.run(img)
        c = cer(res.text, truth)
        w = wer(res.text, truth)
        cers.append(c)
        wers.append(w)
        confs.append(res.mean_confidence)
        if res.text.strip() == truth.strip():
            exact += 1
        if verbose and len(samples) < 5:
            samples.append({"truth": truth, "pred": res.text, "cer": round(c, 3)})

    n = max(len(cers), 1)
    out = {
        "name": cfg["name"],
        "method": cfg["method"],
        "deskew": cfg["deskew"],
        "lexicon": cfg["lexicon"],
        "n": len(cers),
        "cer": float(np.mean(cers)),
        "wer": float(np.mean(wers)),
        "exact_match": exact / n,
        "mean_confidence": float(np.mean(confs)),
    }
    if samples:
        out["samples"] = samples
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="batasi jumlah baris uji (0 = semua)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("Memuat data uji dan model...")
    images, labels = load_lines()
    if args.limit:
        images, labels = images[: args.limit], labels[: args.limit]
    print(f"  {len(images)} baris uji")

    ocr = ClassicalOCR(method="sauvola", use_deskew=True, use_lexicon=True)

    results = []
    for cfg in CONFIGS:
        print(f"\n[{cfg['name']}]")
        r = run_config(ocr, cfg, images, labels, args.verbose)
        results.append(r)
        print(
            f"  CER={r['cer'] * 100:5.2f}%  WER={r['wer'] * 100:5.2f}%  "
            f"exact={r['exact_match'] * 100:5.2f}%  conf={r['mean_confidence'] * 100:5.2f}%"
        )

    out_path = C.OUTPUT_DIR / "ablation.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RINGKASAN ABLASI ===")
    print(f"{'konfigurasi':28} {'CER':>8} {'WER':>8} {'exact':>8} {'conf':>8}")
    for r in results:
        print(
            f"{r['name']:28} {r['cer'] * 100:7.2f}% {r['wer'] * 100:7.2f}% "
            f"{r['exact_match'] * 100:7.2f}% {r['mean_confidence'] * 100:7.2f}%"
        )
    print(f"\nTersimpan: {out_path}")


if __name__ == "__main__":
    main()
