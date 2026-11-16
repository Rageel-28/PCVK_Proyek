"""Evaluasi OCR memakai potongan baris dari struk **asli**.

Tidak ada citra sintetis di sini. Setiap sampel adalah crop piksel nyata
yang diambil dari `ML/Dataset Citra` memakai bbox anotasi COCO.

Dua mode:

  items  (default)
      Untuk tiap bbox item, potong baris asli, jalankan pipeline pada crop
      itu saja, lalu bandingkan dengan transkripsi manual. Ini mengukur
      **kualitas pengenalan** tanpa dipengaruhi kesalahan segmentasi baris.

  doc
      Jalankan pipeline pada seluruh foto, lalu bandingkan gabungan baris
      dengan transkripsi penuh. Ini mengukur **kualitas menyeluruh**:
      segmentasi baris + pengenalan + rekonstruksi spasi.

Metrik: CER, WER, dan exact match. Levenshtein memakai implementasi
manual yang sama dengan leksikon, jadi bisa ditiru di Dart.

Jalankan:
    python evaluate_real.py
    python evaluate_real.py --mode doc
    python evaluate_real.py --split test
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import config as C
import real_data as RD
from lexicon import levenshtein
from ocr_pipeline import ClassicalOCR

TRANSCRIPTS = C.ROOT / "transcripts_real.json"


# ---------------------------------------------------------------------------
# Metrik
# ---------------------------------------------------------------------------
def cer(pred: str, truth: str) -> float:
    if not truth:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, truth) / len(truth)


def wer(pred: str, truth: str) -> float:
    p, t = pred.split(), truth.split()
    if not t:
        return 0.0 if not p else 1.0
    return levenshtein(p, t) / len(t)


@dataclass
class Sample:
    stem: str
    kind: str
    truth: str
    pred: str
    confidence: float
    crop: np.ndarray | None = field(default=None, repr=False)

    @property
    def cer(self) -> float:
        return cer(self.pred, self.truth)

    @property
    def wer(self) -> float:
        return wer(self.pred, self.truth)

    @property
    def exact(self) -> bool:
        return self.pred.strip() == self.truth.strip()


def load_transcripts() -> dict:
    data = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Mode: potongan baris item
# ---------------------------------------------------------------------------
def run_items(ocr: ClassicalOCR, split: str | None) -> list[Sample]:
    images = {im.stem: im for im in RD.load_real_images()}
    transcripts = load_transcripts()
    out: list[Sample] = []

    for stem, entry in transcripts.items():
        im = images.get(stem)
        if im is None:
            print(f"  ! tidak ada di indeks: {stem}")
            continue
        if split and im.split != split:
            continue
        truths = entry.get("items", [])
        if len(truths) != len(im.lines):
            print(
                f"  ! jumlah item tidak cocok: {stem} "
                f"({len(im.lines)} bbox vs {len(truths)} transkripsi) -- dilewati"
            )
            continue

        gray = cv2.imread(str(im.path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        for box, truth in zip(im.lines, truths):
            crop = RD.crop_line(gray, box, pad=2)
            if crop.size == 0:
                continue
            result = ocr.run(crop)
            out.append(
                Sample(
                    stem=stem,
                    kind=entry.get("kind", "?"),
                    truth=truth,
                    pred=result.text.strip().replace("\n", " "),
                    confidence=result.mean_confidence,
                    crop=crop,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Mode: potongan baris item, satu baris utama per crop
# ---------------------------------------------------------------------------
def apply_lexicon(ocr: ClassicalOCR, text: str) -> tuple[str, int]:
    """Terapkan koreksi leksikon seperti yang dilakukan aplikasi.

    ``read_line`` hanya mengembalikan hasil klasifikasi; koreksi kamus
    dilakukan di :meth:`ClassicalOCR.run`. Agar angka evaluasi mewakili
    aplikasi, langkah itu harus ikut dijalankan di sini.
    """
    if ocr.lexicon is None:
        return text, 0
    return ocr.lexicon.correct_text(text)


def run_items_primary(
    ocr: ClassicalOCR, split: str | None, use_lexicon: bool
) -> list[Sample]:
    """Sama seperti :func:`run_items`, tetapi bila crop berisi lebih dari
    satu pita (teks + garis pemisah) hanya pita terpanjang yang dibaca.

    Crop berasal dari bbox COCO, jadi lokalisasi tetap memakai anotasi
    manusia; pemilihan pita dilakukan oleh pengolahan citra.
    """
    from segment import segment_lines

    images = {im.stem: im for im in RD.load_real_images()}
    transcripts = load_transcripts()
    out: list[Sample] = []

    for stem, entry in transcripts.items():
        im = images.get(stem)
        if im is None or (split and im.split != split):
            continue
        truths = entry.get("items", [])
        if len(truths) != len(im.lines):
            continue

        gray = cv2.imread(str(im.path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        for box, truth in zip(im.lines, truths):
            crop = RD.crop_line(gray, box, pad=2)
            if crop.size == 0:
                continue

            prepared_gray, ink, _angle = ocr.binarize(crop)
            lines = segment_lines(ink)
            primary = RD.pick_primary_line(lines)
            if primary is None:
                continue

            pred = ocr.read_line(prepared_gray, ink, primary)
            text = pred.text.strip()
            if use_lexicon:
                text, _fixes = apply_lexicon(ocr, text)
            out.append(
                Sample(
                    stem=stem,
                    kind=entry.get("kind", "?"),
                    truth=truth,
                    pred=text,
                    confidence=pred.confidence,
                    crop=crop,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Mode: dokumen penuh
# ---------------------------------------------------------------------------
def run_doc(ocr: ClassicalOCR, split: str | None) -> list[Sample]:
    images = {im.stem: im for im in RD.load_real_images()}
    transcripts = load_transcripts()
    out: list[Sample] = []

    for stem, entry in transcripts.items():
        im = images.get(stem)
        if im is None or (split and im.split != split):
            continue
        gray = cv2.imread(str(im.path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        result = ocr.run(gray)
        out.append(
            Sample(
                stem=stem,
                kind=entry.get("kind", "?"),
                truth="\n".join(entry["full"]),
                pred=result.text,
                confidence=result.mean_confidence,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Ringkasan
# ---------------------------------------------------------------------------
def summarize(samples: list[Sample], label: str) -> dict:
    if not samples:
        return {"label": label, "n": 0}
    cers = [s.cer for s in samples]
    wers = [s.wer for s in samples]
    return {
        "label": label,
        "n": len(samples),
        "cer": float(np.mean(cers)),
        "cer_median": float(statistics.median(cers)),
        "wer": float(np.mean(wers)),
        "exact_match": sum(1 for s in samples if s.exact) / len(samples),
        "mean_confidence": float(np.mean([s.confidence for s in samples])),
    }


def print_table(rows: list[dict]) -> None:
    print(
        f"{'kelompok':26s} {'n':>4s} {'CER':>8s} {'WER':>8s} "
        f"{'exact':>8s} {'conf':>7s}"
    )
    print("-" * 66)
    for r in rows:
        if r["n"] == 0:
            print(f"{r['label']:26s} {0:4d}  (tidak ada sampel)")
            continue
        print(
            f"{r['label']:26s} {r['n']:4d} {r['cer'] * 100:7.2f}% "
            f"{r['wer'] * 100:7.2f}% {r['exact_match'] * 100:7.2f}% "
            f"{r['mean_confidence'] * 100:6.2f}%"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["items", "items1", "doc"], default="items1",
                    help="items1 = satu baris utama per crop (disarankan)")
    ap.add_argument("--split", default=None,
                    help="batasi ke satu split COCO: train/valid/test")
    ap.add_argument("--method", default="sauvola",
                    choices=["sauvola", "otsu", "adaptive"])
    ap.add_argument("--lexicon", action="store_true",
                    help="aktifkan koreksi kamus (default mati; lihat catatan "
                         "di ClassicalOCR.__init__)")
    ap.add_argument("--show", type=int, default=8,
                    help="tampilkan N contoh terburuk")
    args = ap.parse_args()

    print(f"Mode: {args.mode}   split: {args.split or 'semua'}")
    ocr = ClassicalOCR(
        method=args.method,
        use_deskew=True,
        use_lexicon=args.lexicon,
    )

    samples = run_items(ocr, args.split) if args.mode == "items" else run_doc(ocr, args.split)
    if args.mode == "items1":
        samples = run_items_primary(ocr, args.split, use_lexicon=args.lexicon)
    if not samples:
        print("Tidak ada sampel. Periksa ML/Dataset Citra dan kedua ZIP anotasi.")
        return

    rows = [summarize(samples, "SEMUA")]
    for kind in sorted({s.kind for s in samples}):
        rows.append(summarize([s for s in samples if s.kind == kind], f"jenis={kind}"))
    for stem in sorted({s.stem for s in samples}):
        group = [s for s in samples if s.stem == stem]
        rows.append(summarize(group, f"  {stem[:22]}"))

    print()
    print_table(rows)

    worst = sorted(samples, key=lambda s: -s.cer)[: args.show]
    if worst:
        print(f"\n=== {len(worst)} contoh CER terburuk ===")
        for s in worst:
            print(f"\n[{s.stem} | {s.kind}] CER={s.cer * 100:.1f}% conf={s.confidence:.2f}")
            print(f"  asli : {s.truth}")
            print(f"  ocr  : {s.pred}")

    out_path = C.OUTPUT_DIR / f"real_{args.mode}.json"
    out_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "split": args.split,
                "method": args.method,
                "lexicon": args.lexicon,
                "summary": rows,
                "samples": [
                    {
                        "stem": s.stem,
                        "kind": s.kind,
                        "truth": s.truth,
                        "pred": s.pred,
                        "cer": round(s.cer, 4),
                        "wer": round(s.wer, 4),
                        "confidence": round(s.confidence, 4),
                    }
                    for s in samples
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nTersimpan: {out_path}")


if __name__ == "__main__":
    main()
