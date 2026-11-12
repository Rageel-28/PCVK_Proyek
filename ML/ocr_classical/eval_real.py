"""Ukur CER/WER pipeline OCR pada gambar struk nyata.

Membandingkan teks hasil OCR dengan ground truth, lalu melaporkan:
  - CER (character error rate)
  - WER (word error rate)
  - akurasi per baris

Dipakai untuk mengukur dampak augmentasi: angka ini adalah yang penting,
bukan akurasi validasi sintetis (yang selalu ~99% karena distribusinya
sama dengan data latih).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import config as C
import ocr_pipeline as P


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref = ref.strip()
    if not ref:
        return 0.0 if not hyp.strip() else 1.0
    return levenshtein(ref, hyp.strip()) / len(ref)


def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.strip().split()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="JSON: [{image: path, lines: [teks baris...]}, ...]")
    ap.add_argument("--out", default=None, help="simpan hasil per gambar")
    ap.add_argument("--show", action="store_true", help="cetak teks OCR tiap gambar")
    ap.add_argument("--method", default="sauvola",
                    choices=["sauvola", "otsu", "adaptive"])
    ap.add_argument("--upscale", type=float, default=1.5,
                    help="paksa faktor upscale (maksimum)")
    args = ap.parse_args()

    entries = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    # Paksa faktor upscale agar eksperimen bisa membandingkan setelan.
    C.MAX_UPSCALE = args.upscale
    C.TARGET_LINE_HEIGHT = 999.0
    ocr = P.ClassicalOCR(method=args.method)
    rows = []

    for e in entries:
        img_path = Path(e["image"])
        if not img_path.exists():
            print(f"  ! tidak ada: {img_path}")
            continue
        if cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE) is None:
            print(f"  ! gagal dekode: {img_path}")
            continue

        got = ocr.run_file(img_path).text
        ref_lines = e["lines"]

        if args.show:
            print(f"\n##### {img_path.name} #####")
            print(got)

        # Bandingkan per baris acuan terhadap baris OCR TERBAIK.
        # Ini metrik yang benar untuk OCR struk: urutan dan jumlah baris
        # pada keluaran OCR tidak harus sama, yang penting setiap baris
        # acuan (item, total, PPN) berhasil dibaca pada suatu baris.
        hyp_lines = [l for l in got.splitlines() if l.strip()]
        norm = lambda s: " ".join(s.upper().split())

        line_cer = []
        for r in ref_lines:
            rn = norm(r)
            best = min((cer(rn, norm(h)) for h in hyp_lines), default=1.0)
            line_cer.append(best)

        avg = float(np.mean(line_cer)) if line_cer else 1.0
        rows.append({"image": str(img_path), "cer": avg, "per_line": line_cer})
        exact = sum(1 for c in line_cer if c == 0.0)
        print(f"{img_path.name:28s} CER={avg*100:6.2f}%  "
              f"baris tepat={exact}/{len(ref_lines)}")

    if not rows:
        print("tidak ada gambar yang diukur")
        return

    mc = float(np.mean([r["cer"] for r in rows]))
    all_line = [c for r in rows for c in r["per_line"]]
    exact_rate = sum(1 for c in all_line if c == 0.0) / len(all_line)
    print(f"\nRATA-RATA  CER={mc*100:.2f}%  "
          f"baris tepat={exact_rate*100:.1f}%  (n={len(rows)} gambar, "
          f"{len(all_line)} baris)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"mean_cer": mc, "exact_line_rate": exact_rate,
                        "items": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"tersimpan: {args.out}")


if __name__ == "__main__":
    main()
