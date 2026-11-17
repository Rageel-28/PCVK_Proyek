"""Reproduksi bug tampilan alat pelabelan: citra kecil & baris raksasa.

Gejala yang dilaporkan:
  - citra di panel atas hanya menempati sepetak kecil dari jendela
  - panel "baris aktif" menampilkan BANYAK baris sekaligus (bukan satu)

Dua dugaan penyebab, diuji terpisah:
  A. `canvas.winfo_width()` mengembalikan 1 sebelum jendela di-layout,
     sehingga `1 or 1240` tetap 1 dan skala gambar menjadi sangat kecil.
  B. Deteksi baris menggabungkan beberapa baris menjadi satu pita karena
     latar gelap / batas hasil rotasi ikut terhitung sebagai tinta.

Pemakaian:
    python diag_label_gui_bug.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OCR = ROOT / "ML" / "ocr_classical"
sys.path.insert(0, str(OCR))
sys.path.insert(0, str(HERE))

import dataset_source as DS  # noqa: E402
import label_lines as L  # noqa: E402
import segment as SG  # noqa: E402


def probe(name_fragment: str) -> None:
    index = L.build_index()
    name = next((n for n in index if name_fragment in n), None)
    if name is None:
        print(f"tidak ditemukan: {name_fragment}")
        return

    gray, ink, scale = L.load_mask(index[name])
    h, w = gray.shape
    print("=" * 72)
    print(f"berkas : {name[:58]}")
    print(f"ukuran : {w}x{h}  (skala {scale:.2f}, sudut "
          f"{L.state_angles.get(name, 0.0)} deg)")
    ink_ratio = float((ink > 0).mean())
    print(f"rasio tinta pada mask: {ink_ratio:.4f}")

    # Baris mentah dari segmentasi
    raw = SG.segment_lines(ink)
    print(f"\nbaris mentah (segment_lines): {len(raw)}")
    heights = [b - a + 1 for a, b in raw]
    if heights:
        print(f"  tinggi: min={min(heights)} median={int(np.median(heights))} "
              f"max={max(heights)}")

    # Baris setelah filter usable_lines
    usable = L.usable_lines(ink)
    print(f"\nbaris layak (usable_lines): {len(usable)}")
    for i, (top, bottom, n) in enumerate(usable[:12]):
        height = bottom - top + 1
        cols = np.nonzero((ink[top:bottom + 1, :] > 0).sum(axis=0) > 0)[0]
        span = int(cols[-1] - cols[0] + 1) if len(cols) else 0
        flag = ""
        if height > 90:
            flag += " [TINGGI]"
        if n > 70:
            flag += " [GLYPH BANYAK]"
        if span > w * 0.9:
            flag += " [LEBAR PENUH]"
        print(f"  {i:2d}: y={top}-{bottom} h={height:3d} lebar={span:4d} "
              f"glyph={n:3d}{flag}")

    # Apakah tepi citra ikut jadi tinta (ciri latar gelap / border rotasi)?
    border = np.concatenate([
        ink[0, :], ink[-1, :], ink[:, 0], ink[:, -1]
    ])
    print(f"\ntinta di tepi citra: {float((border > 0).mean()) * 100:.1f}% "
          f"(tinggi = latar/border ikut terhitung)")

    # Dugaan A: perilaku `x or fallback`
    print("\n--- dugaan A: canvas.winfo_width() sebelum layout ---")
    for value in (0, 1, 1240):
        chosen = value or 1240
        scale_img = min(chosen / w, (chosen * 0.42) / h, 1.0)
        print(f"  winfo_width()={value:4d} -> dipakai {chosen:4d} "
              f"-> skala {scale_img:.3f} (lebar hasil {int(w * scale_img)} px)")


def main() -> None:
    probe("struk-17")
    print()
    probe("89120d1aa3d849d0d7fe128b14b96e")


if __name__ == "__main__":
    main()
