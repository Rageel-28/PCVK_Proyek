"""Sumber citra untuk pelabelan: dataset v10, hanya struk Indonesia.

Kenapa v10, bukan v2i
---------------------
`OCR Struk Belanja.v2i.coco` (771 citra) hampir seluruhnya struk Malaysia
dan Amerika. `OCR - Struk Belanja.v10-raw-dataset.coco` (1.505 citra)
memuat 71 struk Indonesia yang teridentifikasi dari nama file, termasuk
Indomaret, Toko H. Endang, Kasir Pintar, dan Santa Supermarket.

Kenapa disaring dari nama file
------------------------------
Sebagian besar citra v10 bernama angka polos (`0_jpg`, `100_jpg`) dan
isinya campur Malaysia/AS/Indonesia. Tidak ada metadata bahasa, dan
heuristik struktural (rasio aspek, kepadatan tinta) tidak dapat memisahkan
keduanya dengan andal. Jadi strok Indonesia diambil dari nama file yang
mengandung kata penanda.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASET = ROOT / "ML" / "OCR - Struk Belanja.v10-raw-dataset.coco"

# Kata penanda struk Indonesia pada nama file. "struk" dan "nota" adalah
# penanda terkuat; sisanya menangkap merek toko.
INDONESIA_MARKERS = (
    "struk",
    "nota",
    "indomaret",
    "alfamart",
    "alfamidi",
    "kasir",
)

SPLITS = ("train", "valid", "test")


def is_indonesian(name: str) -> bool:
    low = name.lower()
    return any(marker in low for marker in INDONESIA_MARKERS)


def indonesian_files(split: str = "train") -> list[Path]:
    """Citra struk Indonesia pada satu split, terurut."""
    folder = DATASET / split
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.jpg") if is_indonesian(p.name))


def all_indonesian_files() -> list[Path]:
    """Semua citra struk Indonesia dari ketiga split."""
    out: list[Path] = []
    for split in SPLITS:
        out.extend(indonesian_files(split))
    return out


def count_by_split() -> dict[str, tuple[int, int]]:
    """{'train': (indonesia, total), ...} untuk laporan."""
    result = {}
    for split in SPLITS:
        folder = DATASET / split
        total = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        result[split] = (len(indonesian_files(split)), total)
    return result


if __name__ == "__main__":
    counts = count_by_split()
    print(f"dataset: {DATASET.name}\n")
    total_id = total_all = 0
    for split, (n_id, n_all) in counts.items():
        print(f"  {split:6s} {n_id:4d} Indonesia / {n_all:5d} total")
        total_id += n_id
        total_all += n_all
    print(f"  {'TOTAL':6s} {total_id:4d} Indonesia / {total_all:5d} total")
