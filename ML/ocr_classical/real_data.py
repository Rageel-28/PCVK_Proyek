"""Akses dataset struk **asli** beserta anotasi barisnya.

Sumber data:
  ML/Dataset Citra/                         622 foto struk Indomaret/Alfamart
  ML/Receipt Indomaret.v1i.coco.zip         500 gambar, 2.157 bbox baris
  ML/Alfamart Indomaret Struk.v1i.coco.zip  122 gambar, 630 bbox baris

Kenapa modul ini ada
--------------------
Evaluasi dan pelaporan proyek ini **wajib** memakai struk asli. Data
sintetis hanya dipakai sebagai pelengkap latih, bukan sebagai bahan
pengukuran. Tanpa modul ini, satu-satunya angka yang bisa dilaporkan
berasal dari citra render sendiri -- yang tidak membuktikan apa pun
tentang kemampuan membaca struk sungguhan.

Catatan penting soal dua export
-------------------------------
Folder `Dataset Citra` dan kedua ZIP berasal dari project Roboflow yang
sama tetapi diekspor berbeda, sehingga akhiran `.rf.<hash>` tidak sama.
Pencocokan dilakukan lewat *stem* sebelum `.rf.` (mis.
`receipt-000_png`, `CamScanner-18-11-2024-12-48_7_jpg`).

`Receipt Indomaret` diekspor dengan *Resize 640x640 (Stretch)*, jadi
koordinat bbox harus dipetakan balik ke ukuran piksel asli. Stretch
bersifat terpisah per sumbu, sehingga pemetaan tetap eksak.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config as C

REAL_ROOT = C.ML_DIR / "Dataset Citra"

SOURCES = {
    "receipt_indomaret": C.ML_DIR / "Receipt Indomaret.v1i.coco.zip",
    "alfamart_indomaret": C.ML_DIR / "Alfamart Indomaret Struk.v1i.coco.zip",
}

SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class LineBox:
    """Satu bbox baris teks dalam koordinat piksel citra lokal."""

    left: int
    top: int
    right: int
    bottom: int
    category: str

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass(frozen=True)
class RealImage:
    """Satu foto struk asli + anotasi barisnya."""

    path: Path
    stem: str
    source: str
    split: str
    width: int
    height: int
    lines: tuple[LineBox, ...]

    @property
    def is_photo(self) -> bool:
        """True bila berasal dari kamera, bukan render/template."""
        return self.stem.startswith("CamScanner")


def _stem(name: str) -> str:
    """Buang akhiran `.rf.<hash>` dan ekstensi."""
    base = Path(name).name
    idx = base.find(".rf.")
    if idx >= 0:
        base = base[:idx]
    return base


def _local_index() -> dict[str, Path]:
    """Peta stem -> path gambar lokal."""
    if not REAL_ROOT.exists():
        return {}
    out: dict[str, Path] = {}
    for p in REAL_ROOT.iterdir():
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        out.setdefault(_stem(p.name), p)
    return out


def _read_coco(zf: zipfile.ZipFile, member: str) -> dict:
    with zf.open(member) as fh:
        return json.loads(fh.read().decode("utf-8"))


def load_real_images(
    sources: tuple[str, ...] | None = None,
    splits: tuple[str, ...] = SPLITS,
    require_lines: bool = True,
    stems: set[str] | None = None,
) -> list[RealImage]:
    """Muat seluruh foto asli yang punya anotasi baris.

    Bbox dipetakan ke koordinat piksel gambar **lokal** (yang tidak
    terdistorsi), bukan ke gambar di dalam ZIP.
    """
    local = _local_index()
    if not local:
        return []

    out: list[RealImage] = []
    for source_name in (sources or tuple(SOURCES)):
        zip_path = SOURCES[source_name]
        if not zip_path.exists():
            continue
        with zipfile.ZipFile(zip_path) as zf:
            for split in splits:
                member = f"{split}/_annotations.coco.json"
                if member not in zf.namelist():
                    continue
                coco = _read_coco(zf, member)
                cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
                by_image: dict[int, list[dict]] = {}
                for ann in coco.get("annotations", []):
                    by_image.setdefault(ann["image_id"], []).append(ann)

                for info in coco.get("images", []):
                    stem = _stem(info["file_name"])
                    if stems is not None and stem not in stems:
                        continue
                    path = local.get(stem)
                    if path is None:
                        continue

                    try:
                        with Image.open(path) as im_hdr:
                            w, h = im_hdr.size
                    except Exception:  # noqa: BLE001
                        continue

                    sx = w / float(info["width"])
                    sy = h / float(info["height"])

                    boxes: list[LineBox] = []
                    for ann in by_image.get(info["id"], []):
                        x, y, bw, bh = ann["bbox"]
                        left = int(round(x * sx))
                        top = int(round(y * sy))
                        right = int(round((x + bw) * sx)) - 1
                        bottom = int(round((y + bh) * sy)) - 1
                        left = max(0, min(left, w - 1))
                        top = max(0, min(top, h - 1))
                        right = max(left, min(right, w - 1))
                        bottom = max(top, min(bottom, h - 1))
                        if right - left < 4 or bottom - top < 3:
                            continue
                        boxes.append(
                            LineBox(
                                left=left,
                                top=top,
                                right=right,
                                bottom=bottom,
                                category=cats.get(ann["category_id"], "?"),
                            )
                        )

                    if require_lines and not boxes:
                        continue

                    boxes.sort(key=lambda b: b.top)
                    out.append(
                        RealImage(
                            path=path,
                            stem=stem,
                            source=source_name,
                            split=split,
                            width=w,
                            height=h,
                            lines=tuple(boxes),
                        )
                    )

    out.sort(key=lambda r: (r.source, r.split, r.stem))
    return out


# ---------------------------------------------------------------------------
# Pemotongan baris
# ---------------------------------------------------------------------------
def crop_line(gray: np.ndarray, box: LineBox, pad: int = 2) -> np.ndarray:
    """Potong satu baris dari citra grayscale dengan sedikit margin.

    ``pad=2`` dipilih dari pengukuran: menambah bantalan putih lebih lebar
    justru menurunkan akurasi (31-37% CER vs 28,75%), karena pipeline
    sudah melakukan normalisasi sendiri per karakter.
    """
    h, w = gray.shape
    l = max(0, box.left - pad)
    t = max(0, box.top - pad)
    r = min(w - 1, box.right + pad)
    b = min(h - 1, box.bottom + pad)
    if r <= l or b <= t:
        return np.full((1, 1), 255, np.uint8)
    return gray[t : b + 1, l : r + 1]


def trim_ink_rows(crop: np.ndarray, ink_ratio: float = 0.01) -> np.ndarray:
    """Pangkas baris tepi yang hampir tidak mengandung tinta.

    Tidak dipakai di jalur evaluasi utama (lihat catatan di
    :func:`crop_line`), tetapi berguna untuk analisis manual.
    """
    if crop.size == 0:
        return crop
    dark = (crop < 128).mean(axis=1)
    rows = np.nonzero(dark > ink_ratio)[0]
    if len(rows) == 0:
        return crop
    top = int(rows[0])
    bottom = int(rows[-1])
    if bottom <= top:
        return crop
    return crop[top : bottom + 1, :]



def pick_primary_line(lines: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Pilih satu baris utama dari hasil segmentasi di dalam crop.

    Bbox COCO kadang menyentuh garis pemisah ``-----`` di bawah baris item,
    sehingga crop berisi dua pita: teks setinggi ~85 px dan garis setipis
    ~10 px. Baris terpanjang dipilih karena teks selalu jauh lebih tinggi
    daripada garis pemisah.
    """
    if not lines:
        return None
    return max(lines, key=lambda b: b[1] - b[0] + 1)


def summary(images: list[RealImage]) -> dict:
    """Ringkasan jumlah gambar/baris per sumber, split, dan jenis."""
    def bucket(key) -> dict:
        acc: dict[str, list[int]] = {}
        for im in images:
            k = key(im)
            entry = acc.setdefault(k, [0, 0])
            entry[0] += 1
            entry[1] += len(im.lines)
        return {k: {"images": v[0], "lines": v[1]} for k, v in sorted(acc.items())}

    return {
        "total": {
            "images": len(images),
            "lines": sum(len(im.lines) for im in images),
        },
        "per_source": bucket(lambda im: im.source),
        "per_split": bucket(lambda im: im.split),
        "per_kind": bucket(lambda im: "foto" if im.is_photo else "render"),
    }


def main() -> None:
    images = load_real_images()
    if not images:
        print("Tidak ada gambar asli yang cocok. Periksa ML/Dataset Citra dan ZIP.")
        return
    s = summary(images)
    print(f"Gambar asli  : {s['total']['images']}")
    print(f"Bbox baris   : {s['total']['lines']}")
    print("\nPer sumber:")
    for k, v in s["per_source"].items():
        print(f"  {k:20s} {v['images']:4d} gambar  {v['lines']:5d} baris")
    print("\nPer split:")
    for k, v in s["per_split"].items():
        print(f"  {k:20s} {v['images']:4d} gambar  {v['lines']:5d} baris")
    print("\nPer jenis:")
    for k, v in s["per_kind"].items():
        print(f"  {k:20s} {v['images']:4d} gambar  {v['lines']:5d} baris")


if __name__ == "__main__":
    main()
