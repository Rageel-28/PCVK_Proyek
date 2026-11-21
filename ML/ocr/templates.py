"""Pencocokan baris terhadap templat baris struk yang dikenal.

Sebagian baris struk adalah **teks tetap**: tagline toko, alamat, kalimat
terima kasih, peringatan barang tidak dapat ditukar. Teks itu dicetak kecil
dan tipis, sehingga justru paling sering rusak segmentasinya
(``SERBA ADA SERBA MUDAH`` terbaca ``S--XRBA ADA SERBA MUDAH``).

Classifier tidak bisa menolong di situ: yang rusak adalah pemotongan glyph,
bukan pengenalan bentuk. Yang bisa menolong adalah fakta bahwa kalimatnya
sudah diketahui. Modul ini mencocokkan baris hasil OCR ke daftar templat,
dengan dua penjaga supaya tidak merusak baris barang:

1. **Derajat kemiripan** — jarak edit harus <= ``max_ratio`` dari panjang
   templat. Baris yang benar-benar berbeda tidak akan menempel.
2. **Urutan angka harus sama** — kalau angka pada baris hasil OCR berbeda
   dengan templat, baris itu bukan templat tersebut. Penjaga ini melindungi
   baris harga, tanggal, dan nomor struk.
"""

from __future__ import annotations

import json
from pathlib import Path

from ocr.decode import _DIGIT_CONFUSION, _is_numeric_ish, _is_numeric_token
from ocr.decode import _DIGIT_CONFUSION, _is_numeric_ish, _is_numeric_token
from ocr.lexicon import weighted_levenshtein


def _number_signature(text: str) -> str:
    """Rangkaian angka pada baris, tanpa angka yang menempel di kata.

    Angka di dalam kata (``DIPONEGOR0``) bukan data, melainkan huruf yang
    salah baca. Yang perlu dibandingkan hanyalah token yang memang angka
    (``0005``, ``021-5777148``), termasuk yang hampir angka (``000S`` karena
    ``S`` mirip ``5``). Perbandingan inilah yang melindungi harga dan nomor
    struk tanpa menghalangi perbaikan alamat.
    """
    parts: list[str] = []
    for token in text.split():
        if _is_numeric_token(token) or _is_numeric_ish(token):
            parts.append(
                "".join(_DIGIT_CONFUSION.get(ch, ch) for ch in token)
            )
    return "".join(parts)


def snap_to_template(
    text: str,
    templates: list[str],
    max_ratio: float = 0.2,
    min_length: int = 8,
) -> str:
    """Ganti baris hasil OCR dengan templat terdekat bila cukup mirip.

    Args:
        text: baris hasil OCR.
        templates: daftar baris yang dikenal (biasanya dari korpus).
        max_ratio: jarak edit maksimum relatif terhadap panjang templat.
        min_length: baris lebih pendek dari ini tidak pernah dicocokkan.

    Returns:
        Templat terdekat bila lolos kedua penjaga, selain itu ``text`` asli.
    """
    if not isinstance(text, str):
        raise TypeError(f"text harus str, dapat {type(text).__name__}")

    stripped = text.strip()
    if len(stripped) < min_length or not templates:
        return text

    text_digits = _number_signature(stripped)
    best: tuple[float, str] | None = None

    for template in templates:
        template_digits = _number_signature(template)
        if template_digits:
            # Templat berangka: angka harus sama persis. Inilah penjaga yang
            # melindungi baris harga, tanggal, dan nomor struk.
            if template_digits != text_digits:
                continue
            ratio_limit = max_ratio
        else:
            # Templat tanpa angka (tagline, kalimat penutup). Tidak ada isi
            # angka yang bisa dirusak, jadi penjaga angka dilonggarkan —
            # penting karena huruf yang salah baca sering muncul sebagai
            # digit (``E`` -> ``1``).
            ratio_limit = max(max_ratio, 0.45)

        # Panjang sangat berbeda tidak mungkin templat yang sama; lewati
        # supaya pencarian tetap cepat.
        if abs(len(template) - len(stripped)) > max(2, int(len(template) * 0.4)):
            continue
        distance = weighted_levenshtein(stripped, template)
        ratio = distance / max(1, len(template))
        if ratio > ratio_limit:
            continue
        if best is None or ratio < best[0]:
            best = (ratio, template)

    return best[1] if best is not None else text


class Templates:
    """Daftar templat baris yang bisa disimpan dan dimuat."""

    def __init__(self, lines: list[str]):
        seen: dict[str, None] = {}
        for line in lines:
            cleaned = " ".join(line.split())
            if cleaned:
                seen.setdefault(cleaned, None)
        self.lines = list(seen)

    def __len__(self) -> int:
        return len(self.lines)

    def snap(self, text: str, max_ratio: float = 0.2) -> str:
        return snap_to_template(text, self.lines, max_ratio=max_ratio)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.lines, ensure_ascii=False), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "Templates":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
