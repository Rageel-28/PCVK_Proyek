"""Koreksi kata hasil OCR memakai leksikon produk.

Prinsipnya sederhana namun sangat efektif: hasil OCR yang salah biasanya
masih berjarak edit kecil dari kata yang benar ("IND0M1E" -> "INDOMIE").
Dengan kamus 1.419 nama produk + kata umum struk, kesalahan karakter
bisa dipulihkan menjadi kata yang benar.

Levenshtein diimplementasikan manual agar bisa ditiru di Dart.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

import config as C


# Kata umum yang sering muncul di struk namun tidak ada di nama produk.
EXTRA_WORDS = {
    "TOTAL", "SUBTOTAL", "TUNAI", "KEMBALI", "KEMBALIAN", "DISKON", "PPN",
    "PAJAK", "ITEM", "QTY", "HARGA", "JUMLAH", "BAYAR", "CASH", "CHANGE",
    "GRAND", "TOTAL", "RP", "TGL", "KASIR", "STRUK", "NOTA", "TERIMA",
    "KASIH", "SELAMAT", "DATANG", "KEMBALI", "MINIMARKET", "INDOMARET",
    "ALFAMART", "ALFAMIDI", "INDOMARCO", "TRANSAKSI", "ANDA", "MEMBER",
}


def levenshtein(a: str, b: str, max_dist: int | None = None) -> int:
    """Jarak edit dengan pemotongan dini bila melebihi ``max_dist``."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    if max_dist is not None and abs(la - lb) > max_dist:
        return max_dist + 1

    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        row_min = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if cur[j] < row_min:
                row_min = cur[j]
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[lb]


def _clean(word: str) -> str:
    return re.sub(r"[^A-Z0-9.&/+-]", "", word.upper())


class Lexicon:
    """Kamus kata dengan pencarian cepat berbasis panjang kata."""

    def __init__(self, words: set[str] | None = None) -> None:
        self.words: set[str] = set()
        self._by_len: dict[int, list[str]] = {}
        if words:
            self.add_many(words)

    def add_many(self, words) -> None:
        for w in words:
            w = _clean(w)
            if len(w) >= 2:
                self.words.add(w)
        self._rebuild()

    def _rebuild(self) -> None:
        self._by_len.clear()
        for w in self.words:
            self._by_len.setdefault(len(w), []).append(w)
        for bucket in self._by_len.values():
            bucket.sort()

    @classmethod
    def from_dataset(cls) -> "Lexicon":
        lex = cls()
        words: set[str] = set(EXTRA_WORDS)

        if C.DATASET_CSV.exists():
            df = pd.read_csv(C.DATASET_CSV)
            for col in ("nama_produk", "receipt_text"):
                if col not in df.columns:
                    continue
                for value in df[col].dropna().astype(str):
                    for tok in re.split(r"\s+", value.upper()):
                        tok = _clean(tok)
                        if len(tok) >= 2 and not tok.isdigit():
                            words.add(tok)
        lex.add_many(words)
        return lex

    def candidates(self, word: str, max_dist: int) -> list[str]:
        """Kandidat dengan panjang berbeda maksimal ``max_dist``.

        Bila ``max_dist`` kecil (<= 1), hanya kata berhuruf awal sama yang
        dipertimbangkan. Untuk jarak lebih besar, semua kata pada panjang
        tersebut ikut diuji karena huruf pertama pun bisa salah baca.
        """
        if not word:
            return []
        out: list[str] = []
        first = word[0]
        for length in range(max(1, len(word) - max_dist), len(word) + max_dist + 1):
            bucket = self._by_len.get(length)
            if not bucket:
                continue
            if max_dist <= 1:
                out.extend(w for w in bucket if w[0] == first)
            else:
                out.extend(bucket)
        return out

    def correct(self, word: str, max_dist: int | None = None) -> tuple[str, int, bool]:
        """Kembalikan (kata terbaik, jarak, apakah dikoreksi)."""
        w = _clean(word)
        if len(w) < 2:
            return word, 0, False

        limit = max_dist if max_dist is not None else max(1, min(3, len(w) // 3))

        # Kata yang sudah ada di kamus tidak perlu diubah.
        if w in self.words:
            return w, 0, False

        best = w
        best_d = limit + 1
        for cand in self.candidates(w, limit):
            if cand == w:
                continue
            d = levenshtein(w, cand, limit)
            if d < best_d:
                best, best_d = cand, d
                if d == 1:
                    break

        if best_d <= limit:
            return best, best_d, True
        return word, 0, False

    def correct_text(self, text: str, max_dist: int | None = None) -> tuple[str, int]:
        """Koreksi per kata. Kembalikan (teks, jumlah kata terkoreksi)."""
        fixed = 0
        out: list[str] = []
        for tok in text.split():
            new, dist, changed = self.correct(tok, max_dist)
            if changed:
                fixed += 1
            out.append(new)
        return " ".join(out), fixed
