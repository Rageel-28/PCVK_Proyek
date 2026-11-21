"""Leksikon kata dengan Levenshtein berbobot biaya konfusi.

Leksikon menutup error sistematis yang tidak bisa ditutup classifier:
``TOTAI`` → ``TOTAL``, ``CILIIVCIIVG`` → ``CILINCING``. Biaya edit diambil
dari matriks konfusi sehingga mengganti ``0``↔``O`` jauh lebih murah
daripada mengganti ``A``↔``Z``.
"""

from __future__ import annotations

import json
from pathlib import Path

import ocr.config as C


def weighted_levenshtein(
    a: str, b: str, costs: dict[tuple[str, str], float] | None = None
) -> float:
    """Jarak edit dengan biaya substitusi yang bisa dikustomisasi.

    Biaya bawaan: sisip/hapus = 1.0, substitusi = 1.0, kecuali ada di
    ``costs`` (mis. ``("0","O") -> 0.2``).
    """
    if costs is None:
        costs = {}
    if a == b:
        return 0.0
    if not a:
        return float(len(b))
    if not b:
        return float(len(a))

    previous = [float(j) for j in range(len(b) + 1)]
    for i, ca in enumerate(a, start=1):
        current = [float(i)] + [0.0] * len(b)
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                cost = 0.0
            else:
                cost = costs.get((ca, cb), costs.get((cb, ca), 1.0))
            current[j] = min(
                previous[j] + 1.0,          # hapus
                current[j - 1] + 1.0,       # sisip
                previous[j - 1] + cost,     # substitusi
            )
        previous = current
    return previous[-1]


class Lexicon:
    """Kumpulan kata yang dikenal, dengan pencarian tetangga terdekat."""

    def __init__(
        self, words: list[str], costs: dict[tuple[str, str], float] | None = None
    ):
        self.words = sorted({w for w in words if w})
        self._set = set(self.words)
        self.costs = costs or {}
        # Indeks panjang untuk memangkas pencarian.
        self._by_length: dict[int, list[str]] = {}
        for word in self.words:
            self._by_length.setdefault(len(word), []).append(word)

    @classmethod
    def from_words(
        cls, words: list[str], costs: dict[tuple[str, str], float] | None = None
    ) -> "Lexicon":
        return cls(words, costs)

    @property
    def size(self) -> int:
        return len(self.words)

    def contains(self, word: str) -> bool:
        return word in self._set

    def closest(
        self,
        word: str,
        max_distance: float = 2.0,
        predicate=None,
    ) -> tuple[str | None, float]:
        """Kata terdekat dalam jarak ``max_distance``.

        Kandidat dibatasi pada kata yang panjangnya berselisih paling banyak
        ``ceil(max_distance)`` — tanpa ini pencarian jadi lambat pada
        leksikon besar.

        Args:
            word: kata yang dicari.
            max_distance: jarak edit maksimum.
            predicate: penyaring tambahan, mis. ``lambda w: jumlah_digit(w)
                == 5`` untuk token harga.
        """
        if not word:
            return None, float("inf")

        best_word: str | None = None
        best_distance = float("inf")
        span = int(max_distance) + 1
        for length in range(len(word) - span, len(word) + span + 1):
            for candidate in self._by_length.get(length, ()):
                if predicate is not None and not predicate(candidate):
                    continue
                distance = weighted_levenshtein(word, candidate, self.costs)
                if distance < best_distance:
                    best_distance = distance
                    best_word = candidate
                    if distance == 0.0:
                        return best_word, 0.0
        if best_distance > max_distance:
            return None, best_distance
        return best_word, best_distance

    def save(self, path: str | Path) -> Path:
        payload = {
            "version": 1,
            "words": self.words,
            "costs": [[a, b, c] for (a, b), c in self.costs.items()],
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        costs = {(a, b): float(c) for a, b, c in payload.get("costs", [])}
        return cls(payload["words"], costs=costs)


def costs_from_confusion(confusions: list[tuple[int, str, str]]) -> dict[tuple[str, str], float]:
    """Ubah daftar konfusi ``(jumlah, asli, tebakan)`` jadi biaya edit.

    Biaya = 1 / (1 + jumlah), dibatasi minimal 0.1. Pasangan yang sering
    tertukar jadi murah; pasangan yang tidak pernah tertukar tetap 1.0.
    """
    costs: dict[tuple[str, str], float] = {}
    for count, a, b in confusions:
        if count <= 0:
            continue
        cost = max(0.1, 1.0 / (1.0 + count))
        costs[(a, b)] = cost
        costs[(b, a)] = cost
    return costs


def build_from_corpus(n_lines: int = 15000, seed: int = C.RANDOM_SEED) -> Lexicon:
    """Bangun leksikon dari kata-kata di korpus struk."""
    from ocr.corpus import build_corpus

    words: set[str] = set()
    for line in build_corpus(n_lines, seed=seed):
        for token in line.split():
            cleaned = token.strip(".")
            if cleaned:
                words.add(cleaned)
    return Lexicon.from_words(sorted(words))
