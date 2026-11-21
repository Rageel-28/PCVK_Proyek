"""Model bahasa karakter (n-gram) dengan interpolasi Kneser-Ney sederhana.

Kenapa perlu model bahasa sama sekali? Karena sebagian keputusan tidak bisa
diambil dari bentuk huruf:

- Font tanpa garis miring membuat ``0`` dan ``O`` identik secara piksel.
- ``1`` dan ``I`` sering identik pada font sans.
- ``rn`` dan ``m`` bisa sama.

Konteks kalimat menyelesaikannya: setelah ``TOTAL `` yang wajar adalah
digit, bukan huruf. Model ini memberi skor log-probabilitas karakter
berikutnya berdasarkan 1-4 karakter sebelumnya.

Implementasi memakai interpolasi linier berbobot (stupid-backoff yang
dihaluskan), bukan Kneser-Ney penuh. Alasannya: hasilnya hampir sama untuk
order rendah, jauh lebih sederhana, dan mudah diport ke Dart.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import ocr.config as C

BOS = "\x02"     # awal baris
EOS = "\x03"     # akhir baris


class CharLM:
    """Model bahasa karakter dengan backoff interpolasi.

    Atribut publik yang diekspor ke JSON: ``order``, ``vocab``,
    ``unigram``, ``ngrams`` (dict dari n-gram ke log-prob terinterpolasi).
    """

    def __init__(
        self,
        order: int,
        vocab: list[str],
        ngrams: dict[str, dict[str, float]],
        unigram: dict[str, float],
        floor: float,
    ):
        self.order = order
        self.vocab = vocab
        self.ngrams = ngrams
        self.unigram = unigram
        self.floor = floor
        self._vocab_set = set(vocab)

    # ------------------------------------------------------------------
    # Pelatihan
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        lines: list[str],
        order: int = 5,
        min_count: int = 1,
    ) -> "CharLM":
        """Latih dari daftar baris teks."""
        corpus = [ln for ln in lines if ln]
        if not corpus:
            raise ValueError("korpus kosong")

        counts: list[Counter] = [Counter() for _ in range(order + 1)]
        for line in corpus:
            padded = BOS * (order - 1) + line + EOS
            for n in range(1, order + 1):
                for i in range(n - 1, len(padded)):
                    counts[n][padded[i - n + 1 : i + 1]] += 1

        vocab = sorted({ch for line in corpus for ch in line} | {" "})
        vocab_with_bounds = vocab + [BOS, EOS]

        # Probabilitas unigram dengan penambahan kecil (Laplace halus).
        total = sum(counts[1].values())
        alpha = 0.5
        denom = total + alpha * len(vocab_with_bounds)
        unigram = {
            ch: (counts[1][ch] + alpha) / denom for ch in vocab_with_bounds
        }
        floor = alpha / denom

        # Log-prob per tingkat, lalu interpolasi dari tingkat tertinggi.
        log_prob: list[dict[str, float]] = [dict() for _ in range(order + 1)]
        for ch in vocab_with_bounds:
            log_prob[1][ch] = math.log(unigram[ch])

        for n in range(2, order + 1):
            context_totals = Counter()
            for gram, count in counts[n].items():
                context_totals[gram[:-1]] += count
            for gram, count in counts[n].items():
                if count < min_count:
                    continue
                context = gram[:-1]
                p = count / context_totals[context]
                # Interpolasi dengan tingkat di bawahnya.
                lower = log_prob[n - 1].get(gram[1:], math.log(floor))
                p_lower = math.exp(lower)
                weight = 0.85
                mixed = weight * p + (1 - weight) * p_lower
                log_prob[n][gram] = math.log(max(mixed, 1e-12))

        # Kelompokkan per konteks untuk pencarian cepat.
        ngrams: dict[str, dict[str, float]] = defaultdict(dict)
        for n in range(2, order + 1):
            for gram, value in log_prob[n].items():
                ngrams[gram[:-1]][gram[-1]] = value

        return cls(
            order=order,
            vocab=vocab,
            ngrams={k: dict(v) for k, v in ngrams.items()},
            unigram=log_prob[1],
            floor=math.log(floor),
        )

    # ------------------------------------------------------------------
    # Skoring
    # ------------------------------------------------------------------
    def next_char_logprobs(self, context: str, char: str) -> float:
        """Log-probabilitas ``char`` setelah ``context`` (hanya konteks akhir)."""
        for n in range(self.order, 1, -1):
            key = context[-(n - 1) :].rjust(n - 1, BOS) if len(context) < n - 1 else context[-(n - 1) :]
            table = self.ngrams.get(key)
            if table is not None and char in table:
                return table[char]
        return self.unigram.get(char, self.floor)

    def logprob(self, text: str) -> float:
        """Total log-probabilitas teks."""
        context = BOS * (self.order - 1)
        total = 0.0
        for ch in text + EOS:
            total += self.next_char_logprobs(context, ch)
            context = (context + ch)[-(self.order - 1) :]
        return total

    # ------------------------------------------------------------------
    # Serialisasi
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        payload = {
            "version": 1,
            "order": self.order,
            "vocab": self.vocab,
            "floor": self.floor,
            "unigram": self.unigram,
            "ngrams": self.ngrams,
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "CharLM":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            order=payload["order"],
            vocab=payload["vocab"],
            ngrams=payload["ngrams"],
            unigram=payload["unigram"],
            floor=payload["floor"],
        )


def build_from_corpus(n_lines: int = 15000, seed: int = C.RANDOM_SEED) -> CharLM:
    from ocr.corpus import build_corpus

    return CharLM.train(build_corpus(n_lines, seed=seed), order=C.LM_ORDER)
