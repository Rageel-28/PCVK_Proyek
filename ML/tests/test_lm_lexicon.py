"""Tes model bahasa karakter (n-gram) dan leksikon.

Ini bagian yang menyelesaikan ambiguitas yang mustahil diselesaikan bentuk
huruf: ``0`` vs ``O`` pada font tanpa garis miring, ``1`` vs ``I``, dan
kata yang terpotong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C
from ocr.lexicon import Lexicon, weighted_levenshtein
from ocr.lm import CharLM


# ---------------------------------------------------------------------------
# Model bahasa
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lm() -> CharLM:
    corpus = [
        "TOTAL 47.200",
        "TOTAL 49.700",
        "TOTAL 86.300",
        "TUNAI 72.900",
        "KEMBALI 5.000",
        "DISKON 2.500",
        "PPN 11% 4.677",
        "DPP 42.523",
        "INDOMARET",
        "SERBA ADA SERBA MUDAH",
    ]
    return CharLM.train(corpus, order=4)


def test_lm_logprob_is_finite(lm: CharLM) -> None:
    assert np.isfinite(lm.logprob("TOTAL 47.200"))


def test_lm_prefers_seen_text_over_random(lm: CharLM) -> None:
    seen = lm.logprob("TOTAL 47.200")
    unseen = lm.logprob("QXZJ 99.999")
    assert seen > unseen


def test_lm_context_changes_score(lm: CharLM) -> None:
    # 'TOTAL ' harus lebih mungkin diikuti angka daripada 'INDOMARET '.
    after_total = lm.next_char_logprobs("TOTAL ", "4")
    after_store = lm.next_char_logprobs("INDOMARET ", "4")
    assert after_total > after_store


def test_lm_handles_unknown_context(lm: CharLM) -> None:
    assert np.isfinite(lm.logprob("ZZZZZZ"))


def test_lm_is_deterministic(lm: CharLM) -> None:
    assert lm.logprob("TOTAL 47.200") == lm.logprob("TOTAL 47.200")


def test_lm_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError):
        CharLM.train([], order=3)


def test_lm_exports_and_reloads(lm: CharLM, tmp_path: Path) -> None:
    path = lm.save(tmp_path / "lm.json")
    loaded = CharLM.load(path)
    assert loaded.logprob("TOTAL 47.200") == pytest.approx(
        lm.logprob("TOTAL 47.200")
    )


# ---------------------------------------------------------------------------
# Leksikon
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lex() -> Lexicon:
    return Lexicon.from_words(
        ["TOTAL", "TUNAI", "KEMBALI", "DISKON", "INDOMARET", "POCARI", "SWEAT"]
    )


def test_lexicon_contains_words(lex: Lexicon) -> None:
    assert lex.contains("TOTAL")
    assert not lex.contains("TOTALL")


def test_lexicon_finds_exact_match(lex: Lexicon) -> None:
    assert lex.closest("TOTAL")[0] == "TOTAL"


def test_lexicon_corrects_one_edit(lex: Lexicon) -> None:
    word, distance = lex.closest("TOTAI")
    assert word == "TOTAL"
    assert distance == 1


def test_lexicon_respects_max_distance(lex: Lexicon) -> None:
    word, distance = lex.closest("ZZZZZZZZ", max_distance=2)
    assert word is None


def test_lexicon_returns_none_for_garbage(lex: Lexicon) -> None:
    assert lex.closest("QQQQQ", max_distance=1)[0] is None


def test_weighted_levenshtein_is_symmetric_for_equal_costs() -> None:
    assert weighted_levenshtein("ABC", "ABD") == 1
    assert weighted_levenshtein("ABD", "ABC") == 1


def test_weighted_levenshtein_prefers_cheap_substitution() -> None:
    # Biaya 0->O dibuat murah: 'O' seharusnya lebih dekat ke '0'.
    costs = {("0", "O"): 0.2}
    cheap = weighted_levenshtein("1O", "10", costs)
    expensive = weighted_levenshtein("1X", "10", costs)
    assert cheap < expensive


def test_lexicon_uses_confusion_costs(lex: Lexicon) -> None:
    costs = {("0", "O"): 0.1, ("O", "0"): 0.1}
    lex2 = Lexicon.from_words(["1000", "1000"], costs=costs)
    assert lex2.closest("1OOO")[0] == "1000"


def test_lexicon_serialises(lex: Lexicon, tmp_path: Path) -> None:
    path = lex.save(tmp_path / "lex.json")
    loaded = Lexicon.load(path)
    assert loaded.contains("TOTAL")
    assert loaded.size == lex.size


def test_lexicon_size_counts_words(lex: Lexicon) -> None:
    assert lex.size == 7
