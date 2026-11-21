"""Tes korpus baris struk sintetis.

Korpus adalah sumber label otomatis untuk classifier karakter, jadi ia
harus deterministik, hanya memakai karakter yang ada di CHARSET, dan
menutupi seluruh 42 kelas.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr.config as C
from ocr.corpus import build_corpus, classify_line, format_price, line_kinds


def test_corpus_is_deterministic_for_a_seed() -> None:
    assert build_corpus(200, seed=7) == build_corpus(200, seed=7)


def test_different_seeds_give_different_corpus() -> None:
    assert build_corpus(200, seed=1) != build_corpus(200, seed=2)


def test_corpus_respects_requested_length() -> None:
    assert len(build_corpus(137, seed=3)) == 137


def test_every_line_uses_only_charset_characters() -> None:
    allowed = set(C.CHARSET) | {" "}
    for line in build_corpus(400, seed=11):
        assert line.strip() == line
        assert line, "baris kosong tidak boleh masuk korpus"
        assert line.upper() == line, f"baris tidak kapital: {line!r}"
        unexpected = set(line) - allowed
        assert not unexpected, f"karakter di luar charset {unexpected} pada {line!r}"


def test_corpus_covers_all_42_classes() -> None:
    seen = {ch for line in build_corpus(3000, seed=5) for ch in line if ch != " "}
    missing = set(C.CHARSET) - seen
    assert not missing, f"kelas tidak pernah muncul: {sorted(missing)}"


def test_corpus_contains_each_line_kind() -> None:
    kinds = line_kinds(build_corpus(3000, seed=5))
    assert kinds["item"] > 0
    assert kinds["qty"] > 0
    assert kinds["summary"] > 0
    assert kinds["rule"] > 0


def test_format_price_uses_indonesian_thousands_separator() -> None:
    assert format_price(8700) == "8.700"
    assert format_price(86300) == "86.300"
    assert format_price(12500) == "12.500"
    assert format_price(500) == "500"
    assert format_price(1234567) == "1.234.567"


def test_item_lines_end_with_a_price() -> None:
    lines = build_corpus(2000, seed=13)
    items = [ln for ln in lines if classify_line(ln) == "item"]
    assert len(items) > 50, "harus ada cukup baris item"
    for line in items[:50]:
        assert line.split()[-1].replace(".", "").isdigit(), line


def test_classify_line_separates_the_four_kinds() -> None:
    assert classify_line("--------------------------------") == "rule"
    assert classify_line("3 X 19.800") == "qty"
    assert classify_line("TOTAL 86.300") == "summary"
    assert classify_line("POCARI SWEAT 500ML 8.700") == "item"
    assert classify_line("INDOMARET") == "text"
