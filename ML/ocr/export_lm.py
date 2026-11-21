"""Latih dan ekspor model bahasa karakter + leksikon untuk decoder.

Keduanya adalah yang menyelesaikan ambiguitas bentuk yang mustahil:
``0`` vs ``O`` pada font tanpa garis miring, ``1`` vs ``I``, dan kata yang
terpotong satu huruf.

Keluaran:

- ``ocr/models/lm.json``      model bahasa karakter 5-gram
- ``ocr/models/lexicon.json`` leksikon kata + biaya edit dari konfusi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ocr.config as C
from ocr.corpus import boilerplate_lines, build_corpus
from ocr.lexicon import Lexicon, costs_from_confusion
from ocr.lm import CharLM
from ocr.templates import Templates


def collect_words(lines: list[str]) -> list[str]:
    """Ambil semua kata unik dari korpus, tanpa tanda baca pinggir."""
    words: set[str] = set()
    for line in lines:
        for token in line.split():
            cleaned = token.strip(".,:")
            if cleaned:
                words.add(cleaned)
    return sorted(words)


def load_confusion_costs(path: Path) -> dict[tuple[str, str], float]:
    """Biaya edit dari laporan konfusi classifier, bila tersedia."""
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    pairs = report.get("top_confusions_test", [])
    return costs_from_confusion([tuple(p) for p in pairs])


def main() -> None:
    ap = argparse.ArgumentParser(description="Latih LM + leksikon.")
    ap.add_argument("--n-lines", type=int, default=15000)
    ap.add_argument("--order", type=int, default=C.LM_ORDER)
    ap.add_argument(
        "--report",
        type=str,
        default=str(C.OUTPUT_DIR / "char_report.json"),
    )
    args = ap.parse_args()

    lines = build_corpus(args.n_lines, seed=C.RANDOM_SEED)
    print(f"korpus: {len(lines)} baris")

    lm = CharLM.train(lines, order=args.order)
    lm_path = lm.save(C.MODEL_DIR / "lm.json")
    print(
        f"LM: {lm_path} ({lm_path.stat().st_size / 1e6:.2f} MB, "
        f"{len(lm.ngrams)} konteks)"
    )

    costs = load_confusion_costs(Path(args.report))
    print(f"biaya konfusi: {len(costs) // 2} pasangan")

    lexicon = Lexicon.from_words(collect_words(lines), costs=costs)
    lex_path = lexicon.save(C.MODEL_DIR / "lexicon.json")
    print(
        f"leksikon: {lex_path} ({lex_path.stat().st_size / 1e6:.2f} MB, "
        f"{lexicon.size} kata)"
    )

    templates = Templates(boilerplate_lines())
    tpl_path = templates.save(C.MODEL_DIR / "templates.json")
    print(
        f"templat: {tpl_path} ({tpl_path.stat().st_size / 1e6:.2f} MB, "
        f"{len(templates)} baris)"
    )


if __name__ == "__main__":
    main()
