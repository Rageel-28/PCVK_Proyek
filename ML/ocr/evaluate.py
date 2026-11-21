"""Evaluasi pipeline OCR pada himpunan fixture terkunci.

Metrik yang dilaporkan:

- **CER** — character error rate setelah semua baris digabung.
- **exact-line** — persentase baris ground truth yang muncul persis.
- **item-accuracy** — persentase baris barang (nama + harga) yang benar.
  Ini metrik yang paling penting untuk pemakaian nyata: struk dibaca untuk
  mengambil daftar belanjaan, bukan untuk transkripsi puitis.

Himpunan ``receipts_v2`` adalah **test terkunci**: tidak boleh dipakai
melatih atau menyetel apa pun. Skrip ini hanya membaca.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import ocr.config as C
from ocr.lexicon import Lexicon
from ocr.lm import CharLM
from ocr.model import load_model
from ocr.pipeline import recognize_page
from ocr.templates import Templates

PRICE_PATTERN = r"\d{1,3}(?:\.\d{3})+"


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            )
        previous = current
    return previous[-1]


def cer(hypothesis: str, reference: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(hypothesis, reference) / len(reference)


def wer(hypothesis: str, reference: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein(" ".join(hyp_words), " ".join(ref_words)) / len(ref_words)


@dataclass
class ItemScore:
    """Ringkasan kecocokan baris barang."""

    total: int = 0
    name_ok: int = 0
    price_ok: int = 0
    full_ok: int = 0


def evaluate(
    fixtures_dir: Path,
    model_path: Path | None = None,
    verbose: bool = False,
) -> dict:
    model = load_model(model_path)
    lm = CharLM.load(C.MODEL_DIR / "lm.json")
    lexicon = Lexicon.load(C.MODEL_DIR / "lexicon.json")
    templates = Templates.load(C.MODEL_DIR / "templates.json")

    truth = json.loads((fixtures_dir / "ground_truth.json").read_text("utf-8"))

    edit_distance = 0
    reference_chars = 0
    word_errors = 0
    reference_words = 0
    exact_lines = 0
    total_lines = 0
    extra_lines = 0
    items = ItemScore()

    for entry in truth:
        image_path = fixtures_dir / entry["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)

        result = recognize_page(image, model, lm, lexicon, templates)
        got = [line.text for line in result.lines]
        want = entry["lines"]

        # Cocokkan satu-satu: setiap baris ground truth mengambil baris
        # terbaca yang paling mirip, dan baris terbaca tidak boleh dipakai
        # dua kali. Tanpa ini, CER ikut menghukum baris header/footer yang
        # memang tidak ada di ground truth.
        pairs = _match_lines(want, got)
        matched_hypotheses = {id(h) for _, h in pairs}
        extra_lines += sum(1 for h in got if id(h) not in matched_hypotheses)

        for reference, hypothesis in pairs:
            total_lines += 1
            # Spasi ganda pada ground truth hanya menandai jarak kolom, bukan
            # karakter; bandingkan setelah spasi dirapikan.
            if hypothesis is not None and _norm_space(hypothesis) == _norm_space(
                reference
            ):
                exact_lines += 1

            # Jarak dihitung pada teks yang spasinya sudah dirapikan, sama
            # seperti perbandingan exact-line. Spasi ganda pada ground truth
            # hanya menandai jarak kolom, bukan karakter.
            clean_reference = _norm_space(reference)
            if hypothesis is None:
                edit_distance += len(clean_reference)
                word_errors += len(clean_reference.split())
            else:
                clean_hypothesis = _norm_space(hypothesis)
                edit_distance += levenshtein(clean_hypothesis, clean_reference)
                word_errors += levenshtein(
                    " ".join(clean_hypothesis.split()),
                    " ".join(clean_reference.split()),
                )
            reference_chars += len(reference)
            reference_words += len(reference.split())

        for line in want:
            if not _has_price(line):
                continue
            items.total += 1
            match = _best_match(line, got)
            if match is None:
                continue
            if _norm_space(_item_name(match)) == _norm_space(_item_name(line)):
                items.name_ok += 1
            if _item_price(match) == _item_price(line):
                items.price_ok += 1
            if _norm_space(match) == _norm_space(line):
                items.full_ok += 1

        if verbose:
            print(f"\n=== {entry['image']}")
            for reference, hypothesis in pairs:
                same = hypothesis is not None and _norm_space(
                    hypothesis
                ) == _norm_space(reference)
                status = "OK " if same else "BAD"
                print(f"  [{status}] {reference}")
                if not same:
                    print(f"        -> {hypothesis}")
            print("  --- baris ekstra ---")
            for line in got:
                if id(line) not in matched_hypotheses:
                    print(f"        {line}")

    return {
        "cer": edit_distance / reference_chars if reference_chars else 1.0,
        "wer": word_errors / reference_words if reference_words else 1.0,
        "exact_line": exact_lines / total_lines if total_lines else 0.0,
        "exact_lines": exact_lines,
        "total_lines": total_lines,
        "extra_lines": extra_lines,
        "item_total": items.total,
        "item_name_accuracy": items.name_ok / items.total if items.total else 0.0,
        "item_price_accuracy": items.price_ok / items.total if items.total else 0.0,
        "item_full_accuracy": items.full_ok / items.total if items.total else 0.0,
        "n_images": len(truth),
    }


def _match_lines(
    want: list[str], got: list[str]
) -> list[tuple[str, str | None]]:
    """Pasangkan baris ground truth dengan baris terbaca terbaik.

    Setiap baris terbaca hanya boleh dipakai sekali. Pasangan dengan jarak
    edit terkecil dipilih lebih dulu (greedy), sisanya dianggap tidak
    terbaca (``None``).
    """
    candidates: list[tuple[float, int, int]] = []
    for i, reference in enumerate(want):
        for j, hypothesis in enumerate(got):
            distance = levenshtein(hypothesis, reference) / max(1, len(reference))
            candidates.append((distance, i, j))
    candidates.sort()

    used_want: set[int] = set()
    used_got: set[int] = set()
    assignment: dict[int, str] = {}
    for distance, i, j in candidates:
        if i in used_want or j in used_got:
            continue
        if distance > 0.5:
            break
        used_want.add(i)
        used_got.add(j)
        assignment[i] = got[j]

    return [(reference, assignment.get(i)) for i, reference in enumerate(want)]


def _norm_space(text: str) -> str:
    return " ".join(text.split())


def _has_price(line: str) -> bool:
    import re

    return bool(re.search(PRICE_PATTERN, line))


def _item_name(line: str) -> str:
    import re

    return re.split(PRICE_PATTERN, line)[0].strip()


def _item_price(line: str) -> str:
    import re

    found = re.findall(PRICE_PATTERN, line)
    return found[-1] if found else ""


def _best_match(reference: str, candidates: list[str]) -> str | None:
    """Baris terbaca yang paling mirip dengan baris ground truth."""
    if not candidates:
        return None
    scored = sorted(
        candidates,
        key=lambda c: levenshtein(c, reference) / max(1, len(reference)),
    )
    best = scored[0]
    if levenshtein(best, reference) / max(1, len(reference)) > 0.6:
        return None
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluasi OCR pada fixture terkunci.")
    ap.add_argument(
        "--fixtures",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2]
            / "smart_receipt"
            / "test"
            / "fixtures"
            / "receipts_v2"
        ),
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    report = evaluate(Path(args.fixtures), verbose=args.verbose)
    print("\n" + "=" * 60)
    print(f"gambar        : {report['n_images']}")
    print(f"CER           : {report['cer']:.2%}")
    print(f"WER           : {report['wer']:.2%}")
    print(
        f"exact-line    : {report['exact_line']:.1%} "
        f"({report['exact_lines']}/{report['total_lines']})"
    )
    print(f"baris barang  : {report['item_total']}")
    print(f"  nama benar  : {report['item_name_accuracy']:.1%}")
    print(f"  harga benar : {report['item_price_accuracy']:.1%}")
    print(f"  utuh benar  : {report['item_full_accuracy']:.1%}")

    out = C.OUTPUT_DIR / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nlaporan: {out}")


if __name__ == "__main__":
    main()
