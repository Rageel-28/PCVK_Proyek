"""Referensi Python untuk pipeline OCR GABUNGAN (hybrid) — branch rijalmix.

Gabungan yang dimaksud:

    pra-proses + fitur + classifier  -> pipeline manual
        grayscale, batas 2600 px, upscale adaptif (baris -> 60 px),
        Otsu global, hapus garis tabel, fitur 172 dim, Random Forest
    segmentasi + rekonstruksi + leksikon -> pipeline klasik yang sudah ada
        proyeksi baris (Otsu 1D untuk foto), connected components + closing
        + split lembah, spasi dari median celah, Levenshtein ke kamus

Skrip ini menjalankan gabungan tersebut di Python lalu menulis teks acuan
ke `smart_receipt/test/fixtures/hybrid_lines.json` untuk pengujian paritas
port Dart (`lib/ocr/manual/hybrid_ocr_engine.dart`).

Pemakaian:
    python export_hybrid_reference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "ML" / "Percobaan OCR Manual" / "tools"
CLASSICAL = ROOT / "ML" / "ocr_classical"
FIXTURES = ROOT / "smart_receipt" / "test" / "fixtures"

sys.path.insert(0, str(CLASSICAL))
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import annotate_lines as A  # noqa: E402
import segment as SG  # noqa: E402
from lexicon import Lexicon  # noqa: E402
from export_manual_assets import Forest, feat_g12, MODEL_JSON, join_with_spaces  # noqa: E402

LEXICON_JSON = ROOT / "smart_receipt" / "assets" / "ocr_lexicon.json"
FILES = ("manual_line_0.png", "manual_line_1.png", "manual_line_2.png")


def hybrid_read(path: Path, forest: Forest, lexicon: Lexicon) -> list[dict]:
    image = A.load_image(path)          # grayscale + upscale adaptif (60 px)
    mask = A.prepare_mask(image)        # Otsu global + hapus garis tabel
    ink = (mask.astype(np.uint8) * 255)  # konvensi segmenter klasik

    lines: list[dict] = []
    for (top, bottom) in SG.segment_lines(ink):
        boxes = SG.segment_characters(ink, (top, bottom))
        if not boxes:
            continue

        chars: list[str] = []
        geo: list[tuple[int, int, int, int]] = []
        for (left, t, right, b) in boxes:
            sub = mask[t:b + 1, left:right + 1]
            if sub.size == 0 or not sub.any():
                continue
            chars.append(forest.predict(feat_g12(sub)))
            geo.append((t, left, b - t + 1, right - left + 1))
        if not chars:
            continue

        raw = join_with_spaces([(0, g[1], g[2], g[3]) for g in geo], chars)
        corrected, _fixes = lexicon.correct_text(raw)
        lines.append({"text": corrected, "chars": len(chars)})

    return lines


def main() -> None:
    forest = Forest(json.loads(MODEL_JSON.read_text(encoding="utf-8")))
    words = set(json.loads(LEXICON_JSON.read_text(encoding="utf-8"))["words"])
    lexicon = Lexicon(words)

    entries = []
    for name in FILES:
        result = hybrid_read(FIXTURES / name, forest, lexicon)
        entries.append({
            "file": name,
            "text": "\n".join(line["text"] for line in result),
            "lines": result,
        })
        print("%s: %d baris" % (name, len(result)))

    target = FIXTURES / "hybrid_lines.json"
    target.write_text(json.dumps({"images": entries}, indent=1), encoding="utf-8")
    print("ditulis:", target)


if __name__ == "__main__":
    main()
