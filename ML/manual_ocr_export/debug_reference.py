"""Cetak kotak glyph & karakter referensi untuk citra fixture (debug paritas).

Dipakai saat membandingkan port Dart dengan Python: hasilnya JSON
`ML/manual_ocr_export/reference_boxes.json` berisi, per baris, daftar
kotak glyph (top, left, height, width) dan karakter hasil Random Forest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "ML" / "Percobaan OCR Manual" / "tools"
FIXTURES = ROOT / "smart_receipt" / "test" / "fixtures"

sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import annotate_lines as A  # noqa: E402
from export_manual_assets import Forest, feat_g12, MODEL_JSON  # noqa: E402


def main() -> None:
    forest = Forest(json.loads(MODEL_JSON.read_text(encoding="utf-8")))
    out = {}
    for name in ("manual_line_0.png", "manual_line_1.png", "manual_line_2.png"):
        path = FIXTURES / name
        image = A.load_image(path)
        mask = A.prepare_mask(image)

        lines = []
        for ln in A.detect_lines(mask, path.name):
            band = mask[ln.top:ln.bottom + 1, ln.left:ln.right + 1]
            raw_boxes = A.segment_glyphs(band)
            if not raw_boxes:
                continue
            med_h = float(np.median([b[2] for b in raw_boxes]))
            min_h = max(3.0, med_h * 0.10)
            kept = [b for b in raw_boxes if b[2] >= min_h and b[3] >= 2]
            if not kept:
                continue

            chars = []
            boxes = []
            for (gy, gx, gh, gw) in kept:
                sub = mask[ln.top + gy:ln.top + gy + gh, ln.left + gx:ln.left + gx + gw]
                chars.append(forest.predict(feat_g12(sub)))
                boxes.append([int(ln.top + gy), int(ln.left + gx), int(gh), int(gw)])

            lines.append({
                "top": int(ln.top),
                "bottom": int(ln.bottom),
                "n_raw": len(raw_boxes),
                "boxes": boxes,
                "chars": "".join(chars),
            })
        out[name] = lines

    target = Path(__file__).resolve().parent / "reference_boxes.json"
    target.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("ditulis:", target)
    for name, lines in out.items():
        print(name, "baris:", len(lines))
        for line in lines:
            print("  y=%d-%d n_raw=%d n=%d %s"
                  % (line["top"], line["bottom"], line["n_raw"],
                     len(line["boxes"]), line["chars"]))


if __name__ == "__main__":
    main()
