"""Ekspor leksikon koreksi OCR ke JSON untuk aplikasi Flutter.

Leksikon dibangun dari `Dataset_SmartReceipt_Sintetis.csv` plus daftar kata
umum struk di `lexicon.py`. Aset di `smart_receipt/assets/ocr_lexicon.json`
harus dihasilkan ulang setiap kali dataset atau aturan pembersihan kata
berubah; kalau tidak, aplikasi memakai kamus lama dan mengoreksi kata yang
seharusnya dibiarkan (mis. `GAM` menjadi `GA`).

Jalankan:
    python export_lexicon.py
"""

from __future__ import annotations

import json
from pathlib import Path

import config as C
from lexicon import Lexicon

OUT_MODEL = C.MODEL_DIR / "ocr_lexicon.json"
OUT_ASSET = C.ASSET_DIR / "ocr_lexicon.json"


def main() -> None:
    lex = Lexicon.from_dataset()
    words = sorted(lex.words)
    payload = {"words": words}

    text = json.dumps(payload, separators=(",", ":"))
    for path in (OUT_MODEL, OUT_ASSET):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Tersimpan: {path}  ({path.stat().st_size / 1024:.0f} KB)")

    print(f"Jumlah kata: {len(words)}")
    shortest = sorted(words, key=len)[:5]
    print(f"Terpendek: {shortest}")


if __name__ == "__main__":
    main()
