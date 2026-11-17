"""Ekspor potongan baris struk **asli** + hasil Python untuk uji paritas Dart.

Fixture ini memakai piksel nyata dari `ML/Dataset Citra`, bukan citra
render. Tujuannya agar uji paritas Dart membuktikan port bekerja pada
domain sesungguhnya: cetakan thermal yang pucat, kertas kusut, dan
variasi kontras.

Setiap entri berisi:
  pixels      : isi grayscale crop baris (baris demi baris)
  truth       : transkripsi manual
  python_text : keluaran pipeline Python pada crop yang sama

Jalankan:
    python export_real_parity.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2

import config as C
import real_data as RD
from evaluate_real import load_transcripts
from ocr_pipeline import ClassicalOCR

OUT = (
    C.PROJECT_DIR / "smart_receipt" / "test" / "fixtures" / "real_line_parity.json"
)


def main() -> None:
    transcripts = load_transcripts()
    images = {im.stem: im for im in RD.load_real_images(stems=set(transcripts))}
    ocr = ClassicalOCR(method="sauvola", use_deskew=True, use_lexicon=False)

    samples: list[dict] = []
    for stem, entry in transcripts.items():
        im = images.get(stem)
        if im is None:
            continue
        truths = entry.get("items", [])
        if len(truths) != len(im.lines):
            continue
        gray = cv2.imread(str(im.path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        for index, (box, truth) in enumerate(zip(im.lines, truths)):
            crop = RD.crop_line(gray, box, pad=2)
            if crop.size == 0:
                continue

            # Jalankan pipeline penuh pada crop, sama seperti aplikasi
            # memanggil `run()`. Dengan begitu uji paritas membandingkan
            # dua implementasi pada masukan dan jalur yang identik.
            result = ocr.run(crop)

            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                continue
            png_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            samples.append(
                {
                    "stem": stem,
                    "index": index,
                    "kind": entry.get("kind", "?"),
                    "truth": truth,
                    "python_text": result.text.strip(),
                    "width": int(crop.shape[1]),
                    "height": int(crop.shape[0]),
                    "png_b64": png_b64,
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"samples": samples}, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Tersimpan: {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"Sampel: {len(samples)}")
    exact = sum(1 for s in samples if s["python_text"].strip() == s["truth"].strip())
    print(f"Cocok dengan transkripsi: {exact}/{len(samples)}")


if __name__ == "__main__":
    main()
