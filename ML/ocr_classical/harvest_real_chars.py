"""Ekstrak patch karakter asli dari dataset foto struk tanpa label eksternal.

**DEPRECATED -- jangan dipakai untuk melatih model.**

Modul ini memakai *self-training*: label berasal dari prediksi model itu
sendiri. Ketika model sedang rusak (charset masih memuat huruf kecil,
kelas ``0`` tidak terwakili), modul ini memanen kesalahan model dan
mengunci kesalahan itu ke data latih. Terbukti: model hasil panen ini
membaca ``0`` sebagai ``8`` secara sistematis.

Penggantinya adalah ``build_real_chars.py``, yang mengambil label dari
transkripsi manual (``transcripts_real.json``) sehingga setiap label
benar. Berkas ``chars_real.npz`` yang dihasilkan modul ini sudah dibuang
dari ``data/`` dan tidak lagi dimuat ``train_char.py``.

Isi asli modul (untuk catatan sejarah):

  1. Jalankan ClassicalOCR pada foto struk asli (`ML/Dataset Citra` & `ML/OCR Struk Belanja.v2i.coco`).
  2. Panen patch 32x32 yang:
     - berkeyakinan tinggi (confidence >= 0.72, margin >= 0.8), ATAU
     - berasal dari kata yang terverifikasi leksikon (panjang sama, jarak edit <= 1).
  3. Simpan ke `data/chars_real.npz` untuk digabungkan saat melatih ulang Linear SVM.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config as C
import segment as SG
from lexicon import levenshtein
from ocr_pipeline import CharPrediction, ClassicalOCR
from render_synthetic import normalize_char


def _split_words(preds: list[CharPrediction]) -> list[list[CharPrediction]]:
    if not preds:
        return []
    if len(preds) == 1:
        return [preds]
    gaps = [b.bbox[0] - a.bbox[2] for a, b in zip(preds, preds[1:])]
    median_gap = float(np.median(gaps)) if gaps else 1.0
    threshold = max(2.0, median_gap * C.SPACE_GAP_RATIO)

    words: list[list[CharPrediction]] = [[preds[0]]]
    for prev, cur in zip(preds, preds[1:]):
        if (cur.bbox[0] - prev.bbox[2]) > threshold:
            words.append([cur])
        else:
            words[-1].append(cur)
    return words


def harvest(
    max_citra: int = 200,
    max_coco: int = 150,
    max_per_class: int = 600,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    ocr = ClassicalOCR(method="sauvola", use_deskew=True, use_lexicon=True)
    lex = ocr.lexicon

    citra_dir = C.ML_DIR / "Dataset Citra"
    coco_dir = C.COCO_DIR / "train"

    paths: list[Path] = []
    if citra_dir.exists():
        citra_files = sorted(citra_dir.glob("*.jpg"))
        if len(citra_files) > max_citra:
            idx = rng.choice(len(citra_files), size=max_citra, replace=False)
            citra_files = [citra_files[i] for i in sorted(idx)]
        paths.extend(citra_files)

    if coco_dir.exists():
        coco_files = sorted(coco_dir.glob("*.jpg"))
        if len(coco_files) > max_coco:
            idx = rng.choice(len(coco_files), size=max_coco, replace=False)
            coco_files = [coco_files[i] for i in sorted(idx)]
        paths.extend(coco_files)

    print(f"Memanen karakter dari {len(paths)} gambar struk asli...")

    buckets: dict[int, list[np.ndarray]] = defaultdict(list)

    for i, path in enumerate(paths):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        gray, ink, _angle = ocr.binarize(img)
        line_boxes = SG.segment_lines(ink)

        for box in line_boxes:
            lp = ocr.read_line(gray, ink, box)
            if not lp.chars:
                continue

            # 1. Panen karakter berkeyakinan tinggi.
            for ch in lp.chars:
                if ch.char in C.CHAR_TO_IDX and ch.confidence >= 0.72 and ch.margin >= 0.8:
                    cls_id = C.CHAR_TO_IDX[ch.char]
                    if len(buckets[cls_id]) < max_per_class:
                        buckets[cls_id].append(normalize_char(gray, ch.bbox))

            # 2. Panen karakter dari kata yang terverifikasi leksikon (panjang sama).
            if lex is None:
                continue
            for word_preds in _split_words(lp.chars):
                if len(word_preds) < 4:
                    continue
                raw_word = "".join(p.char for p in word_preds)
                if not raw_word.isalpha():
                    continue
                if raw_word in lex.words:
                    target = raw_word
                else:
                    # Cari kandidat leksikon dengan panjang yang sama dan jarak edit == 1.
                    same_len = [w for w in lex._by_len.get(len(raw_word), []) if w.isalpha()]
                    matches = [w for w in same_len if levenshtein(raw_word, w, 1) == 1]
                    if len(matches) != 1:
                        continue
                    # Pastikan mayoritas huruf lain cukup yakin.
                    if np.mean([p.confidence for p in word_preds]) < 0.38:
                        continue
                    target = matches[0]

                for p, true_ch in zip(word_preds, target):
                    if true_ch in C.CHAR_TO_IDX:
                        cls_id = C.CHAR_TO_IDX[true_ch]
                        if len(buckets[cls_id]) < max_per_class:
                            buckets[cls_id].append(normalize_char(gray, p.bbox))

        if (i + 1) % 50 == 0:
            total = sum(len(v) for v in buckets.values())
            print(f"  {i + 1}/{len(paths)} gambar -> {total} patch terkumpul ({len(buckets)} kelas)")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for cls_id, patches in sorted(buckets.items()):
        for p in patches:
            X_list.append(p)
            y_list.append(cls_id)

    Xa = np.stack(X_list).astype(np.uint8) if X_list else np.zeros((0, C.CHAR_SIZE, C.CHAR_SIZE), np.uint8)
    ya = np.array(y_list, dtype=np.int32)
    out_path = C.DATA_DIR / "chars_real.npz"
    np.savez_compressed(out_path, X=Xa, y=ya)

    print(f"Tersimpan: {out_path.name}  X={Xa.shape}  kelas={len(buckets)}")
    return {"path": str(out_path), "n": int(len(ya)), "classes": len(buckets)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-citra", type=int, default=200)
    ap.add_argument("--max-coco", type=int, default=150)
    ap.add_argument("--max-per-class", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    harvest(args.max_citra, args.max_coco, args.max_per_class, args.seed)


if __name__ == "__main__":
    main()
