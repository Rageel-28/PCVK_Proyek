"""Bangun data latih karakter dari struk **asli** yang sudah ditranskripsi.

Berbeda dengan `harvest_real_chars.py` yang lama, modul ini tidak memakai
self-training. Label diambil dari transkripsi manual
(`transcripts_real.json`), sehingga model tidak mungkin mengunci
kesalahannya sendiri.

Sumber lokalisasi baris adalah anotasi COCO (`product_item`), bukan
segmentasi otomatis. Alasannya: pada foto struk kusut, segmentasi baris
otomatis masih menghasilkan blok yang menyatu (logo, QR, lipatan), jadi
tidak bisa dipakai untuk melabeli.

Aturan penerimaan sampel sengaja ketat:

1. Hanya baris yang punya bbox COCO **dan** transkripsi yang dipakai.
2. Segmentasi karakter pada crop baris harus menghasilkan jumlah
   karakter yang sama dengan jumlah karakter transkripsi (spasi
   diabaikan). Kalau tidak, baris itu dilewati.
3. Hanya karakter di `CHARSET` yang disimpan.

Konsekuensinya data yang dihasilkan sedikit, tetapi setiap label benar.
Data sintetis tetap dipakai sebagai pelengkap untuk kelas yang tidak
terwakili (lihat `train_char.py`).

Jalankan:
    python build_real_chars.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import cv2
import numpy as np

import config as C
import real_data as RD
import segment as SG
from ocr_pipeline import ClassicalOCR
from render_synthetic import normalize_char

TRANSCRIPTS = C.ROOT / "transcripts_real.json"


def _truth_chars(line: str) -> str:
    """Buang spasi dan petakan ke bentuk kanonik di dalam charset."""
    out: list[str] = []
    for ch in line:
        if ch == " ":
            continue
        canon = C.canonical_char(ch)
        if canon in C.CHAR_TO_IDX:
            out.append(canon)
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="chars_real_gold")
    args = ap.parse_args()

    data = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    images = {im.stem: im for im in RD.load_real_images(require_lines=False, stems=set(data))}
    local_idx = RD._local_index()

    ocr = ClassicalOCR(method="sauvola", use_deskew=True, use_lexicon=False)

    # Tahap 1: panen hanya dari crop `items` COCO (lokasi dijamin oleh anotator).
    stage1_X: list[np.ndarray] = []
    stage1_y: list[int] = []
    used_items = 0
    skipped = 0
    doc_cache: list[tuple[str, np.ndarray, dict]] = []

    for stem, entry in data.items():
        im = images.get(stem)
        if im is not None and im.split == "test":
            continue
        img_path = im.path if im is not None else local_idx.get(stem)
        if img_path is None:
            print(f"  ! tidak ada di indeks: {stem}")
            continue

        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        doc_cache.append((stem, gray, entry))

        truths = entry.get("items", [])
        if im is not None and truths and len(truths) == len(im.lines):
            for box, truth in zip(im.lines, truths):
                target = _truth_chars(truth)
                if not target:
                    continue

                crop = RD.crop_line(gray, box, pad=2)
                if crop.size == 0:
                    continue

                g, ink, _angle = ocr.binarize(crop)
                line_boxes = SG.segment_lines(ink)
                if not line_boxes:
                    skipped += 1
                    continue

                char_boxes = SG.segment_characters(ink, line_boxes[0])
                if len(char_boxes) != len(target):
                    skipped += 1
                    continue

                for cb, ch in zip(char_boxes, target):
                    patch = normalize_char(g, cb)
                    if patch.shape != (C.CHAR_SIZE, C.CHAR_SIZE):
                        continue
                    stage1_X.append(patch)
                    stage1_y.append(C.CHAR_TO_IDX[ch])
                used_items += 1

    # Latih verifier cepat khusus tahap 1 agar tahap 2 tidak pernah salah
    # memasangkan baris dokumen `full` yang kebetulan panjangnya sama.
    import features as F
    from sklearn.svm import LinearSVC

    X: list[np.ndarray] = list(stage1_X)
    y: list[int] = list(stage1_y)
    used_full = 0

    if stage1_X:
        clf1 = LinearSVC(C=0.5, max_iter=10000, dual=True, random_state=42)
        clf1.fit(F.hog_batch(np.stack(stage1_X)), np.array(stage1_y, dtype=np.int32))
        s1_counts = Counter(C.IDX_TO_CHAR[int(v)] for v in stage1_y)
        known_upper = {
            ch for ch, cnt in s1_counts.items()
            if cnt >= 20 and (ch.isdigit() or ch.isupper() or ch in ".,-:/")
        }

        doc_bin_cache: list[tuple[str, np.ndarray, list[tuple[int, int]], list[list[tuple[int, int, int, int]]], list[str], set[int]]] = []
        for stem, gray, entry in doc_cache:
            full_lines = entry.get("full", [])
            if not full_lines:
                continue
            g_doc, ink_doc, _ = ocr.binarize(gray)
            doc_boxes = SG.segment_lines(ink_doc)
            doc_chars_list = [SG.segment_characters(ink_doc, lb) for lb in doc_boxes]
            used_di: set[int] = set()
            doc_bin_cache.append((stem, g_doc, doc_boxes, doc_chars_list, full_lines, used_di))

            for fi, fl in enumerate(full_lines):
                t = _truth_chars(fl)
                if len(t) < 4:
                    continue
                cands: list[tuple[float, float, int, float, int, list[str]]] = []
                for di, cbs in enumerate(doc_chars_list):
                    if di in used_di or len(cbs) != len(t):
                        continue
                    patches = np.stack([normalize_char(g_doc, cb) for cb in cbs])
                    preds = [C.IDX_TO_CHAR[int(c)].upper() for c in clf1.predict(F.hog_batch(patches))]
                    score = sum(1 for p, ch in zip(preds, t) if p == ch.upper()) / len(t)
                    k_pairs = [(idx, p, ch) for idx, (p, ch) in enumerate(zip(preds, t)) if ch in known_upper]
                    k_sc = (
                        sum(1 for _, p, ch in k_pairs if p == ch.upper()) / len(k_pairs)
                        if k_pairs else score
                    )
                    rel_err = abs(fi / max(len(full_lines) - 1, 1) - di / max(len(doc_boxes) - 1, 1))
                    cands.append((score, k_sc, len(k_pairs), rel_err, di, preds))
                if not cands:
                    continue
                cands.sort(reverse=True)
                best_sc, k_sc, kn, rel_err, best_di, best_preds = cands[0]
                min_req = 0.70 if len(t) < 6 else 0.50
                accepted = best_sc >= min_req
                if accepted and len(cands) > 1 and cands[1][0] >= best_sc - 0.05:
                    rel_2 = cands[1][3]
                    if rel_err > 0.25 or rel_2 - rel_err < 0.10:
                        continue
                if not accepted:
                    same_len_near = [c for c in cands if c[3] <= 0.25]
                    if (
                        any(ch.islower() for ch in t)
                        and len(t) >= 8
                        and len(same_len_near) == 1
                        and rel_err <= 0.16
                        and kn >= 4
                        and k_sc >= 0.65
                    ):
                        mid = len(t) // 2
                        lk = [(p, ch) for idx, (p, ch) in enumerate(zip(best_preds, t)) if idx < mid and ch in known_upper]
                        rk = [(p, ch) for idx, (p, ch) in enumerate(zip(best_preds, t)) if idx >= mid and ch in known_upper]
                        lk_sc = sum(1 for p, ch in lk if p == ch.upper()) / len(lk) if lk else 0.0
                        rk_sc = sum(1 for p, ch in rk if p == ch.upper()) / len(rk) if rk else 0.0
                        shift_l = sum(1 for p, ch in zip(best_preds[1:], t[:-1]) if p == ch.upper()) / max(len(t) - 1, 1)
                        shift_r = sum(1 for p, ch in zip(best_preds[:-1], t[1:]) if p == ch.upper()) / max(len(t) - 1, 1)
                        if lk and rk and lk_sc >= 0.50 and rk_sc >= 0.65 and best_sc >= shift_l and best_sc >= shift_r:
                            accepted = True
                            print(f"  + stage2 mixed: [{stem[-12:]}] {fl!r}")
                if not accepted:
                    continue

                used_di.add(best_di)
                for cb, ch in zip(doc_chars_list[best_di], t):
                    patch = normalize_char(g_doc, cb)
                    if patch.shape != (C.CHAR_SIZE, C.CHAR_SIZE):
                        continue
                    X.append(patch)
                    y.append(C.CHAR_TO_IDX[ch])
                used_full += 1

        # Retrain verifier on Stage 1 + Stage 2A before Stage 2B word harvesting.
        clf2 = LinearSVC(C=0.5, max_iter=10000, dual=True, random_state=42)
        clf2.fit(F.hog_batch(np.stack(X)), np.array(y, dtype=np.int32))

        for _stem, g_doc, doc_boxes, doc_chars_list, full_lines, used_di in doc_bin_cache:
            # Tahap 2B: panen kata huruf kecil/jarang pada baris yang gagal
            # cocok utuh satu baris (mis. koma nominal menyatu), selama jumlah
            # kata cocok dan verifier menyepakati kata lain di baris itu.
            for fi, fl in enumerate(full_lines):
                words_raw = fl.split()
                word_targets = [_truth_chars(w) for w in words_raw]
                if not any(any(c.islower() or c in "QqZzXx&@" for c in w) and len(tw) >= 3 for w, tw in zip(words_raw, word_targets)):
                    continue
                for di, cbs in enumerate(doc_chars_list):
                    if di in used_di or not cbs:
                        continue
                    rel_diff = abs(fi / max(len(full_lines) - 1, 1) - di / max(len(doc_boxes) - 1, 1))
                    if rel_diff > 0.25:
                        continue
                    med_w = float(np.median([cb[2] - cb[0] + 1 for cb in cbs]))
                    thresh = max(2.0, med_w * C.SPACE_WIDTH_RATIO)
                    groups: list[list[tuple[int, int, int, int]]] = [[cbs[0]]]
                    for prev_cb, cur_cb in zip(cbs, cbs[1:]):
                        if (cur_cb[0] - prev_cb[2]) > thresh:
                            groups.append([cur_cb])
                        else:
                            groups[-1].append(cur_cb)
                    if len(groups) != len(word_targets):
                        continue

                    matched_ok = 0
                    matched_tot = 0
                    cand_pairs: list[tuple[list[tuple[int, int, int, int]], str]] = []
                    for w_orig, tw, grp in zip(words_raw, word_targets, groups):
                        if len(tw) >= 3 and len(grp) == len(tw):
                            patches = np.stack([normalize_char(g_doc, cb) for cb in grp])
                            preds = [C.IDX_TO_CHAR[int(c)].upper() for c in clf2.predict(F.hog_batch(patches))]
                            hits = sum(1 for p, ch in zip(preds, tw) if p == ch.upper())
                            matched_ok += hits
                            matched_tot += len(tw)
                            if hits / len(tw) >= 0.50 and any(c.islower() or c in "QqZzXx&@" for c in w_orig):
                                cand_pairs.append((grp, tw))

                    if cand_pairs and matched_tot >= 4 and (matched_ok / matched_tot) >= 0.58:
                        used_di.add(di)
                        for grp, tw in cand_pairs:
                            for cb, ch in zip(grp, tw):
                                patch = normalize_char(g_doc, cb)
                                if patch.shape != (C.CHAR_SIZE, C.CHAR_SIZE):
                                    continue
                                X.append(patch)
                                y.append(C.CHAR_TO_IDX[ch])
                        used_full += 1
                        break

    used_lines = used_items + used_full

    Xa = (
        np.stack(X).astype(np.uint8)
        if X
        else np.zeros((0, C.CHAR_SIZE, C.CHAR_SIZE), np.uint8)
    )
    ya = np.array(y, dtype=np.int32)
    out_path = C.DATA_DIR / f"{args.out}.npz"
    np.savez_compressed(out_path, X=Xa, y=ya)

    counts = Counter(C.IDX_TO_CHAR[int(v)] for v in ya)
    print(f"\nTersimpan: {out_path.name}  X={Xa.shape}  kelas={len(counts)}")
    print(f"  baris dipakai: {used_lines}   baris dilewati: {skipped}")
    print("  distribusi:", " ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    digits = sum(v for k, v in counts.items() if k.isdigit())
    letters = sum(v for k, v in counts.items() if k.isalpha())
    print(f"  angka={digits}  huruf={letters}")


if __name__ == "__main__":
    main()
