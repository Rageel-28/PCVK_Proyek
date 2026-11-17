"""Segmentasi baris dan karakter pada citra struk biner.

Baris dipisahkan dengan proyeksi horizontal.

Karakter dipisahkan dengan pendekatan hibrida:
  1. Morphological closing kecil untuk menyatukan bagian huruf yang
     terpecah akibat ambang (misal kaki huruf yang tipis).
  2. Connected components -> kandidat bbox.
  3. Gabungkan komponen yang tumpang tindih horizontal (bagian dari
     huruf yang sama, misalnya titik pada huruf).
  4. Pecah bbox yang terlalu lebar pada lembah proyeksi vertikal
     (dua huruf yang tintanya menyatu).

Pendekatan ini jauh lebih tahan terhadap noise dibanding proyeksi
murni, karena proyeksi menggabungkan semua huruf dalam satu kolom.
"""

from __future__ import annotations

import cv2
import numpy as np

import config as C
import resize_cubic as RC


# ---------------------------------------------------------------------------
# Utilitas run-length
# ---------------------------------------------------------------------------
def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    """Ubah array boolean menjadi daftar rentang (start, end) inklusif."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask_1d):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d) - 1))
    return runs


def _merge_gaps(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return runs
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# ---------------------------------------------------------------------------
# Baris
# ---------------------------------------------------------------------------
def _otsu_1d(values: np.ndarray) -> float:
    """Otsu untuk data 1 dimensi (profil), tanpa ketergantungan cv2.

    Dipakai untuk memisahkan baris berisi teks dari celah/baseline noise
    pada foto struk, di mana ambang absolut tidak pernah tercapai.
    """
    v = np.clip(values, 0, 255).astype(np.uint8)
    hist = np.bincount(v, minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    sum_all = float(np.dot(np.arange(256), hist))
    weight_bg = 0.0
    sum_bg = 0.0
    best_var = -1.0
    best_t = 0.0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var = var
            best_t = float(t)
    return best_t


def segment_lines(
    ink: np.ndarray,
    min_ink_ratio: float = C.LINE_MIN_INK_RATIO,
    gap_merge: int = C.LINE_GAP_MERGE,
    min_height: int = 6,
) -> list[tuple[int, int]]:
    """Kembalikan daftar (top, bottom) tiap baris teks.

    Dua mode dipilih otomatis dari bentuk profil baris:

    - **Citra bersih** (hasil render sintetis): celah antarbaris benar-benar
      nol, sehingga ambang absolut sudah tepat dan paling stabil.
    - **Foto struk**: tekstur kertas dan bayangan menghasilkan tinta tipis
      di hampir semua baris. Ambang absolut tidak akan pernah tercapai,
      jadi dipakai Otsu 1D pada profil untuk memisahkan puncak teks dari
      baseline noise.

    Pemilihannya didasarkan pada rasio baris yang mengandung tinta: pada
    citra bersih hanya baris teks yang berisi, pada foto hampir semua.
    """
    if ink.size == 0 or not ink.any():
        return []

    h, w = ink.shape
    profile = (ink > 0).sum(axis=1).astype(np.float32)

    absolute = max(1.0, w * min_ink_ratio)

    # Pembeda yang andal: pada citra bersih selalu ada baris yang benar-benar
    # kosong (celah antarbaris). Pada foto struk, tekstur kertas membuat
    # hampir tidak ada baris yang kosong sama sekali.
    empty_ratio = float((profile == 0).mean())
    is_photo = empty_ratio < C.LINE_PHOTO_EMPTY_RATIO

    if is_photo:
        threshold = max(_otsu_1d(profile), absolute)
    else:
        threshold = absolute

    runs = _runs(profile >= threshold)
    runs = _merge_gaps(runs, gap_merge)
    runs = [(s, e) for s, e in runs if (e - s + 1) >= min_height]

    # Jika foto memiliki noise tepi kertas, beberapa baris bisa menyatu jadi satu
    # blok tinggi (> 80 px dan > 15% tinggi citra). Pecah ulang dengan Otsu 1D lokal.
    tall_limit = max(80, int(0.15 * h))
    if not is_photo and any((e - s + 1) > tall_limit for s, e in runs):
        refined: list[tuple[int, int]] = []
        for s, e in runs:
            if (e - s + 1) <= tall_limit:
                refined.append((s, e))
                continue
            sub_prof = profile[s : e + 1]
            sub_t = max(_otsu_1d(sub_prof) * 0.65, absolute)
            sub_runs = _runs(sub_prof >= sub_t)
            sub_runs = _merge_gaps(sub_runs, gap_merge)
            sub_runs = [(s + a, s + b) for a, b in sub_runs if (b - a + 1) >= min_height]
            refined.extend(sub_runs if sub_runs else [(s, e)])
        runs = refined

    return runs


# ---------------------------------------------------------------------------
# Morfologi
# ---------------------------------------------------------------------------
def close_ink(ink: np.ndarray, k: int) -> np.ndarray:
    """Closing kotak k x k untuk menyambung bagian huruf yang terputus."""
    if k <= 1:
        return ink
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------
def _components(ink: np.ndarray, min_area: int) -> list[list[int]]:
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    comps: list[list[int]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            comps.append([int(x), int(y), int(x + w - 1), int(y + h - 1), int(area)])
    comps.sort(key=lambda c: c[0])
    return comps


def _merge_overlapping(comps: list[list[int]], overlap: float = 0.5) -> list[list[int]]:
    """Gabungkan komponen yang tumpang tindih horizontal (bagian huruf sama)."""
    merged: list[list[int]] = []
    for c in comps:
        if merged:
            p = merged[-1]
            lo = max(p[0], c[0])
            hi = min(p[2], c[2])
            ov = max(0, hi - lo + 1)
            narrower = min(p[2] - p[0] + 1, c[2] - c[0] + 1)
            if narrower > 0 and ov >= overlap * narrower:
                p[2] = max(p[2], c[2])
                p[1] = min(p[1], c[1])
                p[3] = max(p[3], c[3])
                p[4] += c[4]
                continue
        merged.append(list(c))
    return merged


def _split_wide(
    boxes: list[list[int]], ink: np.ndarray, target_w: float
) -> list[tuple[int, int, int, int]]:
    """Pecah bbox lebar pada lembah proyeksi vertikal."""
    out: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1, _area in boxes:
        w = x1 - x0 + 1
        if w <= target_w * C.CHAR_SPLIT_RATIO:
            out.append((x0, y0, x1, y1))
            continue

        proj = (ink[y0 : y1 + 1, x0 : x1 + 1] > 0).sum(axis=0).astype(np.int32)
        n_parts = max(2, int(round(w / target_w)))
        n_parts = min(n_parts, 10)

        cuts: list[int] = []
        for k in range(1, n_parts):
            ideal = RC._py_round_half_even(w * k / n_parts)
            margin = max(2, RC._py_round_half_even((w / n_parts) * 0.22))
            prev_cut = cuts[-1] + C.CHAR_MIN_WIDTH if cuts else 1
            lo = max(prev_cut, ideal - margin)
            hi = min(w - 1, ideal + margin)
            if hi <= lo:
                if ideal >= prev_cut and ideal < w:
                    cuts.append(ideal)
                continue
            best_x = lo
            best_cost = int(proj[lo]) * 10 + abs(lo - ideal) * 3
            for cx in range(lo + 1, hi + 1):
                cost = int(proj[cx]) * 10 + abs(cx - ideal) * 3
                if cost < best_cost:
                    best_cost = cost
                    best_x = cx
            cuts.append(best_x)

        bounds = [0] + cuts + [w]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a >= C.CHAR_MIN_WIDTH:
                out.append((x0 + a, y0, x0 + b - 1, y1))
    return out


# ---------------------------------------------------------------------------
# Karakter
# ---------------------------------------------------------------------------
def segment_characters(
    ink: np.ndarray,
    line_box: tuple[int, int],
    min_width: int = C.CHAR_MIN_WIDTH,
    min_ink: int = C.CHAR_MIN_INK,
    close_k: int = C.CHAR_CLOSE_KERNEL,
) -> list[tuple[int, int, int, int]]:
    """Kembalikan bbox (left, top, right, bottom) tiap karakter pada satu baris.

    Closing adaptif: stroke tipis (font thermal) sering terpecah menjadi
    beberapa komponen setelah binarisasi. Bila jumlah komponen jauh
    melebihi perkiraan jumlah huruf, closing 2x2 diterapkan untuk
    menyambungnya kembali. Pada citra yang sudah utuh, closing tidak
    dipakai agar huruf tidak saling menempel.
    """
    top, bottom = line_box
    h, w = ink.shape
    top = max(0, top)
    bottom = min(h - 1, bottom)
    if bottom <= top:
        return []

    strip = ink[top : bottom + 1, :]
    if not strip.any():
        return []

    def _candidates(strip_img: np.ndarray, line_h: int) -> list[list[int]]:
        comps = _components(strip_img, max(min_ink, 8))
        if not comps:
            return []
        comps = _merge_overlapping(comps, overlap=0.5)
        widths = [c[2] - c[0] + 1 for c in comps]
        heights = [c[3] - c[1] + 1 for c in comps]
        med_w = float(np.median(widths)) if widths else 8.0
        med_h = float(np.median(heights)) if heights else float(line_h)
        # Batasi med_w maksimum ke ~0.68x tinggi baris HANYA untuk baris teks
        # (bukan garis pemisah `---`/`===` yang tipis) agar huruf/angka rapat
        # yang saling menyentuh di >50% baris tetap terpecah.
        if line_h >= 14 and med_h >= max(10.0, line_h * 0.55) and (med_w / max(med_h, 1.0)) <= 1.8:
            med_w = min(med_w, max(line_h * 0.68, 3.0))
        # Pada cetakan dot-matrix / pudar, banyak huruf terpecah sehingga median lebar
        # komponen menyusut jauh di bawah lebar huruf asli (~0.48x tinggi baris).
        # Jangkar dengan tinggi baris mencegah huruf utuh ikut terpotong dua.
        target = max(med_w, line_h * 0.48, 3.0)
        boxes = _split_wide(comps, strip_img, target)
        return boxes

    line_height = bottom - top + 1
    boxes = _candidates(strip, line_height)

    # Perkiraan jumlah huruf dari lebar tinta dibagi lebar karakter tipikal.
    ink_cols = np.nonzero((strip > 0).sum(axis=0) > 0)[0]
    if len(ink_cols) > 0 and line_height > 0:
        span = int(ink_cols[-1] - ink_cols[0] + 1)
        expected = max(1, span / max(line_height * 0.55, 4.0))
        eff_k = max(close_k, 3) if (close_k > 1 and line_height >= 40) else close_k
        if len(boxes) > expected * C.CHAR_OVERSPLIT_RATIO and eff_k > 1:
            closed = close_ink(strip, eff_k)
            alt = _candidates(closed, line_height)
            if 0 < len(alt) < len(boxes):
                strip, boxes = closed, alt

    result: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in boxes:
        if (x1 - x0 + 1) < min_width:
            continue
        sub = strip[y0 : y1 + 1, x0 : x1 + 1]
        rows = np.nonzero((sub > 0).sum(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        if int((sub > 0).sum()) < min_ink:
            continue
        result.append((x0, top + y0 + int(rows[0]), x1, top + y0 + int(rows[-1])))

    result = _merge_fragments(result, line_height=bottom - top + 1)
    result.sort(key=lambda b: b[0])
    return result


def _merge_fragments(
    boxes: list[tuple[int, int, int, int]],
    line_height: int,
) -> list[tuple[int, int, int, int]]:
    """Gabungkan pecahan glyph akibat cetakan thermal yang pucat.

    Pada foto struk asli, stroke yang hilang membuat satu huruf terpecah
    menjadi dua komponen bersebelahan, mis. ``M`` menjadi dua batang
    tegak dan ``0`` menjadi dua lengkung. Tanpa penggabungan, satu huruf
    menghasilkan dua karakter palsu dan CER melonjak.

    Dua aturan, keduanya aman karena tidak mungkin terjadi pada dua huruf
    berbeda:

    1. Kotak yang **tumpang tindih** horizontal (celah <= 0) pasti satu
       glyph, sebab dua huruf tidak bisa menempati kolom piksel yang sama.
    2. Pecahan yang **sangat sempit** (lebar < 35% lebar median) dan
       menempel (celah <= 2 px) digabung ke tetangganya.
    """
    if len(boxes) < 2:
        return boxes

    widths = [b[2] - b[0] + 1 for b in boxes]
    median_w = float(np.median(widths))
    ref_w = max(median_w, line_height * 0.45)
    narrow = max(2.0, ref_w * 0.45)
    # Gabungan dua huruf asli lebarnya kira-kira 2x lebar karakter, sedangkan satu
    # glyph yang terpecah tetap selebar satu huruf. Batas 1,25x memisahkan
    # keduanya tanpa menebak-nebak.
    max_merged_w = ref_w * 1.25

    merged: list[tuple[int, int, int, int]] = [boxes[0]]
    for box in boxes[1:]:
        prev = merged[-1]
        gap = box[0] - prev[2]
        w_box = box[2] - box[0] + 1
        w_prev = prev[2] - prev[0] + 1
        merged_w = max(prev[2], box[2]) - prev[0] + 1
        should_merge = gap <= 0 or (
            gap <= 8
            and merged_w <= max_merged_w
            and (w_box < narrow or w_prev < narrow)
        )
        if should_merge:
            merged[-1] = (
                prev[0],
                min(prev[1], box[1]),
                max(prev[2], box[2]),
                max(prev[3], box[3]),
            )
        else:
            merged.append(box)
    return merged


def segment_all(ink: np.ndarray) -> list[tuple[tuple[int, int], list[tuple[int, int, int, int]]]]:
    """Segmentasi penuh: daftar (baris, daftar bbox karakter)."""
    result = []
    for line_box in segment_lines(ink):
        result.append((line_box, segment_characters(ink, line_box)))
    return result


# ---------------------------------------------------------------------------
# Debug visual
# ---------------------------------------------------------------------------
def draw_boxes(image: np.ndarray, boxes: list[tuple[int, int, int, int]], color=(0, 0, 255)) -> np.ndarray:
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for l, t, r, b in boxes:
        cv2.rectangle(out, (l, t), (r, b), color, 1)
    return out
