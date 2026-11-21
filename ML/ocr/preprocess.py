"""Praproses citra struk: grayscale → denoise → binarisasi Sauvola → upscale.

Kenapa Sauvola, bukan Otsu global? Foto struk punya pencahayaan tidak rata
(bayangan tangan, lampu dari satu sisi). Otsu memilih satu ambang untuk
seluruh gambar, sehingga sisi yang gelap jadi tinta semua dan sisi terang
jadi kertas semua. Sauvola menghitung ambang per jendela lokal, jadi
keduanya tertangani.

Konvensi keluaran: ``255 = tinta``, ``0 = kertas``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import ocr.config as C


def to_gray(image: np.ndarray) -> np.ndarray:
    """Pastikan citra menjadi grayscale uint8."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        code = cv2.COLOR_BGR2GRAY if arr.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        return cv2.cvtColor(arr.astype(np.uint8), code)
    raise ValueError(f"bentuk citra tidak didukung: {arr.shape}")


def denoise(gray: np.ndarray) -> np.ndarray:
    """Redam noise sensor tanpa menggerus goresan tipis.

    Median 3x3 membunuh salt & pepper, lalu bilateral halus menjaga tepi
    huruf tetap tajam — penting karena binarisasi berikutnya sangat
    bergantung pada tepi.
    """
    out = cv2.medianBlur(gray, 3)
    out = cv2.bilateralFilter(out, d=5, sigmaColor=40, sigmaSpace=40)
    return out


def sauvola(
    gray: np.ndarray,
    window: int | None = None,
    k: float | None = None,
    r: float | None = None,
) -> np.ndarray:
    """Binarisasi Sauvola: ambang lokal dari mean dan stddev jendela.

    Rumus: ``T = m * (1 + k * (s / R - 1))`` dengan ``m`` rata-rata lokal,
    ``s`` simpangan baku lokal, ``R`` rentang dinamis (128 untuk 8-bit),
    dan ``k`` sekitar 0.2.

    Returns:
        uint8 dengan 255 = tinta, 0 = kertas.
    """
    if window is None:
        window = C.SAUVOLA_WINDOW
    if k is None:
        k = C.SAUVOLA_K
    if r is None:
        r = C.SAUVOLA_R

    img = gray.astype(np.float32)
    # Jendela harus ganjil dan tidak lebih besar dari citra.
    window = int(window)
    if window % 2 == 0:
        window += 1
    max_side = min(img.shape)
    if max_side >= 3:
        window = max(3, min(window, max_side if max_side % 2 else max_side - 1))
    else:
        return np.zeros(img.shape, dtype=np.uint8)

    mean = cv2.blur(img, (window, window))
    mean_sq = cv2.blur(img * img, (window, window))
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)

    threshold = mean * (1.0 + k * (std / r - 1.0))
    # Piksel lebih gelap dari ambang lokal = tinta.
    local = img < threshold

    # Sauvola gagal di daerah gelap yang lebar dan seragam: di situ mean
    # lokal ikut turun, jadi ambangnya berada di bawah piksel itu sendiri.
    # Barcode dan blok tinta padat jadi berlubang. Tambal dengan ambang
    # global yang ditaksir dari kertas (persentil terang) dan tinta
    # (persentil gelap) — bukan Otsu, karena Otsu bergeser ke nilai tinta
    # saat tinta hanya menempati sebagian kecil citra.
    if img.size:
        paper = float(np.percentile(img, 80))
        ink_ref = float(np.percentile(img, 2))
        if paper - ink_ref > 20:
            solid = img < (paper + ink_ref) / 2.0
            local = local | solid

    return (local.astype(np.uint8)) * C.INK


def binarize(gray: np.ndarray) -> np.ndarray:
    """Denoise + Sauvola. Keluaran 255 = tinta, 0 = kertas."""
    return sauvola(denoise(gray))


@dataclass(frozen=True)
class Prepared:
    """Hasil praproses lengkap beserta faktor skala yang dipakai."""

    gray: np.ndarray        # grayscale asli (sebelum upscale)
    binary: np.ndarray      # biner setelah upscale, 255 = tinta
    scale: float            # faktor upscale yang diterapkan
    skew: float = 0.0       # sudut koreksi yang diterapkan (derajat)


def _profile_score(binary: np.ndarray) -> float:
    """Ketajaman profil horizontal; maksimum saat baris sejajar sumbu-x.

    Dipakai sebagai fungsi objektif pencarian sudut. Jumlah kuadrat selisih
    antar baris besar ketika tinta terkumpul di pita sempit dan nol di
    antaranya — persis kondisi saat teks lurus.
    """
    profile = (binary == C.INK).sum(axis=1).astype(np.float64)
    if profile.size < 3:
        return 0.0
    diff = np.diff(profile)
    return float((diff * diff).sum())


def estimate_skew(
    binary: np.ndarray,
    max_angle: float | None = None,
    coarse_step: float = 0.5,
    fine_step: float = 0.1,
) -> float:
    """Taksir sudut kemiringan halaman dari citra biner.

    Metode projection profile: putar citra pada berbagai sudut, pilih sudut
    yang membuat profil horizontal paling tajam (baris paling terpisah).
    Ini klasik, deterministik, dan tidak butuh deteksi tepi yang rapuh.

    Returns:
        Sudut koreksi dalam derajat yang harus **ditambahkan** agar lurus.
        Nol bila halaman sudah lurus atau tidak ada tinta.
    """
    import cv2

    arr = np.asarray(binary)
    if arr.ndim != 2:
        raise ValueError(f"citra harus 2-d, dapat {arr.shape}")
    if not (arr == C.INK).any():
        return 0.0

    if max_angle is None:
        max_angle = C.DESKEW_MAX_ANGLE

    # Perkiraan sudut dilakukan pada citra kecil: cukup akurat, jauh lebih
    # cepat, dan tidak terpengaruh noise resolusi tinggi.
    scale = min(1.0, C.DESKEW_WORKING_HEIGHT / max(1, arr.shape[0]))
    small = cv2.resize(
        arr, (max(1, int(arr.shape[1] * scale)), max(1, int(arr.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )

    def score_at(angle: float) -> float:
        center = (small.shape[1] / 2 - 0.5, small.shape[0] / 2 - 0.5)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            small, matrix, (small.shape[1], small.shape[0]),
            flags=cv2.INTER_NEAREST, borderValue=C.PAPER,
        )
        return _profile_score(rotated)

    coarse = np.arange(-max_angle, max_angle + 1e-9, coarse_step)
    best_coarse = max(coarse, key=score_at)

    fine = np.arange(
        best_coarse - coarse_step, best_coarse + coarse_step + 1e-9, fine_step
    )
    best_fine = max(fine, key=score_at)

    # ``best_fine`` adalah rotasi yang membuat baris paling sejajar sumbu-x,
    # jadi itulah sudut koreksi yang harus diterapkan.
    return float(best_fine)


def deskew(binary: np.ndarray, max_angle: float | None = None) -> tuple[np.ndarray, float]:
    """Luruskan halaman biner.

    Returns:
        ``(citra_lurus, sudut_yang_dipakai)``. Bila koreksi lebih kecil dari
        ambang, citra asli dikembalikan apa adanya supaya tidak ada
        interpolasi yang tidak perlu.
    """
    import cv2

    arr = np.asarray(binary)
    if arr.ndim != 2:
        raise ValueError(f"citra harus 2-d, dapat {arr.shape}")

    angle = estimate_skew(arr, max_angle=max_angle)
    if abs(angle) < C.DESKEW_MIN_ANGLE:
        return arr, 0.0

    center = (arr.shape[1] / 2 - 0.5, arr.shape[0] / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        arr, matrix, (arr.shape[1], arr.shape[0]),
        flags=cv2.INTER_NEAREST, borderValue=C.PAPER,
    )
    return rotated, angle


def estimate_line_height(binary: np.ndarray) -> float:
    """Perkiraan tinggi baris dari median tinggi komponen tinta.

    Dipakai untuk memutuskan berapa banyak upscale dibutuhkan supaya glyph
    mencapai ukuran yang dilihat classifier saat latih (32 px).
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0

    # Persentil ke-75, bukan median. Median biasa mudah jatuh ke komponen
    # kecil: halaman dengan banyak garis pemisah (``-----``) menyumbang
    # ratusan potongan setinggi 2-4 px sehingga median ikut mengecil
    # (terukur 4 px padahal teks 10 px), dan upscale jadi berlebihan.
    # Persentil ke-75 tetap mewakili tinggi huruf karena huruf mengisi
    # sebagian besar komponen.
    heights = [
        cv2.boundingRect(c)[3]
        for c in contours
        if cv2.contourArea(c) >= 3
    ]
    if not heights:
        return 0.0
    return float(np.percentile(heights, 75))


def _paper_mask(gray: np.ndarray) -> np.ndarray | None:
    """Mask daerah kertas (blob terang terbesar) bila ada latar gelap.

    Foto struk yang dipegang di atas kain/meja gelap punya masalah: patch
    solid global di :func:`sauvola` memakai persentil-2 sebagai acuan tinta,
    dan di citra seperti itu persentil-2 jatuh ke **latar gelap**, bukan
    tinta. Akibatnya seluruh latar menjadi tinta dan semua baris menyatu.

    Fungsi ini mencari blob terang terbesar (kertasnya). Bila blob itu
    **menyentuh keempat tepi** bingkai, struk memenuhi bingkai dan tidak ada
    latar — kembalikan ``None`` supaya praproses berjalan seperti biasa.
    Bila blob dikelilingi latar (atau hanya menempel sebagian tepi), kembalikan
    mask-nya; pemanggil akan memotong citra ke blob itu dan membuang tinta di
    luarnya.
    """
    if gray.size == 0:
        return None
    paper_tone = float(np.percentile(gray, 90))
    threshold = paper_tone * C.PAPER_MASK_THRESHOLD_FRAC
    mask = (gray > threshold).astype(np.uint8) * C.INK

    # Open dulu untuk memotong "jembatan" terang tipis yang menghubungkan
    # kertas dengan kilau latar, lalu close untuk menutup lubang teks sehingga
    # kertas menjadi satu blob utuh.
    open_k = C.PAPER_MASK_OPEN
    close_k = C.PAPER_MASK_CLOSE
    if open_k % 2 == 0:
        open_k += 1
    if close_k % 2 == 0:
        close_k += 1
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    area = int(stats[best, cv2.CC_STAT_AREA])
    if area < mask.shape[0] * mask.shape[1] * C.PAPER_MASK_MIN_AREA_FRAC:
        return None

    out = np.zeros_like(mask)
    out[labels == best] = C.INK

    # Menyentuh keempat tepi = struk memenuhi bingkai, tidak ada latar.
    touches_all = (
        (out[0, :] == C.INK).any()
        and (out[-1, :] == C.INK).any()
        and (out[:, 0] == C.INK).any()
        and (out[:, -1] == C.INK).any()
    )
    if touches_all:
        return None
    return out


def _rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    """Putar mask biner mengikuti koreksi sudut (tetap biner)."""
    center = (mask.shape[1] / 2 - 0.5, mask.shape[0] / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        mask, matrix, (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST, borderValue=0,
    )


def artifact_mask(binary: np.ndarray) -> np.ndarray:
    """Masker piksel yang termasuk komponen artefak (``255`` = artefak).

    Foto struk sering membawa latar gelap atau bayangan di tepi bingkai.
    Sauvola menandai daerah itu sebagai tinta, dan karena komponennya
    menyambung dari atas ke bawah, **setiap** baris citra tetap punya tinta.
    Proyeksi horizontal lalu tidak pernah turun ke nol, semua baris menyatu
    menjadi satu pita raksasa, dan pita itu dibuang sebagai "graphic" — struk
    yang jelas terbaca menghasilkan nol baris.

    Komponen yang lebih lebar dari ``ARTIFACT_MAX_WIDTH_FRAC`` **dan** lebih
    tinggi dari ``ARTIFACT_MAX_HEIGHT_FRAC`` tidak mungkin satu karakter atau
    satu baris teks, jadi ditandai. Strip tipis yang menjulang lebih dari
    ``ARTIFACT_FULL_HEIGHT_FRAC`` tinggi citra juga ditandai.
    """
    h, w = binary.shape
    mask = np.zeros((h, w), np.uint8)
    if w == 0 or h == 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    drop = np.zeros(n, bool)
    for label in range(1, n):
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        huge = (
            cw > C.ARTIFACT_MAX_WIDTH_FRAC * w
            and ch > C.ARTIFACT_MAX_HEIGHT_FRAC * h
        )
        tall = (
            ch > C.ARTIFACT_FULL_HEIGHT_FRAC * h
            and cw < C.ARTIFACT_THIN_WIDTH_FRAC * w
        )
        if huge or tall:
            drop[label] = True
    mask[drop[labels]] = 255
    return mask


def artifact_mask_at_scale(binary: np.ndarray) -> np.ndarray:
    """``artifact_mask`` yang hemat memori untuk citra besar.

    Pelabelan komponen pada citra hasil upscale (bisa >9 juta piksel) butuh
    memori besar. Artefak selalu jauh lebih besar daripada karakter, jadi
    pencarian dilakukan pada versi diperkecil lalu maskernya dibesarkan lagi.
    """
    h, w = binary.shape
    factor = int(np.ceil(max(w, h) / 800.0))
    if factor <= 1:
        return artifact_mask(binary)
    small = cv2.resize(
        binary,
        (int(np.ceil(w / factor)), int(np.ceil(h / factor))),
        interpolation=cv2.INTER_NEAREST,
    )
    mask = artifact_mask(small)
    if not mask.any():
        return np.zeros((0, 0), np.uint8)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _strip_artifacts(binary: np.ndarray) -> None:
    """Hapus artefak dari ``binary`` di tempat."""
    mask = artifact_mask_at_scale(binary)
    if mask.size == 0:
        return
    binary[mask != 0] = 0


def prepare(gray: np.ndarray, target_line_height: float | None = None) -> Prepared:
    """Jalur praproses lengkap: [potong kertas] → deskew → binarisasi → upscale.

    Urutan ini penting. Kemiringan harus dikoreksi **sebelum** upscale dan
    sebelum deteksi baris; kalau tidak, satu baris teks tersebar ke puluhan
    baris piksel dan proyeksi horizontal tidak bisa memisahkannya.

    Bila struk berada di atas latar gelap, citra dipotong dulu ke daerah
    kertas (lihat :func:`_paper_mask`) supaya patch solid Sauvola tidak
    salah mengklasifikasikan latar sebagai tinta.

    Upscale dihitung dari tinggi baris hasil binarisasi, bukan ditebak. Kalau
    teks sudah cukup besar, tidak ada upscale sama sekali — upscale yang
    tidak perlu hanya memperbesar noise.
    """
    if target_line_height is None:
        target_line_height = C.TARGET_LINE_HEIGHT

    base = to_gray(gray)
    paper = _paper_mask(base)

    # 0. Potong ke daerah kertas bila ada latar gelap di sekelilingnya.
    if paper is not None:
        x, y, w, h = _largest_bbox(paper)
        margin = max(4, int(0.015 * max(w, h)))
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(base.shape[1], x + w + margin)
        y1 = min(base.shape[0], y + h + margin)
        if (x1 - x0) >= 60 and (y1 - y0) >= 60:
            base = base[y0:y1, x0:x1]
            paper = paper[y0:y1, x0:x1]
        else:
            paper = None

    # 1. Binarisasi awal, hanya untuk menaksir sudut.
    probe = binarize(base)
    angle = estimate_skew(probe)

    # 2. Putar citra grayscale (bukan biner): rotasi pada grayscale dengan
    #    interpolasi kubik menghasilkan tepi yang jauh lebih halus.
    if abs(angle) >= C.DESKEW_MIN_ANGLE:
        base = _rotate_gray(base, angle)
        if paper is not None:
            paper = _rotate_mask(paper, angle)

    # 3. Binarisasi final.
    binary = binarize(base)

    # 3b. Buang artefak latar/tepi. Harus sebelum estimasi tinggi baris:
    #     artefak setinggi citra membuat tinggi baris terukur terlalu besar
    #     (upscale jadi 1.0) sekaligus menyatukan semua baris.
    _strip_artifacts(binary)

    height = estimate_line_height(binary)
    scale = 1.0
    if height > 0:
        scale = float(np.clip(target_line_height / height, 1.0, C.MAX_UPSCALE))

    if scale > 1.01:
        size = (
            int(round(base.shape[1] * scale)),
            int(round(base.shape[0] * scale)),
        )
        up = cv2.resize(base, size, interpolation=cv2.INTER_CUBIC)
        binary = binarize(up)
        # Binarisasi ulang memperkenalkan lagi latar gelap sebagai tinta.
        _strip_artifacts(binary)
        if paper is not None:
            paper = cv2.resize(paper, size, interpolation=cv2.INTER_NEAREST)

    # 4. Buang tinta di luar kertas: erode mask dulu agar garis transisi di
    #    tepi (yang Sauvola tandai sebagai tinta) ikut hilang, lalu nol-kan
    #    daerah di luar mask. Tanpa ini garis tepi vertikal menjaga profil
    #    horizontal tetap di atas ambang dan semua baris menyatu.
    if paper is not None:
        k = int(max(3, round(C.PAPER_MASK_ERODE_FRAC * min(binary.shape))))
        if k % 2 == 0:
            k += 1
        paper = cv2.erode(
            paper, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        )
        binary[paper == 0] = 0

    return Prepared(gray=base, binary=binary, scale=scale, skew=angle)


def _largest_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box ``(x, y, w, h)`` dari blob terang terbesar dalam mask."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return 0, 0, mask.shape[1], mask.shape[0]
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (
        int(stats[best, cv2.CC_STAT_LEFT]),
        int(stats[best, cv2.CC_STAT_TOP]),
        int(stats[best, cv2.CC_STAT_WIDTH]),
        int(stats[best, cv2.CC_STAT_HEIGHT]),
    )


def _rotate_gray(gray: np.ndarray, angle: float) -> np.ndarray:
    center = (gray.shape[1] / 2 - 0.5, gray.shape[0] / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
