"""Render korpus teks menjadi dataset karakter berlabel otomatis.

Alur:

    korpus teks
        -> render tiap baris dengan font x ukuran
        -> ambil bbox tight tiap karakter dari piksel tinta
        -> normalisasi ke kanvas 32x32
        -> augmentasi (blur, noise, rotasi, erosi/dilasi, ...)
        -> (X, y, text_index)

Label didapat gratis karena kita tahu teks yang dirender. Yang penting:
bbox diambil dari **piksel tinta** (tight), bukan dari metrik font,
supaya distribusinya sama dengan hasil segmentasi saat inferensi.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ocr.config as C
from ocr.corpus import build_corpus

# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------
# Kandidat font per platform. Yang penting adalah keluarga monospace
# (struk thermal) ditambah dua sans untuk variasi.
_FONT_CANDIDATES = (
    # macOS
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/Courier.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    # Windows
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/lucon.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def find_fonts() -> list[Path]:
    """Kembalikan daftar font yang benar-benar ada di mesin ini."""
    return [Path(p) for p in _FONT_CANDIDATES if Path(p).exists()]


def _load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    """Muat font, termasuk koleksi ``.ttc`` (ambil wajah pertama)."""
    index = 0
    return ImageFont.truetype(str(path), size=size, index=index)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_line(
    text: str, font_path: Path, size: int, margin: int | None = None
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Render satu baris teks dan kembalikan bbox tight tiap karakter.

    Citra keluaran memakai konvensi proyek: ``255 = tinta``, ``0 = kertas``.
    Bbox dikembalikan sebagai ``(x0, y0, x1, y1)`` dengan ``x1``/``y1``
    eksklusif, dan hanya untuk karakter non-spasi.
    """
    if margin is None:
        margin = C.RENDER_MARGIN

    font = _load_font(font_path, size)
    advance = font.getlength(text)
    width = int(np.ceil(advance)) + 2 * margin
    height = int(np.ceil(size * 1.8)) + 2 * margin
    if width <= 2 * margin or height <= 2 * margin:
        raise ValueError(f"teks terlalu pendek untuk dirender: {text!r}")

    canvas = Image.new("L", (width, height), 255)      # kertas putih
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), text, font=font, fill=0)  # tinta hitam

    gray = 255 - np.asarray(canvas, dtype=np.uint8)    # 255 = tinta

    boxes: list[tuple[int, int, int, int]] = []
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        band_x0 = margin + int(np.floor(font.getlength(text[:i])))
        band_x1 = margin + int(np.ceil(font.getlength(text[: i + 1])))
        band_x0 = max(0, band_x0 - 1)
        band_x1 = min(width, band_x1 + 1)
        if band_x1 <= band_x0:
            continue
        band = gray[:, band_x0:band_x1]
        rows = np.where(band.any(axis=1))[0]
        cols = np.where(band.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            # Glyph tidak tersedia di font ini; lewati.
            continue
        boxes.append(
            (
                band_x0 + int(cols[0]),
                int(rows[0]),
                band_x0 + int(cols[-1]) + 1,
                int(rows[-1]) + 1,
            )
        )
    return gray, boxes


def crop_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Potong ``box`` (x0, y0, x1, y1) dari citra."""
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Normalisasi
# ---------------------------------------------------------------------------
def normalize_char(crop: np.ndarray, size: int | None = None, pad: int | None = None) -> np.ndarray:
    """Skalakan crop karakter ke kanvas persegi, rasio aspek dijaga.

    Karakter diletakkan di tengah kanvas dengan latar kertas (0).
    """
    if size is None:
        size = C.CHAR_SIZE
    if pad is None:
        pad = C.PAD

    if crop.size == 0:
        return np.zeros((size, size), dtype=np.uint8)

    h, w = crop.shape
    inner = size - 2 * pad
    scale = inner / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    new_w = min(new_w, inner)
    new_h = min(new_h, inner)

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size), dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    # Geser agar pusat massa tinta tepat di tengah kanvas. Tanpa ini, fitur
    # proyeksi/zoning sangat sensitif terhadap jitter bbox 1-2 piksel.
    ys, xs = np.nonzero(canvas)
    if ys.size:
        target = (size - 1) / 2.0
        dx = target - float(xs.mean())
        dy = target - float(ys.mean())
        if abs(dx) > 0.05 or abs(dy) > 0.05:
            m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
            canvas = cv2.warpAffine(
                canvas,
                m,
                (size, size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=C.PAPER,
            )
    return canvas


# ---------------------------------------------------------------------------
# Augmentasi
# ---------------------------------------------------------------------------
def binarize(crop: np.ndarray, threshold: int | None = None) -> np.ndarray:
    """Binerkan crop glyph: 255 = tinta, 0 = kertas.

    Inferensi selalu menerima citra biner hasil tahap binarisasi struk, jadi
    data latih juga harus biner. Ambang tetap (bukan Otsu) supaya goresan
    tipis seperti garis miring pada angka nol tidak ikut hilang.
    """
    if threshold is None:
        threshold = C.AUG_BINARIZE_THRESHOLD
    return np.where(crop > threshold, C.INK, C.PAPER).astype(np.uint8)


def _tight_crop(binary: np.ndarray) -> np.ndarray:
    """Potong ke bbox tinta terketat, seperti keluaran segmentasi."""
    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        return binary
    return binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def augment(binary: np.ndarray, seed: int) -> np.ndarray:
    """Terapkan degradasi acak pada glyph **biner** berukuran asli.

    Augmentasi dilakukan sebelum normalisasi ke 32x32, bukan sesudahnya.
    Urutan ini penting: di jalur inferensi, binarisasi terjadi pada resolusi
    asli lalu hasilnya diperkecil. Kalau augmentasi dilakukan pada kanvas
    32x32 yang sudah anti-aliased, erosi + ambang akan menghapus goresan
    tipis dan akurasi anjlok.

    Deterministik untuk ``seed`` yang sama. Keluaran selalu biner dan sudah
    dipotong ketat ke bbox tinta.
    """
    rng = np.random.default_rng(seed)
    if binary.ndim != 2:
        raise ValueError(f"glyph harus 2-d, dapat {binary.shape}")

    pad = 4
    out = np.pad(binarize(binary), pad, constant_values=C.PAPER)

    # 1. Rotasi kecil.
    angle = float(rng.uniform(-C.AUG_ROTATE_DEG_MAX, C.AUG_ROTATE_DEG_MAX))
    h, w = out.shape
    m = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle, 1.0)
    out = cv2.warpAffine(
        out, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=C.PAPER
    )

    # 2. Shear horizontal.
    shear = float(rng.uniform(-C.AUG_SHEAR_MAX, C.AUG_SHEAR_MAX))
    m = np.array(
        [[1.0, shear, -shear * h / 2], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    out = cv2.warpAffine(
        out, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=C.PAPER
    )

    # 3. Penebalan tinta. Erosi sengaja tidak dipakai: pada resolusi asli
    #    goresan hanya 2-3 piksel, sehingga erosi 2x2 memutus struktur glyph
    #    (mis. '+' kehilangan batang tegaknya dan terbaca sebagai '-').
    #    Penipisan goresan sudah terjadi di tahap binarisasi struk.
    if min(out.shape) >= 6 and rng.random() < 0.35:
        out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)

    # 4. Dropout tinta: bintik kertas thermal yang membuat sebagian piksel
    #    hilang. Sengaja hanya menghapus tinta (bukan menambah), supaya tidak
    #    ada bintik liar yang memperlebar bbox dan mengecilkan glyph.
    density = float(rng.uniform(C.AUG_SP_NOISE_MIN, C.AUG_SP_NOISE_MAX))
    dropout = (rng.random(out.shape) < density) & (out > 0)
    out[dropout] = C.PAPER

    # 5. Potong ketat: normalisasi berikutnya mengasumsikan bbox tight.
    cropped = _tight_crop(out)
    if cropped.size == 0 or not cropped.any():
        return _tight_crop(binarize(binary))
    return cropped


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _sample_line(
    text: str,
    line_idx: int,
    fonts: list[Path],
    rng: np.random.Generator,
    augment_rounds: int,
    only_label: int | None = None,
) -> list[tuple[np.ndarray, int, int, int]]:
    """Render satu baris dan kembalikan ``(gambar, label, indeks_baris, varian)``.

    ``varian`` = 0 untuk glyph bersih, 1..N untuk hasil augmentasi ke-N.
    ``only_label`` membatasi keluaran ke satu kelas; dipakai booster kelas
    langka supaya kelas umum tidak ikut membanjiri dataset.
    """
    font_path = fonts[int(rng.integers(0, len(fonts)))]
    size = int(C.RENDER_SIZES[int(rng.integers(0, len(C.RENDER_SIZES)))])
    # Struk asli mencampur huruf besar dan kecil: label seperti ``Total Item``
    # dan ``Tunai`` dicetak huruf kecil, sedangkan baris barang kapital.
    # Charset sengaja tetap kapital saja (lihat ``config.CHARSET``), jadi
    # huruf kecil dirender apa adanya tetapi **dilabeli** versi kapitalnya.
    # Dengan begitu ``o`` tidak lagi terbaca ``0`` dan ``t`` tidak ``1``.
    #
    # Polanya mengikuti struk sungguhan, bukan acak per huruf: satu baris
    # dipilih kapital semua, Title Case, atau huruf kecil semua. Mengacak
    # per huruf menghasilkan kata seperti ``TotaL`` yang tidak pernah ada di
    # struk, dan membuat bentuk kapital/huruf kecil saling mengaburkan.
    render_text = text
    if rng.random() < C.AUG_LOWERCASE_PROB:
        # ``l`` (huruf kecil L) bentuknya persis garis tegak — sama dengan
        # ``I``. Kalau dilatih sebagai label ``L``, classifier belajar bahwa
        # garis tegak boleh berarti ``L``, dan ``TOTAL`` kapital mulai
        # terbaca ``TOTAI``. Baris yang memuat huruf itu dibiarkan kapital;
        # ``L`` kapital punya kaki yang membedakannya, dan ``KEMBAII`` yang
        # salah masih bisa diperbaiki leksikon.
        if not (set(text.lower()) & C.AUG_LOWERCASE_SKIP):
            mode = int(rng.integers(0, 2))
            render_text = text.title() if mode == 1 else text.lower()
    try:
        image, boxes = render_line(render_text, font_path, size)
    except (OSError, ValueError):
        return []

    # Sempitkan seluruh baris bersama-sama, seperti properti font, bukan
    # derau per huruf. Printer thermal memakai font sempit (~0.42 lebar/tinggi)
    # sedangkan font vektor biasa ~0.85; tanpa variasi ini glyph struk asli
    # berada di luar sebaran data latih.
    condense = 1.0
    if rng.random() < C.AUG_CONDENSE_PROB:
        condense = float(
            rng.uniform(C.AUG_CONDENSE_MIN, C.AUG_CONDENSE_MAX)
        )

    out: list[tuple[np.ndarray, int, int, int]] = []
    visible = [ch for ch in render_text if ch != " "]
    for ch, box in zip(visible, boxes):
        idx = C.CHAR_TO_IDX.get(ch.upper())
        if idx is None:
            continue
        if only_label is not None and idx != only_label:
            continue
        crop = crop_box(image, box)
        if crop.size == 0:
            continue
        if condense != 1.0:
            ch_h, ch_w = crop.shape
            new_w = max(1, int(round(ch_w * condense)))
            # Glyph yang menyusut di bawah 5 px tidak lagi punya bentuk yang
            # bisa dikenali setelah diregangkan ke 32x32; sampel seperti itu
            # hanya mengaburkan batas antar kelas. Baris seperti itu dilewati.
            if new_w < 5:
                continue
            crop = cv2.resize(
                crop, (new_w, ch_h), interpolation=cv2.INTER_AREA
            )
        # Binerkan pada resolusi asli, lalu normalisasi. Ini meniru jalur
        # inferensi: struk dibinerkan dulu, glyph diperkecil sesudahnya.
        binary = binarize(crop)
        char = normalize_char(binary)
        if not char.any():
            continue

        rounds = 1 + max(0, augment_rounds)
        for r in range(rounds):
            # Round 0 selalu glyph bersih: classifier harus tetap mengenal
            # bentuk tajam, bukan hanya versi yang sudah dirusak augmentasi.
            sample = (
                char
                if r == 0
                else normalize_char(
                    augment(binary, seed=int(rng.integers(0, 2**31 - 1)))
                )
            )
            out.append((sample, idx, line_idx, r))
    return out


def _booster_line(target: str, rng: np.random.Generator) -> str:
    """Baris sintetis yang memaksa ``target`` muncul beberapa kali."""
    others = [ch for ch in C.CHARSET if ch != target]
    n_target = int(rng.integers(3, 7))
    n_other = int(rng.integers(2, 6))
    seq = [target] * n_target + [
        str(others[int(rng.integers(0, len(others)))]) for _ in range(n_other)
    ]
    order = rng.permutation(len(seq))
    return "".join(seq[i] for i in order)


def _boost_minimum(
    samples: list[tuple[np.ndarray, int, int, int]],
    fonts: list[Path],
    rng: np.random.Generator,
    augment_rounds: int,
    min_per_class: int,
) -> None:
    """Tambahkan sampel sampai setiap kelas mencapai ``min_per_class``."""
    counts = np.bincount(
        np.asarray([label for _, label, _, _ in samples], dtype=np.int64),
        minlength=len(C.CHARSET),
    )
    next_line_idx = (
        max((idx for _, _, idx, _ in samples), default=-1) + 1
    ) if samples else 0

    for class_idx in range(len(C.CHARSET)):
        need = min_per_class - int(counts[class_idx])
        if need <= 0:
            continue
        target = C.IDX_TO_CHAR[class_idx]
        added = 0
        guard = 0
        while added < need and guard < 400:
            guard += 1
            text = _booster_line(target, rng)
            produced = _sample_line(
                text,
                next_line_idx,
                fonts,
                rng,
                augment_rounds,
                only_label=class_idx,
            )
            next_line_idx += 1
            if not produced:
                continue
            samples.extend(produced)
            added += len(produced)


def _cap_maximum(
    samples: list[tuple[np.ndarray, int, int, int]],
    rng: np.random.Generator,
    max_per_class: int,
) -> list[tuple[np.ndarray, int, int, int]]:
    """Acak-hapus sampel kelas yang melebihi ``max_per_class``.

    Tanpa ini, baris garis putus-putus membuat ``-`` menguasai ~44% dataset
    dan classifier jadi bias ke satu kelas.
    """
    labels = np.asarray([label for _, label, _, _ in samples], dtype=np.int64)
    keep = np.ones(len(samples), dtype=bool)
    for class_idx in range(len(C.CHARSET)):
        idxs = np.flatnonzero(labels == class_idx)
        if idxs.size <= max_per_class:
            continue
        drop = rng.choice(idxs, size=idxs.size - max_per_class, replace=False)
        keep[drop] = False
    return [sample for sample, k in zip(samples, keep) if k]


def build_dataset(
    n_lines: int = C.DEFAULT_N_LINES,
    fonts: list[Path] | None = None,
    samples_per_line: int = 2,
    seed: int = C.RANDOM_SEED,
    augment_rounds: int = 2,
    min_per_class: int | None = None,
    max_per_class: int | None = None,
    progress_every: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bangun dataset karakter.

    Args:
        min_per_class: bila diisi, kelas yang masih kurang akan ditambah
            dengan baris sintetis sampai mencapai jumlah ini. Kelas langka
            seperti ``Q``, ``V``, ``Z``, ``&``, ``+`` memang jarang muncul di
            struk sungguhan, jadi tanpa booster ia tidak akan pernah cukup.
        max_per_class: bila diisi, kelas yang melebihi batas ini dipangkas
            secara acak. Diperlukan karena baris garis putus-putus membuat
            ``-`` sangat dominan.

    Returns:
        ``(X, y, text_index, variant)`` dengan ``X`` berbentuk
        ``(N, 32, 32)`` uint8, ``y`` indeks kelas, ``text_index`` indeks
        baris sumber (untuk split bebas kebocoran), dan ``variant`` 0 untuk
        glyph bersih / 1..N untuk hasil augmentasi ke-N.
    """
    if fonts is None:
        fonts = find_fonts()
    if not fonts:
        raise RuntimeError(
            "Tidak ada font yang ditemukan. Pasang minimal satu font "
            "monospace (mis. DejaVu Sans Mono)."
        )

    lines = build_corpus(n_lines, seed=seed)
    rng = np.random.default_rng(seed)

    samples: list[tuple[np.ndarray, int, int, int]] = []
    for line_idx, text in enumerate(lines):
        if progress_every and line_idx % progress_every == 0:
            print(f"  baris {line_idx}/{len(lines)} — {len(samples)} sampel", flush=True)
        for _ in range(samples_per_line):
            samples.extend(
                _sample_line(text, line_idx, fonts, rng, augment_rounds)
            )

    if not samples:
        raise RuntimeError("Dataset kosong: tidak ada karakter yang berhasil dirender.")

    if min_per_class:
        _boost_minimum(samples, fonts, rng, augment_rounds, min_per_class)
        if progress_every:
            print(f"  setelah booster: {len(samples)} sampel", flush=True)

    if max_per_class:
        before = len(samples)
        samples = _cap_maximum(samples, rng, max_per_class)
        if progress_every:
            print(
                f"  setelah cap {max_per_class}/kelas: {len(samples)} sampel "
                f"(dibuang {before - len(samples)})",
                flush=True,
            )

    X = np.stack([img for img, _, _, _ in samples]).astype(np.uint8)
    y = np.asarray([label for _, label, _, _ in samples], dtype=np.int64)
    text_index = np.asarray([idx for _, _, idx, _ in samples], dtype=np.int64)
    variant = np.asarray([v for _, _, _, v in samples], dtype=np.int8)
    return X, y, text_index, variant


def class_counts(y: np.ndarray) -> dict[str, int]:
    """Jumlah sampel per karakter."""
    counts = np.bincount(y, minlength=len(C.CHARSET))
    return {C.IDX_TO_CHAR[i]: int(counts[i]) for i in range(len(C.CHARSET))}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bangun dataset karakter dari korpus struk sintetis."
    )
    ap.add_argument(
        "--n-lines",
        type=int,
        default=6000,
        help="jumlah baris korpus (default 6000 agar hemat memori)",
    )
    ap.add_argument("--samples-per-line", type=int, default=1)
    ap.add_argument("--augment-rounds", type=int, default=1)
    ap.add_argument(
        "--min-per-class",
        type=int,
        default=C.MIN_SAMPLES_PER_CLASS,
        help="0 untuk menonaktifkan booster kelas langka",
    )
    ap.add_argument(
        "--max-per-class",
        type=int,
        default=C.MAX_SAMPLES_PER_CLASS,
        help="0 untuk menonaktifkan pemangkasan kelas dominan",
    )
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--out", type=str, default=str(C.DATA_DIR / "chars.npz"))
    args = ap.parse_args()

    fonts = find_fonts()
    print(f"{len(fonts)} font: {[f.name for f in fonts]}")
    print(
        f"membangun dataset: {args.n_lines} baris x {args.samples_per_line} sampel "
        f"x {1 + args.augment_rounds} varian..."
    )
    X, y, text_index, variant = build_dataset(
        n_lines=args.n_lines,
        fonts=fonts,
        samples_per_line=args.samples_per_line,
        seed=args.seed,
        augment_rounds=args.augment_rounds,
        min_per_class=args.min_per_class or None,
        max_per_class=args.max_per_class or None,
        progress_every=500,
    )

    counts = class_counts(y)
    print(f"\ntotal sampel: {len(X)}  bentuk X: {X.shape}")
    n_clean = int((variant == 0).sum())
    print(f"glyph bersih: {n_clean} ({n_clean / len(X) * 100:.1f}%)")
    low = {ch: n for ch, n in counts.items() if n < C.MIN_SAMPLES_PER_CLASS}
    if low:
        print(f"PERINGATAN kelas di bawah {C.MIN_SAMPLES_PER_CLASS}: {low}")
    else:
        print(f"semua {len(C.CHARSET)} kelas >= {C.MIN_SAMPLES_PER_CLASS} sampel")

    arr = np.asarray(list(counts.values()))
    print(
        f"distribusi kelas — min {arr.min()}  median {int(np.median(arr))}  "
        f"maks {arr.max()}  rasio {arr.max() / arr.min():.1f}x"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=X, y=y, text_index=text_index, variant=variant
    )
    print(f"disimpan ke {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
