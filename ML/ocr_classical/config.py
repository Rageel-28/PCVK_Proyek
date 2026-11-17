"""Konfigurasi bersama pipeline OCR klasik.

Semua parameter yang memengaruhi hasil dikumpulkan di sini supaya
eksperimen dapat direproduksi dan ablasi mudah dilakukan.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Lokasi
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ML_DIR = ROOT.parent
PROJECT_DIR = ML_DIR.parent

DATASET_CSV = ML_DIR / "Dataset_SmartReceipt_Sintetis.csv"
COCO_DIR = ML_DIR / "OCR Struk Belanja.v2i.coco"
ASSET_DIR = PROJECT_DIR / "smart_receipt" / "assets"

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

for _d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Karakter yang dikenali
# ---------------------------------------------------------------------------
# Huruf besar saja: struk thermal hampir selalu kapital, dan menyatukan
# huruf besar/kecil membuat kelas saling tumpang tindih (O vs o, C vs c).
DIGITS = "0123456789"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Struk Indonesia memakai pemisah ribuan koma, titik dua label, sama dengan
# untuk DPP/PPN, @ pada email, dan tanda kurung untuk nominal PPN.
SYMBOLS = ".&-/%+,:@()="
# Huruf kecil yang bentuknya berbeda dari kapitalnya (setelah normalisasi
# tight-bbox ke 32x32, c/o/s/v/w/x/z identik dengan C/O/S/V/W/X/Z).
LOWER_DISTINCT = "abdefghijklmnpqrtuy"
CHARSET = DIGITS + UPPER + SYMBOLS + LOWER_DISTINCT
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}


def canonical_char(ch: str) -> str:
    """Petakan huruf ke kelas bentuk kanoniknya (mempertahankan huruf kecil khas)."""
    return ch if ch in LOWER_DISTINCT else ch.upper()


# Kandidat per konteks kata. Dipakai saat karakter sudah diprediksi sekali
# lalu diprediksi ulang dengan batasan, sehingga huruf mirip angka (O/0,
# I/1, S/5, B/8, Z/2, G/6) tidak lagi saling tertukar di kolom harga.
NUMERIC_CHARSET = frozenset(DIGITS + ".,-/:%()=+")
NUMERIC_PRICE_CHARSET = frozenset(DIGITS + ".,-%()+")
NUMERIC_DIGITS_ONLY = frozenset(DIGITS)
ALPHA_CHARSET = frozenset(UPPER + LOWER_DISTINCT + ".&/-/")
ALPHA_NO_DOT_CHARSET = frozenset(UPPER + LOWER_DISTINCT + "&/")
ALPHA_LETTERS_CHARSET = frozenset(UPPER + LOWER_DISTINCT)

# Karakter yang boleh muncul di dalam token angka / token huruf. Dipakai
# untuk memutuskan apakah sebuah kata "murni angka" atau "murni huruf".
_NUMERIC_TOKEN_CHARS = frozenset(DIGITS + ".,-/:%()=+")
_NUMERIC_LOOKALIKE_CHARS = frozenset("OISBGZ")
_NUMERIC_PRICE_LOOKALIKE_CHARS = frozenset("ODQISBGZ")
_ALPHA_TOKEN_CHARS = frozenset(UPPER + ".&/-")


def charset_for_token(chars: list[str]) -> frozenset[str] | None:
    """Kandidat yang boleh dipakai untuk sebuah kata, atau ``None``.

    Aturan:
    1. Murni angka/pemisah -> ``NUMERIC_CHARSET`` (atau ``NUMERIC_PRICE_CHARSET``
       bila mengandung koma ribuan agar ``:`` tidak menggantikan ``8``).
    2. Murni huruf -> ``ALPHA_CHARSET``.
    3. Token ber-digit yang seluruh hurufnya hanya berupa karakter mirip angka:
       - Bila memuat pemisah harga (``,`` atau ``.``), izinkan juga ``D``/``Q``
         (seperti ``3,QO`` atau ``8.QD0``) dengan rasio >= 48%.
       - Tanpa pemisah, tetap pakai syarat ketat >= 65% dan ``O,I,S,B,G,Z``
         agar kode produk seperti ``600ML``, ``14G``, atau ``3D`` tetap aman.
    """
    if not chars:
        return None
    if all(c in _NUMERIC_TOKEN_CHARS for c in chars):
        return NUMERIC_PRICE_CHARSET if "," in chars else NUMERIC_CHARSET
    if all(c in _ALPHA_TOKEN_CHARS for c in chars):
        return ALPHA_CHARSET
    if len(chars) >= 2 and any(c.isdigit() for c in chars):
        has_sep = "," in chars or "." in chars
        allowed_mix = _NUMERIC_TOKEN_CHARS | (
            _NUMERIC_PRICE_LOOKALIKE_CHARS if has_sep else _NUMERIC_LOOKALIKE_CHARS
        )
        if all(c in allowed_mix for c in chars):
            num_count = sum(1 for c in chars if c in _NUMERIC_TOKEN_CHARS and c != ":")
            min_ratio = 0.48 if has_sep and len(chars) >= 3 else 0.65
            if num_count / len(chars) >= min_ratio:
                return NUMERIC_PRICE_CHARSET if "," in chars else NUMERIC_CHARSET
    return None

# Tanda baca kecil (koma, titik, hubung). Glyph setinggi 3-4 px tidak punya
# informasi bentuk yang cukup untuk HOG, jadi kandidatnya dibatasi agar
# hasilnya deterministik dan tidak berubah karena satu piksel.
PUNCT_CHARSET = frozenset(".,-")
TINY_GLYPH_HEIGHT_RATIO = 0.60
TINY_GLYPH_WIDTH_RATIO = 0.55

# ---------------------------------------------------------------------------
# Ukuran kanvas karakter
# ---------------------------------------------------------------------------
CHAR_SIZE = 32          # kanvas normalisasi 32x32
HOG_CELL = 8            # 32/8 = 4 sel per sisi
HOG_BINS = 9
HOG_BLOCK = 2           # 3x3 blok bertumpuk

# 324 dimensi HOG + 16 kerapatan grid 4x4 + 4 geometri tinta = 344 dimensi
HOG_BASE_DIM = ((CHAR_SIZE // HOG_CELL) - HOG_BLOCK + 1) ** 2 * HOG_BLOCK ** 2 * HOG_BINS
HOG_DIM = HOG_BASE_DIM + 20

# ---------------------------------------------------------------------------
# Binarisasi
# ---------------------------------------------------------------------------
SAUVOLA_WINDOW = 25
SAUVOLA_K = 0.2
SAUVOLA_R = 128.0

# Upscale adaptif: teks kecil sulit disegmentasi, jadi citra diperbesar
# sampai tinggi baris mendekati target. Ini menaikkan akurasi segmentasi
# secara signifikan tanpa mengubah model klasifikasi.
TARGET_LINE_HEIGHT = 26.0
MAX_UPSCALE = 3.0

# Deteksi area kertas struk
PAPER_MIN_ASPECT = 0.8      # tinggi/lebar minimal
PAPER_MAX_ASPECT = 6.0      # tinggi/lebar maksimal (struk panjang)

# ---------------------------------------------------------------------------
# Segmentasi
# ---------------------------------------------------------------------------
LINE_MIN_INK_RATIO = 0.01   # baris dianggap berisi bila tinta > 1% lebar
LINE_PHOTO_EMPTY_RATIO = 0.05  # <5% baris kosong -> dianggap foto (pakai Otsu 1D)
LINE_GAP_MERGE = 2          # gabungkan celah baris <= 2 px
CHAR_MIN_WIDTH = 2
CHAR_MAX_WIDTH_RATIO = 0.35  # karakter lebih lebar dari 35% baris -> gabung
CHAR_MIN_INK = 4
CHAR_CLOSE_KERNEL = 2        # closing adaptif bila huruf terpecah
CHAR_SPLIT_RATIO = 1.6       # bbox > 1.6x lebar median -> dipecah
CHAR_OVERSPLIT_RATIO = 1.6   # komponen > 1.6x perkiraan -> coba closing
SPACE_GAP_RATIO = 2.0        # celah > 2x median celah baris -> batas kata
# Ambang spasi berbasis lebar karakter: celah > 0.9x lebar karakter median
# dianggap batas kata. Dipakai karena tanda baca punya celah lebar di kedua
# sisinya, sehingga ukuran berbasis celah menyisipkan spasi di tengah angka.
SPACE_WIDTH_RATIO = 0.9

# ---------------------------------------------------------------------------
# Klasifikasi
# ---------------------------------------------------------------------------
SVM_C = 0.5
SVM_MAX_ITER = 20000
CONFIDENCE_THRESHOLD = 0.55  # di bawah ini -> tandai tidak yakin

# Data struk asli jauh lebih sedikit daripada data sintetis, tetapi jauh
# lebih mewakili domain sesungguhnya. Bobot ini membuat tiap sampel asli
# dihitung berkali-kali saat melatih sehingga pengaruhnya sebanding.
REAL_SAMPLE_WEIGHT = 3

# ---------------------------------------------------------------------------
# Render data sintetis
# ---------------------------------------------------------------------------
FONTS = [
    "OCRAEXT.TTF",
    "consola.ttf",
    "consolab.ttf",
    "cour.ttf",
    "courbd.ttf",
    "lucon.ttf",
    "CascadiaMono.ttf",
    "ARIALN.TTF",
    "ARIALNB.TTF",
    "arial.ttf",
    "arialbd.ttf",
    "tahoma.ttf",
    "tahomabd.ttf",
    "calibri.ttf",
    "calibrib.ttf",
    "bahnschrift.ttf",
    "segoeui.ttf",
    "segoeuib.ttf",
    "trebuc.ttf",
    "verdana.ttf",
]

FONT_DIR = Path(r"C:\Windows\Fonts")

RENDER_SIZES = [22, 26, 30, 34]
RENDER_SCALE = 2          # render 2x lalu turunkan -> anti-aliasing mirip thermal
RENDER_MARGIN = 8
