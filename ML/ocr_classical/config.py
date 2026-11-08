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
SYMBOLS = ".&-/%+"
CHARSET = DIGITS + UPPER + SYMBOLS
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}

# ---------------------------------------------------------------------------
# Ukuran kanvas karakter
# ---------------------------------------------------------------------------
CHAR_SIZE = 32          # kanvas normalisasi 32x32
HOG_CELL = 8            # 32/8 = 4 sel per sisi
HOG_BINS = 9
HOG_BLOCK = 2           # 3x3 blok bertumpuk

# 324 dimensi: (4-2+1)^2 blok x 2x2 sel x 9 bin
HOG_DIM = ((CHAR_SIZE // HOG_CELL) - HOG_BLOCK + 1) ** 2 * HOG_BLOCK ** 2 * HOG_BINS

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

# ---------------------------------------------------------------------------
# Klasifikasi
# ---------------------------------------------------------------------------
SVM_C = 1.0
SVM_MAX_ITER = 20000
CONFIDENCE_THRESHOLD = 0.55  # di bawah ini -> tandai tidak yakin

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
]

FONT_DIR = Path(r"C:\Windows\Fonts")

RENDER_SIZES = [22, 26, 30, 34]
RENDER_SCALE = 2          # render 2x lalu turunkan -> anti-aliasing mirip thermal
RENDER_MARGIN = 8
