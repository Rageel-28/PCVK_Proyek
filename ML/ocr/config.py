"""Konfigurasi bersama pipeline OCR klasik.

Semua parameter yang memengaruhi hasil dikumpulkan di sini supaya
eksperimen dapat direproduksi dan ablasi mudah dilakukan.

Konvensi citra di seluruh proyek: ``255 = tinta``, ``0 = kertas``.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Lokasi
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ML_DIR = ROOT.parent
PROJECT_DIR = ML_DIR.parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"
ASSET_DIR = PROJECT_DIR / "smart_receipt" / "assets"

# ---------------------------------------------------------------------------
# Konvensi biner
# ---------------------------------------------------------------------------
INK = 255
PAPER = 0

# ---------------------------------------------------------------------------
# Charset
# ---------------------------------------------------------------------------
# Huruf besar saja: struk thermal hampir selalu kapital, dan menyatukan
# huruf besar/kecil membuat kelas saling tumpang tindih (O vs o, C vs c).
DIGITS = "0123456789"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYMBOLS = ".&-/%+"

CHARSET = DIGITS + UPPER + SYMBOLS          # 42 kelas
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}

# ---------------------------------------------------------------------------
# Kanvas karakter
# ---------------------------------------------------------------------------
CHAR_SIZE = 32          # kanvas normalisasi 32x32
PAD = 2                 # margin di dalam kanvas

# ---------------------------------------------------------------------------
# Fitur (total 1836 dimensi)
# ---------------------------------------------------------------------------
# Sel 4x4 dipilih, bukan 8x8: pada 32x32 sel 8x8 terlalu kasar untuk detail
# pembeda penting seperti garis miring pada angka nol (0 vs O).
HOG_CELL = 4
HOG_BINS = 9
HOG_BLOCK = 2
# (32/4 - 2 + 1)^2 blok x 2x2 sel x 9 bin = 49 x 36 = 1764
HOG_DIM = ((CHAR_SIZE // HOG_CELL) - HOG_BLOCK + 1) ** 2 * HOG_BLOCK ** 2 * HOG_BINS

ZONING_GRID = 4         # grid 4x4
ZONING_STATS = 3        # densitas, momen-1 x, momen-1 y
ZONING_DIM = ZONING_GRID * ZONING_GRID * ZONING_STATS

PROJECTION_COLS = 16
PROJECTION_ROWS = 8
PROJECTION_DIM = PROJECTION_COLS + PROJECTION_ROWS

# Fitur topologi: jumlah lubang dan ukuran/posisinya. Ini yang membedakan
# '0' (dua lubang karena garis miring) dari 'O' (satu lubang), '8' dari 'B',
# dan seterusnya. Tanpa ini kedua kelas nyaris tidak terpisahkan.
HOLE_MAX = 3            # lubang yang dihitung (sisanya diabaikan)
HOLE_DIM = 1 + HOLE_MAX + 2 * 2   # jumlah + 3 area + 2 x (cx, cy)

FEATURE_DIM = (
    HOG_DIM + ZONING_DIM + PROJECTION_DIM + HOLE_DIM
)   # 1764 + 48 + 24 + 8 = 1844

# ---------------------------------------------------------------------------
# Render data latih
# ---------------------------------------------------------------------------
# Ukuran font saat render; gambar dirender 2x lalu diperkecil agar
# anti-aliasing-nya menyerupai cetakan thermal.
#
# Ukuran kecil (12-16 px) sengaja **tidak** dipakai. Glyph 12 px yang
# disempitkan tinggal 3-5 px lebar, lalu diregangkan ke kanvas 32x32 menjadi
# gumpalan tanpa bentuk; sampel seperti itu tidak mungkin dikenali dan
# menurunkan akurasi validasi dari 99.9% ke 95.5% (terukur).
RENDER_SIZES = (20, 26, 32, 40)
RENDER_SCALE = 2
RENDER_MARGIN = 8

# ---------------------------------------------------------------------------
# Augmentasi
# ---------------------------------------------------------------------------
AUG_ROTATE_DEG_MAX = 3.0
AUG_SHEAR_MAX = 0.10
AUG_BLUR_SIGMA_MIN = 0.2
AUG_BLUR_SIGMA_MAX = 0.6
AUG_SP_NOISE_MIN = 0.001       # dropout tinta
AUG_SP_NOISE_MAX = 0.006
AUG_CONTRAST_MIN = 0.6
AUG_GAMMA_MIN = 0.7
AUG_GAMMA_MAX = 1.4
AUG_ELASTIC_DISP = 1            # piksel, mensimulasikan tinta melebar
AUG_BINARIZE_THRESHOLD = 127    # setelah blur, kembalikan ke biner seperti inferensi

# Penyempitan horizontal (condensed). Printer thermal memakai font sempit:
# huruf besar hanya ~0.42 kali selebar tingginya, sedangkan font vektor biasa
# ~0.85. Tanpa variasi ini classifier tidak mengenali struk asli sama sekali
# (huruf 'A' terbaca '.', 'D' terbaca '0').
AUG_CONDENSE_PROB = 0.55        # peluang satu baris memakai font sempit
AUG_CONDENSE_MIN = 0.38
AUG_CONDENSE_MAX = 0.95

# Huruf kecil. Struk asli mencampur kapital dan huruf kecil; tanpa contoh
# huruf kecil, ``Total`` terbaca ``70181`` (t->1, o->0, a->8, l->1) dan
# ``Tunai`` menjadi derau. Glyph huruf kecil tetap dilabeli kapital, jadi
# jumlah kelas tidak bertambah dan keluarannya tetap kapital.
AUG_LOWERCASE_PROB = 0.30       # peluang satu baris memakai huruf kecil

# Huruf kecil yang bentuknya bertabrakan dengan **kelas lain** sehingga
# lebih baik tidak dilatih: ``l`` sama dengan ``I``, sehingga ``TOTAL``
# kapital terbaca ``TOTAI``.
AUG_LOWERCASE_SKIP = frozenset("l")

# ---------------------------------------------------------------------------
# Target dataset
# ---------------------------------------------------------------------------
MIN_SAMPLES_PER_CLASS = 2000
MAX_SAMPLES_PER_CLASS = 8000
DEFAULT_N_LINES = 15000

# ---------------------------------------------------------------------------
# Praproses
# ---------------------------------------------------------------------------
SAUVOLA_WINDOW = 25           # jendela lokal Sauvola (px, ganjil)
SAUVOLA_K = 0.2               # sensitivitas Sauvola; 0.3 memakan goresan tipis
SAUVOLA_R = 128.0             # rentang dinamis 8-bit
TARGET_LINE_HEIGHT = 48.0     # tinggi baris yang dilihat classifier (px)
MAX_UPSCALE = 5.0             # batas upscale; teks struk bisa sangat kecil
DESKEW_MAX_ANGLE = 8.0        # derajat
DESKEW_WORKING_HEIGHT = 600   # estimasi sudut dilakukan pada citra sekecil ini
DESKEW_MIN_ANGLE = 0.3        # koreksi di bawah ini diabaikan

# Deteksi daerah kertas untuk foto struk yang dipegang di atas latar gelap.
# Latar gelap (kain, meja, tangan) membuat patch solid global Sauvola salah:
# persentil-2 jatuh ke latar, bukan tinta, sehingga seluruh latar jadi tinta
# dan semua baris menyatu. Solusinya: potong ke blob kertas (terang) terbesar,
# lalu buang tinta di luar mask itu. Hanya aktif bila struk TIDAK memenuhi
# bingkai (tidak menyentuh keempat tepi); struk penuh bingkai diproses seperti
# biasa tanpa pemotongan.
PAPER_MASK_THRESHOLD_FRAC = 0.60  # ambang terang = 0.60 * persentil-90
PAPER_MASK_OPEN = 9               # kernel open: buang jembatan latar tipis
PAPER_MASK_CLOSE = 21             # kernel close: tutup lubang teks
PAPER_MASK_MIN_AREA_FRAC = 0.15   # kertas < 15% bingkai -> anggap tidak ada
PAPER_MASK_ERODE_FRAC = 0.02      # erode mask: buang garis transisi di tepi

# Artefak latar/tepi. Komponen tinta yang lebih lebar DAN lebih tinggi dari
# separuh citra tidak mungkin satu karakter atau satu baris teks.
ARTIFACT_MAX_WIDTH_FRAC = 0.50
ARTIFACT_MAX_HEIGHT_FRAC = 0.50
# Strip tipis yang menjulang hampir setinggi citra juga artefak.
ARTIFACT_FULL_HEIGHT_FRAC = 0.80
ARTIFACT_THIN_WIDTH_FRAC = 0.10

# ---------------------------------------------------------------------------
# Segmentasi baris / kata / glyph
# ---------------------------------------------------------------------------
LINE_SPLIT_RATIO = 1.45       # pita > rasio ini x tinggi acuan dipecah
WORD_GAP_TO_WIDTH = 0.6       # ambang celah kata = rasio ini x lebar sel huruf
GLYPH_SPLIT_RATIO = 1.6       # glyph > rasio ini x lebar acuan dipecah
# Glyph lebih pendek dari rasio ini terhadap tinggi huruf baris dianggap
# tanda baca, bukan huruf/angka. Titik pada "47.200" hanya ~11 px sedangkan
# huruf ~44 px.
# Glyph yang tingginya di bawah rasio ini dianggap tanda baca, sehingga
# kandidatnya dibatasi ke ``PUNCTUATION_CLASSES``. Nilai 0.45 terlalu ketat
# untuk struk asli: koma pada ``13,400`` setinggi 23 px dari median 48 px
# (rasio 0.48), jadi lolos sebagai huruf dan terbaca ``9`` — harganya rusak.
# Pada 0.55 koma ikut terbatas ke tanda baca dan harga terbaca benar.
TINY_GLYPH_RATIO = 0.55
PUNCTUATION_CLASSES = ".-/%+&"

# ---------------------------------------------------------------------------
# Model bahasa & leksikon
# ---------------------------------------------------------------------------
LM_ORDER = 5
LM_INTERPOLATION_WEIGHT = 0.85
BEAM_WIDTH = 8
LM_WEIGHT = 0.6
LM_TRUST_THRESHOLD = 1.01     # 1.01 = LM selalu boleh membantah (nonaktif)
LEXICON_WEIGHT = 0.4
LEXICON_MAX_DISTANCE = 2.0
LEXICON_MAX_RATIO = 0.3      # batas jarak edit relatif panjang kata
LEXICON_TRUST_THRESHOLD = 0.95  # keyakinan di atas ini: leksikon tidak dipakai

# ---------------------------------------------------------------------------
# Split dataset (berdasarkan baris sumber, bukan per karakter)
# ---------------------------------------------------------------------------
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15

RANDOM_SEED = 42

for _d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
