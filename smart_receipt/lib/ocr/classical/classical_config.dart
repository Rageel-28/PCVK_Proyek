/// Konstanta pipeline OCR klasik.
///
/// Nilai di sini HARUS sama dengan `ML/ocr_classical/config.py`. Model
/// `ocr_char_svm.json` dilatih dengan parameter ini; bila berbeda, hasil
/// prediksi akan melenceng tanpa terlihat.
library;

/// Kanvas normalisasi karakter (persegi).
const int kCharSize = 32;

/// Ukuran sel HOG. 32 / 8 = 4 sel per sisi.
const int kHogCell = 8;

/// Jumlah bin orientasi per sel.
const int kHogBins = 9;

/// Ukuran blok (2x2 sel, bertumpuk dengan stride 1).
const int kHogBlock = 2;

/// Dimensi vektor HOG: (4-2+1)^2 blok x 2x2 sel x 9 bin = 324.
const int kHogDim = 324;

// --- Binarisasi -------------------------------------------------------------

const int kSauvolaWindow = 25;
const double kSauvolaK = 0.2;
const double kSauvolaR = 128.0;

// --- Upscale adaptif --------------------------------------------------------

/// Tinggi baris target setelah pembesaran.
const double kTargetLineHeight = 26.0;

/// Batas maksimum pembesaran.
const double kMaxUpscale = 3.0;

// --- Deteksi kertas ---------------------------------------------------------

const double kPaperMinAspect = 0.8;
const double kPaperMaxAspect = 6.0;
const double kPaperMinAreaFraction = 0.20;
const double kPaperMinFill = 0.55;

// --- Segmentasi baris -------------------------------------------------------

/// Baris dianggap berisi bila tinta > 1% lebar citra.
const double kLineMinInkRatio = 0.01;

/// Bila kurang dari 5% baris kosong, citra dianggap foto (pakai Otsu 1D).
const double kLinePhotoEmptyRatio = 0.05;

/// Gabungkan celah baris <= 2 px.
const int kLineGapMerge = 2;

/// Tinggi minimum sebuah baris.
const int kLineMinHeight = 6;

// --- Segmentasi karakter ----------------------------------------------------

const int kCharMinWidth = 2;
const int kCharMinInk = 4;

/// Closing adaptif 2x2 untuk menyambung huruf yang terputus.
const int kCharCloseKernel = 2;

/// Bbox lebih lebar dari 1.6x median -> dipecah.
const double kCharSplitRatio = 1.6;

/// Komponen > 1.6x perkiraan jumlah huruf -> coba closing.
const double kCharOversplitRatio = 1.6;

/// Celah > 2x median celah baris -> batas kata.
const double kSpaceGapRatio = 2.0;

// --- Klasifikasi ------------------------------------------------------------

/// Ambang keyakinan; di bawah ini karakter ditandai ragu.
const double kConfidenceThreshold = 0.55;

// --- Deskew -----------------------------------------------------------------

const double kDeskewMaxAngle = 10.0;
const double kDeskewCoarseStep = 1.0;
const double kDeskewFineStep = 0.1;

/// Sisi terpanjang citra saat estimasi sudut (mempercepat).
const int kDeskewMaxDimension = 700;

// --- Berkas aset ------------------------------------------------------------

const String kModelAsset = 'assets/ocr_char_svm.json';
const String kLexiconAsset = 'assets/ocr_lexicon.json';
