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

/// Dimensi vektor fitur: 324 HOG + 16 grid tinta 4x4 + 4 geometri = 344.
const int kHogDim = 344;

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

/// Ambang spasi berbasis lebar karakter: celah > 0.9x lebar karakter median
/// dianggap batas kata. Dipakai karena tanda baca punya celah lebar di kedua
/// sisinya, sehingga ukuran berbasis celah menyisipkan spasi di tengah angka.
const double kSpaceWidthRatio = 0.9;

/// Penggabungan pecahan glyph. Kotak yang tumpang tindih selalu digabung;
/// kotak sempit yang menempel (<= 8 px) digabung bila hasil gabungan masih
/// selebar satu huruf (<= 1.25x lebar median).
const int kCharFragmentMaxGap = 8;
const double kCharFragmentNarrowRatio = 0.45;
const double kCharFragmentMaxWidthRatio = 1.25;

// --- Batasan kandidat per konteks kata --------------------------------------

/// Kandidat saat kata didominasi angka (kolom harga, qty, total).
const String kCharsetNumeric = '0123456789.,-/:%()=+';
const String kCharsetNumericPrice = '0123456789.,-%()+';
const String kCharsetDigitsOnly = '0123456789';

/// Kandidat saat kata didominasi huruf (nama produk, nama toko).
const String kCharsetAlpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ.&/-';
const String kCharsetAlphaNoDot = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ&/';
const String kCharsetAlphaLetters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

/// Karakter yang boleh muncul di dalam token angka / token huruf. Dipakai
/// untuk memutuskan apakah sebuah kata "murni angka" atau "murni huruf".
const String kNumericTokenChars = '0123456789.,-/:%()=+';
const String kNumericLookalikeChars = 'OISBGZ';
const String kNumericPriceLookalikeChars = 'ODQISBGZ';
const String kAlphaTokenChars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ.&/-';

/// Kandidat untuk glyph sangat kecil (koma, titik, hubung). Glyph setinggi
/// 3-4 px tidak punya informasi bentuk yang cukup untuk HOG, jadi kandidatnya
/// dibatasi agar hasilnya deterministik.
const String kCharsetPunct = '.,-';
const double kTinyGlyphHeightRatio = 0.60;
const double kTinyGlyphWidthRatio = 0.55;

/// Panjang minimum token yang boleh dikoreksi leksikon. Di bawah ini jumlah
/// tetangga pada jarak edit 1 terlalu banyak, sehingga koreksi hampir selalu
/// salah. Harus sama dengan `MIN_LEN` di `ML/ocr_classical/lexicon.py`.
const int kLexiconMinLength = 5;

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
