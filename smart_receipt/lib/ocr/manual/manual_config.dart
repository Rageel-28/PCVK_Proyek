/// Konstanta pipeline OCR manual (fitur 172 dimensi + Random Forest).
///
/// Nilai di sini HARUS sama dengan `ML/Percobaan OCR Manual/tools/annotate_lines.py`
/// dan `rebuild_feature_cache.py`. Model `ocr_char_rf.bin` dilatih dengan
/// parameter ini; bila berbeda, akurasi aplikasi menyimpang dari laporan
/// tanpa terlihat.
library;

// --- Berkas aset ------------------------------------------------------------

/// Model Random Forest 100 pohon (biner, 70 kelas, 172 fitur).
const String kManualRfAsset = 'assets/ocr_char_rf.bin';

/// Kosakata koreksi leksikal (kata umum struk + katalog produk).
const String kManualLexiconAsset = 'assets/ocr_manual_lexicon.json';

// --- Ukuran citra -----------------------------------------------------------

/// Batas sisi terpanjang saat memuat citra.
const int kManualMaxDimension = 2600;

/// Tinggi baris yang dituju saat citra diperbesar.
const double kManualTargetLineHeight = 60.0;

/// Batas pembesaran agar citra sangat kecil tidak meledak memorinya.
const double kManualMaxUpscale = 6.0;

// --- Segmentasi -------------------------------------------------------------

/// Baris diterima bila tingginya di rentang ini (piksel).
const int kManualMinLineHeight = 18;
const int kManualMaxLineHeight = 160;

/// Jarak vertikal minimal antar baris sebelum dianggap baris baru.
const int kManualRowGap = 4;

/// Baris dianggap berisi bila tinta > 0,2% lebar citra.
const double kManualLineInkRatio = 0.002;

/// Lebar minimum komponen glyph.
const int kManualGlyphMinWidth = 2;

/// Komponen lebih pendek dari 10% median baris dianggap bintik noise.
const double kManualNoiseHeightRatio = 0.10;

/// Celah antar glyph > 2x median celah dianggap batas kata.
const double kManualSpaceGapRatio = 2.0;

// --- Fitur ------------------------------------------------------------------

/// Kanvas normalisasi glyph: lebar x tinggi (urutan PIL resize).
const int kManualCharWidth = 64;
const int kManualCharHeight = 48;

/// Grid 12x12 (144) + kontur atas/bawah (24) + bentuk global (4).
const int kManualFeatureDim = 172;

// --- Pra-proses -------------------------------------------------------------

/// Baris/kolom dianggap garis tabel bila >= 90% pikselnya bertinta.
const double kManualRuleRunRatio = 0.90;

// --- Klasifikasi ------------------------------------------------------------

/// Di bawah fraksi suara ini karakter ditandai ragu.
const double kManualConfidenceThreshold = 0.55;

/// Penanda daun pada model biner (sklearn memakai -2; di biner jadi 255).
const int kManualLeafSentinel = 255;

/// Nomor versi berkas model; harus sama dengan skrip ekspor.
const String kManualRfMagic = 'RF01';
