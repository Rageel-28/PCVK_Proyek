/// Parameter pipeline OCR klasik.
///
/// Nilai di sini **harus sama** dengan `ML/ocr/config.py`; kalau berbeda,
/// hasil Dart tidak akan cocok dengan hasil Python dan uji paritas gagal.
library;

class P {
  P._();

  // Charset: 42 kelas.
  static const String digits = '0123456789';
  static const String upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  static const String symbols = r'.&-/%+';
  static const String charset = digits + upper + symbols;

  // Kanvas karakter.
  static const int charSize = 32;
  static const int pad = 2;

  // Fitur HOG.
  static const int hogCell = 4;
  static const int hogBins = 9;
  static const int hogBlock = 2;
  static const int hogDim =
      ((charSize ~/ hogCell) - hogBlock + 1) *
      ((charSize ~/ hogCell) - hogBlock + 1) *
      hogBlock *
      hogBlock *
      hogBins;

  // Zoning 4x4 x (densitas, cx, cy).
  static const int zoningGrid = 4;
  static const int zoningStats = 3;
  static const int zoningDim = zoningGrid * zoningGrid * zoningStats;

  // Proyeksi 16 kolom + 8 baris.
  static const int projectionCols = 16;
  static const int projectionRows = 8;
  static const int projectionDim = projectionCols + projectionRows;

  // Topologi: jumlah lubang + 3 luas + 2 posisi (cx, cy).
  static const int holeMax = 3;
  static const int holeDim = 1 + holeMax + 2 * 2;

  static const int featureDim =
      hogDim + zoningDim + projectionDim + holeDim;

  // Praproses.
  static const int sauvolaWindow = 25;
  static const double sauvolaK = 0.2;
  static const double sauvolaR = 128.0;
  static const double targetLineHeight = 48.0;
  static const double maxUpscale = 5.0;
  static const double deskewMaxAngle = 8.0;
  static const double deskewWorkingHeight = 600.0;
  static const double deskewMinAngle = 0.3;

  // Deteksi daerah kertas (foto struk di atas latar gelap).
  static const double paperMaskThresholdFrac = 0.60;
  static const int paperMaskOpen = 9;
  static const int paperMaskClose = 21;
  static const double paperMaskMinAreaFrac = 0.15;
  static const double paperMaskErodeFrac = 0.02;

  // Segmentasi.
  static const double lineSplitRatio = 1.45;
  static const double wordGapToWidth = 0.6;
  static const double glyphSplitRatio = 1.6;
  static const double tinyGlyphRatio = 0.55;
  static const String punctuationClasses = '.-/%+&';

  // Model bahasa & leksikon.
  static const int lmOrder = 5;
  static const int beamWidth = 8;
  static const double lmWeight = 0.6;
  static const double lmTrustThreshold = 1.01;
  static const double lexiconMaxDistance = 2.0;
  static const double lexiconMaxRatio = 0.3;
  static const double lexiconTrustThreshold = 0.95;
  static const double minLineConfidence = 0.30;
}
