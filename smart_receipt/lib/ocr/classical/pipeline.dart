/// Pipeline inferensi OCR klasik — padanan `ML/ocr/pipeline.py`.
///
///     citra
///       -> praproses (grayscale, denoise, Sauvola, upscale adaptif)
///       -> deteksi baris -> buang garis & barcode
///       -> segmentasi kata -> segmentasi glyph
///       -> normalisasi 32x32 -> fitur 1844-d -> classifier linear
///       -> decode: beam search + LM + leksikon -> templat
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'char_classifier.dart';
import 'char_image.dart';
import 'chars.dart';
import 'decode.dart' as decoder;
import 'features.dart';
import 'gray_image.dart';
import 'lexicon.dart';
import 'lines.dart';
import 'lm.dart';
import 'params.dart';
import 'preprocess.dart';
import 'rules.dart';
import 'templates.dart';
import 'words.dart';

const Set<String> _punctuationOnly = <String>{'.', ',', ':', ';', '/', '-', '+', '%'};

class RecognizedLine {
  RecognizedLine({
    required this.text,
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
    required this.confidence,
  });

  final String text;
  final int left;
  final int top;
  final int right;
  final int bottom;
  final double confidence;
}

class PageResult {
  PageResult({required this.lines, required this.scale, required this.readable});

  final List<RecognizedLine> lines;
  final double scale;
  final bool readable;

  String get text => lines.map((RecognizedLine l) => l.text).join('\n');
}

double _median(List<int> values) {
  if (values.isEmpty) return 0.0;
  final List<int> sorted = List<int>.from(values)..sort();
  final int mid = sorted.length ~/ 2;
  if (sorted.length.isOdd) return sorted[mid].toDouble();
  return (sorted[mid - 1] + sorted[mid]) / 2.0;
}

/// Probabilitas classifier untuk seluruh glyph dalam satu kata.
List<Float64List>? _wordProbs(
  BinaryImage band,
  int bandTop,
  int bandLeft,
  Word word,
  LinearCharModel model,
  double referenceWidth,
  double medianHeight,
) {
  final BinaryImage wordBand = band.crop(
    word.left - bandLeft,
    word.top - bandTop,
    word.right - bandLeft,
    word.bottom - bandTop,
  );
  final List<Glyph> glyphs = segmentGlyphs(
    wordBand,
    bandTop: word.top,
    bandLeft: word.left,
    referenceWidth: referenceWidth,
    referenceHeight: medianHeight,
  );
  if (glyphs.isEmpty) return null;

  final List<Float64List> probs = <Float64List>[];
  final List<bool> tiny = <bool>[];
  for (final Glyph glyph in glyphs) {
    final BinaryImage crop = band.crop(
      glyph.left - bandLeft,
      glyph.top - bandTop,
      glyph.right - bandLeft,
      glyph.bottom - bandTop,
    );
    if (crop.width == 0 || crop.height == 0) continue;
    final Uint8List glyphImage = normalizeChar(crop);
    probs.add(model.predictProba(extractFeatures(glyphImage)));
    tiny.add(medianHeight > 0 && glyph.height < medianHeight * P.tinyGlyphRatio);
  }
  if (probs.isEmpty) return null;

  if (tiny.any((bool t) => t)) {
    final List<int> allowed = P.punctuationClasses
        .split('')
        .map((String ch) => P.charset.indexOf(ch))
        .toList();
    for (int i = 0; i < probs.length; i++) {
      if (!tiny[i]) continue;
      final Float64List row = probs[i];
      double total = 0;
      for (int c = 0; c < row.length; c++) {
        if (!allowed.contains(c)) row[c] = 0;
        total += row[c];
      }
      if (total > 0) {
        for (int c = 0; c < row.length; c++) {
          row[c] /= total;
        }
      }
    }
  }
  return probs;
}

/// Lebar dan tinggi glyph acuan dari **seluruh** baris.
(double, double) lineMetrics(BinaryImage band, int bandTop, int bandLeft) {
  final List<Word> words = segmentWords(band, bandTop: bandTop, bandLeft: bandLeft);
  final List<int> widths = <int>[];
  final List<int> heights = <int>[];
  for (final Word word in words) {
    final BinaryImage wordBand = band.crop(
      word.left - bandLeft,
      word.top - bandTop,
      word.right - bandLeft,
      word.bottom - bandTop,
    );
    for (final Glyph glyph in components(wordBand)) {
      widths.add(glyph.width);
      heights.add(glyph.height);
    }
  }
  if (widths.isEmpty) return (0.0, 0.0);
  return (_median(widths), _median(heights));
}

/// Rakit teks satu baris dari glyph-glyphnya.
(String, double) decodeBand(
  BinaryImage band,
  int bandTop,
  int bandLeft,
  LinearCharModel model,
  CharLM? lm,
  Lexicon? lexicon,
) {
  final List<Word> words = segmentWords(band, bandTop: bandTop, bandLeft: bandLeft);
  if (words.isEmpty) return ('', 0.0);

  final (double referenceWidth, double medianHeight) =
      lineMetrics(band, bandTop, bandLeft);

  final List<String> pieces = <String>[];
  final List<double> confidences = <double>[];
  String prefix = '';

  for (final Word word in words) {
    final List<Float64List>? probs = _wordProbs(
      band,
      bandTop,
      bandLeft,
      word,
      model,
      referenceWidth,
      medianHeight,
    );
    if (probs == null) continue;

    String wordText;
    double confidence;
    if (lm != null) {
      final (String, double) result = decoder.decodeLine(
        probs,
        lm,
        lexicon: lexicon,
        prefix: prefix,
      );
      wordText = result.$1;
      confidence = result.$2;
    } else {
      final StringBuffer buffer = StringBuffer();
      double total = 0;
      for (final Float64List row in probs) {
        final int best = model.argmax(row);
        buffer.write(P.charset[best]);
        total += row[best];
      }
      wordText = buffer.toString();
      confidence = probs.isEmpty ? 0.0 : total / probs.length;
    }

    if (wordText.isEmpty) continue;

    // Tanda baca berdiri sendiri ditempelkan ke kata sebelumnya supaya
    // "8 . 700" tidak menjadi tiga kata.
    if (pieces.isNotEmpty &&
        wordText.split('').every(_punctuationOnly.contains)) {
      pieces[pieces.length - 1] = '${pieces.last}$wordText';
    } else {
      pieces.add(wordText);
    }

    prefix = '$prefix $wordText'.trim();
    confidences.add(confidence);
  }

  if (pieces.isEmpty) return ('', 0.0);

  String text = decoder.normalizeSpacing(pieces.join(' '));
  if (lexicon != null) {
    // Koreksi kolom nominal butuh konteks antarkata (`DISKON 50G` ->
    // `DISKON 500`), sedangkan decoder bekerja per kata. Jalankan ulang pada
    // baris utuh tanpa koreksi leksikon.
    text = decoder.correctWords(text, lexicon, useLexicon: false);
  }
  text = decoder.stripRuleArtifacts(text);
  double confidence = 0.0;
  if (confidences.isNotEmpty) {
    confidence = confidences.reduce((double a, double b) => a + b) /
        confidences.length;
  }
  return (text, confidence);
}

/// Jalankan pipeline OCR lengkap pada satu citra struk.
PageResult recognizePage(
  GrayImage image,
  LinearCharModel model, {
  CharLM? lm,
  Lexicon? lexicon,
  Templates? templates,
  double minConfidence = P.minLineConfidence,
  Prepared? prepared,
}) {
  final Prepared prep = prepared ?? prepare(image);
  final BinaryImage binary = prep.binary;

  List<LineBox> bands = detectLines(binary);
  if (bands.isEmpty) {
    return PageResult(lines: <RecognizedLine>[], scale: prep.scale, readable: false);
  }
  bands = filterBands(binary, bands);
  if (bands.isEmpty) {
    return PageResult(lines: <RecognizedLine>[], scale: prep.scale, readable: false);
  }

  final List<RecognizedLine> results = <RecognizedLine>[];
  for (final LineBox band in bands) {
    final BinaryImage crop = binary.crop(band.left, band.top, band.right, band.bottom);
    final (String rawText, double confidence) =
        decodeBand(crop, band.top, band.left, model, lm, lexicon);
    String text = rawText.trim();
    if (templates != null && text.isNotEmpty) {
      text = templates.snap(text);
    }
    if (text.isEmpty) continue;
    if (confidence < minConfidence) continue;
    results.add(RecognizedLine(
      text: text,
      left: band.left,
      top: band.top,
      right: band.right,
      bottom: band.bottom,
      confidence: confidence,
    ));
  }

  return PageResult(
    lines: results,
    scale: prep.scale,
    readable: results.length >= 2,
  );
}

/// Konversi citra RGB (dari paket `image`) ke grayscale dengan bobot
/// yang sama seperti `cv2.COLOR_BGR2GRAY`.
GrayImage grayFromRgb(
  Uint8List rgb,
  int width,
  int height,
) {
  final GrayImage out = GrayImage.zeros(width, height);
  for (int i = 0, p = 0; i < out.length; i++, p += 3) {
    final int r = rgb[p];
    final int g = rgb[p + 1];
    final int b = rgb[p + 2];
    // cv2: 0.299 R + 0.587 G + 0.114 B, dibulatkan.
    out.data[i] = ((r * 0.299 + g * 0.587 + b * 0.114) + 0.5)
        .floor()
        .clamp(0, 255);
  }
  return out;
}

/// Pastikan nilai hasil pembulatan tetap dalam jangkauan byte.
int clampByte(int value) => math.min(255, math.max(0, value));
