import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../classical/gray_image.dart';
import '../ocr_engine.dart';
import '../ocr_raw_text.dart';
import 'manual_config.dart';
import 'manual_features.dart';
import 'manual_lexicon.dart';
import 'manual_preprocess.dart';
import 'manual_segmenter.dart';
import 'rf_classifier.dart';
import 'word_lines.dart';

/// Satu baris hasil OCR manual.
class ManualLineResult {
  const ManualLineResult({
    required this.text,
    required this.rawText,
    required this.confidence,
    required this.top,
    required this.bottom,
    required this.chars,
  });

  final String text;
  final String rawText;
  final double confidence;
  final int top;
  final int bottom;
  final List<ManualChar> chars;
}

/// Hasil lengkap satu pemindaian pipeline manual.
class ManualOcrResult {
  const ManualOcrResult({
    required this.text,
    required this.rawText,
    required this.lines,
    required this.meanConfidence,
    required this.lowConfidenceChars,
    required this.lexiconFixes,
    required this.upscaleFactor,
  });

  final String text;
  final String rawText;
  final List<ManualLineResult> lines;
  final double meanConfidence;
  final int lowConfidenceChars;
  final int lexiconFixes;
  final double upscaleFactor;

  Map<String, Object?> asJson() => <String, Object?>{
        'text': text,
        'raw_text': rawText,
        'mean_confidence': double.parse(meanConfidence.toStringAsFixed(4)),
        'low_confidence_chars': lowConfidenceChars,
        'lexicon_fixes': lexiconFixes,
        'upscale_factor': double.parse(upscaleFactor.toStringAsFixed(3)),
        'n_lines': lines.length,
      };
}

/// Pipeline OCR manual: fitur 172 dimensi + Random Forest 100 pohon.
///
/// Alur:
///   grayscale -> batas 2600 px -> upscale adaptif (baris -> 60 px)
///   -> Otsu global -> hapus garis tabel
///   -> segmentasi baris (projection profile)
///   -> segmentasi glyph (connected components + merge titik i/j)
///   -> normalisasi 64x48 (stretch) -> fitur 172 dim
///   -> Random Forest -> rekonstruksi spasi -> koreksi leksikal
///
/// Seluruh tahap dijalankan di perangkat; tidak memakai library OCR siap
/// pakai dan tidak memakai deep learning.
class ManualOcrEngine implements OcrEngine {
  ManualOcrEngine({
    this.useLexicon = true,
    RandomForestClassifier? classifier,
    ManualLexicon? lexicon,
  })  : _classifier = classifier ?? RandomForestClassifier(),
        _lexicon = lexicon ?? ManualLexicon();

  /// Koreksi leksikal aktif.
  final bool useLexicon;

  final RandomForestClassifier _classifier;
  final ManualLexicon _lexicon;

  bool _ready = false;

  bool get isReady => _ready;
  RandomForestClassifier get classifier => _classifier;

  /// Muat model karakter dan kamus. Aman dipanggil berulang.
  Future<void> load() async {
    if (_ready) return;
    await _classifier.load();
    if (useLexicon) {
      try {
        await _lexicon.load();
      } catch (_) {
        // Kamus opsional: tanpa kamus OCR tetap berjalan.
      }
    }
    _ready = true;
  }

  /// Jalankan pipeline pada citra grayscale yang sudah didekode.
  ManualOcrResult run(GrayImage source) {
    if (!_classifier.isLoaded) {
      throw OcrEngineException('Model belum dimuat. Panggil load() dulu.');
    }

    final GrayImage image = loadAndUpscale(source);
    final GrayImage mask = prepareManualMask(image);
    final double upscale =
        source.width == 0 ? 1.0 : image.width / source.width;

    final List<ManualLineResult> lines = <ManualLineResult>[];
    int lexiconFixes = 0;

    for (final ManualLine line in manualDetectLines(mask)) {
      final GrayImage band = mask.crop(
        line.left,
        line.top,
        line.right,
        line.bottom,
      );
      final List<ManualGlyph> glyphs = manualSegmentGlyphs(band);
      if (glyphs.isEmpty) continue;

      // Buang bintik noise: tinggi < 10% median baris. Ambangnya relatif
      // karena tanda baca memang kecil (`.` bisa 4-8 px, huruf 55 px).
      final List<int> heights = <int>[
        for (final ManualGlyph g in glyphs) g.height,
      ];
      final double medianHeight = medianOf(heights);
      final double minHeight = math.max(3.0, medianHeight * kManualNoiseHeightRatio);

      final List<ManualChar> chars = <ManualChar>[];
      for (final ManualGlyph glyph in glyphs) {
        if (glyph.height < minHeight || glyph.width < kManualGlyphMinWidth) {
          continue;
        }
        final int absoluteLeft = line.left + glyph.left;
        final int absoluteTop = line.top + glyph.top;
        final GrayImage patch = mask.crop(
          absoluteLeft,
          absoluteTop,
          absoluteLeft + glyph.width - 1,
          absoluteTop + glyph.height - 1,
        );
        final ManualCharPrediction prediction =
            _classifier.predict(manualFeatures(patch));
        chars.add((
          char: prediction.char,
          confidence: prediction.confidence,
          top: absoluteTop,
          left: absoluteLeft,
          height: glyph.height,
          width: glyph.width,
        ));
      }
      if (chars.isEmpty) continue;

      final String rawLineText = manualJoinWithSpaces(chars);

      // Koreksi leksikal per kelompok kata. `correctManualLine` bekerja
      // per token, jadi hasilnya sama dengan mengoreksi seluruh baris.
      final List<ManualWord> words = manualSplitWords(chars);
      final List<String> correctedWords = <String>[];
      for (final ManualWord word in words) {
        if (useLexicon && _lexicon.isLoaded) {
          final ({String text, List<ManualCorrection> changes}) corrected =
              _lexicon.correctLine(word.text);
          lexiconFixes += corrected.changes.length;
          correctedWords.add(corrected.text);
        } else {
          correctedWords.add(word.text);
        }
      }

      final double confidence = chars
              .map((ManualChar c) => c.confidence)
              .reduce((double a, double b) => a + b) /
          chars.length;

      lines.add(ManualLineResult(
        text: correctedWords.join(' '),
        rawText: rawLineText,
        confidence: confidence,
        top: line.top,
        bottom: line.bottom,
        chars: chars,
      ));
    }

    final List<ManualChar> allChars = <ManualChar>[
      for (final ManualLineResult line in lines) ...line.chars,
    ];

    return ManualOcrResult(
      text: lines.map((ManualLineResult l) => l.text).join('\n'),
      rawText: lines.map((ManualLineResult l) => l.rawText).join('\n'),
      lines: lines,
      meanConfidence: allChars.isEmpty
          ? 0.0
          : allChars
                  .map((ManualChar c) => c.confidence)
                  .reduce((double a, double b) => a + b) /
              allChars.length,
      lowConfidenceChars: allChars
          .where((ManualChar c) => c.confidence < kManualConfidenceThreshold)
          .length,
      lexiconFixes: lexiconFixes,
      upscaleFactor: upscale,
    );
  }

  /// Baca berkas gambar dari disk.
  Future<ManualOcrResult> runFile(String path) async {
    final File file = File(path);
    if (!file.existsSync()) {
      throw OcrEngineException('File gambar tidak ditemukan: $path');
    }
    final List<int> bytes = await file.readAsBytes();
    final img.Image? decoded = img.decodeImage(Uint8List.fromList(bytes));
    if (decoded == null) {
      throw OcrEngineException('Gambar tidak dapat didekode: $path');
    }
    final img.Image upright = img.bakeOrientation(decoded);
    return run(GrayImage.fromImage(upright));
  }

  // -- Kontrak OcrEngine --------------------------------------------------

  /// Implementasi [OcrEngine]: satu `OcrLine` per kelompok kata.
  @override
  Future<OcrRawResult> recognize(String imagePath) async {
    if (!_ready) {
      await load();
    }
    final ManualOcrResult result = await runFile(imagePath);

    final List<OcrLine> out = <OcrLine>[];
    for (final ManualLineResult line in result.lines) {
      final List<ManualWord> words = manualSplitWords(line.chars);
      final List<String> corrected = <String>[];
      for (final ManualWord word in words) {
        if (useLexicon && _lexicon.isLoaded) {
          corrected.add(_lexicon.correctLine(word.text).text);
        } else {
          corrected.add(word.text);
        }
      }

      for (int i = 0; i < words.length; i++) {
        final ManualWord word = words[i];
        out.add(OcrLine(
          text: i < corrected.length ? corrected[i] : word.text,
          left: word.left.toDouble(),
          top: word.top.toDouble(),
          right: (word.right - 1).toDouble(),
          bottom: (word.bottom - 1).toDouble(),
        ));
      }
    }

    return OcrRawResult(lines: out, fullText: result.text);
  }

  @override
  Future<void> dispose() async {
    // Tidak ada sumber daya native yang perlu dilepas.
  }
}
