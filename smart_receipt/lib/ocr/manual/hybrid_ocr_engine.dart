import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../classical/classical_config.dart';
import '../classical/gray_image.dart';
import '../classical/lexicon_corrector.dart';
import '../classical/segmenter.dart';
import '../ocr_engine.dart';
import '../ocr_raw_text.dart';
import 'manual_config.dart';
import 'manual_features.dart';
import 'manual_preprocess.dart';
import 'rf_classifier.dart';
import 'word_lines.dart';

/// Satu baris hasil pipeline gabungan.
class HybridLineResult {
  const HybridLineResult({
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

/// Hasil lengkap satu pemindaian pipeline gabungan.
class HybridOcrResult {
  const HybridOcrResult({
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
  final List<HybridLineResult> lines;
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

/// Pipeline OCR gabungan: pra-proses/fitur/klasifikasi manual, segmentasi
/// dan leksikon dari pipeline klasik yang sudah ada.
///
/// Alur:
///   grayscale -> batas 2600 px -> upscale adaptif (baris -> 60 px)
///   -> Otsu global -> hapus garis tabel        [manual]
///   -> segmentasi baris & karakter             [klasik: proyeksi + Otsu 1D,
///                                               closing 2x2, split lembah]
///   -> normalisasi 64x48 + fitur 172 dim       [manual]
///   -> Random Forest 100 pohon                 [manual]
///   -> spasi dari median celah + Levenshtein   [klasik]
///
/// Yang diambil dari pipeline klasik adalah segmentasi yang lebih toleran
/// (huruf menempel dipecah di lembah proyeksi, stroke terputus disambung
/// closing) dan kamus 3.641 kata. Yang diambil dari pipeline manual adalah
/// pra-proses berbasis Otsu dan pengenal karakter berbasis fitur manual.
class HybridOcrEngine implements OcrEngine {
  HybridOcrEngine({
    this.useLexicon = true,
    this.minLineHeight = kLineMinHeight,
    RandomForestClassifier? classifier,
    LexiconCorrector? lexicon,
  })  : _classifier = classifier ?? RandomForestClassifier(),
        _lexicon = lexicon ?? LexiconCorrector();

  /// Koreksi leksikon aktif.
  final bool useLexicon;

  /// Tinggi minimum baris. Bawaan pipeline klasik (6 px) menangkap baris
  /// pendek seperti `Rp` dan garis pemisah; naikkan bila garis putus-putus
  /// pada struk ikut terbaca sebagai baris teks.
  final int minLineHeight;

  final RandomForestClassifier _classifier;
  final LexiconCorrector _lexicon;

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
  HybridOcrResult run(GrayImage source) {
    if (!_classifier.isLoaded) {
      throw OcrEngineException('Model belum dimuat. Panggil load() dulu.');
    }

    final GrayImage image = loadAndUpscale(source);
    final GrayImage ink = prepareManualMask(image);
    final double upscale = source.width == 0 ? 1.0 : image.width / source.width;

    final List<HybridLineResult> lines = <HybridLineResult>[];
    int lexiconFixes = 0;

    for (final ({int top, int bottom}) lineBox
        in segmentLines(ink, minHeight: minLineHeight)) {
      final List<({int left, int top, int right, int bottom})> boxes =
          segmentCharacters(ink, lineBox);
      if (boxes.isEmpty) continue;

      final List<ManualChar> chars = <ManualChar>[];
      for (final ({int left, int top, int right, int bottom}) box in boxes) {
        final GrayImage patch =
            ink.crop(box.left, box.top, box.right, box.bottom);
        bool hasInk = false;
        for (int i = 0; i < patch.length; i++) {
          if (patch.data[i] != 0) {
            hasInk = true;
            break;
          }
        }
        if (!hasInk) continue;

        final ManualCharPrediction prediction =
            _classifier.predict(manualFeatures(patch));
        chars.add((
          char: prediction.char,
          confidence: prediction.confidence,
          top: box.top,
          left: box.left,
          height: box.bottom - box.top + 1,
          width: box.right - box.left + 1,
        ));
      }
      if (chars.isEmpty) continue;

      final String rawLineText = manualJoinWithSpaces(chars);

      final List<ManualWord> words = manualSplitWords(chars);
      final List<String> correctedWords = <String>[];
      for (final ManualWord word in words) {
        if (useLexicon && _lexicon.isLoaded) {
          final ({String word, int distance, bool changed}) result =
              _lexicon.correct(word.text);
          if (result.changed) lexiconFixes++;
          correctedWords.add(result.word);
        } else {
          correctedWords.add(word.text);
        }
      }

      final double confidence = chars
              .map((ManualChar c) => c.confidence)
              .reduce((double a, double b) => a + b) /
          chars.length;

      lines.add(HybridLineResult(
        text: correctedWords.join(' '),
        rawText: rawLineText,
        confidence: confidence,
        top: lineBox.top,
        bottom: lineBox.bottom,
        chars: chars,
      ));
    }

    final List<ManualChar> allChars = <ManualChar>[
      for (final HybridLineResult line in lines) ...line.chars,
    ];

    return HybridOcrResult(
      text: lines.map((HybridLineResult l) => l.text).join('\n'),
      rawText: lines.map((HybridLineResult l) => l.rawText).join('\n'),
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
  Future<HybridOcrResult> runFile(String path) async {
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
    final HybridOcrResult result = await runFile(imagePath);

    final List<OcrLine> out = <OcrLine>[];
    for (final HybridLineResult line in result.lines) {
      final List<ManualWord> words = manualSplitWords(line.chars);
      for (final ManualWord word in words) {
        String text = word.text;
        if (useLexicon && _lexicon.isLoaded) {
          text = _lexicon.correct(word.text).word;
        }
        out.add(OcrLine(
          text: text,
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
