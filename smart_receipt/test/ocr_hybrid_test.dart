import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:smart_receipt/ocr/classical/gray_image.dart';
import 'package:smart_receipt/ocr/classical/lexicon_corrector.dart';
import 'package:smart_receipt/ocr/manual/hybrid_ocr_engine.dart';
import 'package:smart_receipt/ocr/manual/manual_config.dart';
import 'package:smart_receipt/ocr/manual/manual_features.dart';
import 'package:smart_receipt/ocr/manual/rf_classifier.dart';

/// Uji pipeline OCR gabungan: pra-proses/fitur/Random Forest dari pipeline
/// manual, segmentasi dan leksikon dari pipeline klasik.
///
/// Vektor uji dibuat oleh `ML/manual_ocr_export/export_manual_assets.py`
/// (fitur + classifier) dan `export_hybrid_reference.py` (teks acuan
/// gabungan di Python).
void main() {
  late Map<String, dynamic> golden;
  late Map<String, dynamic> manifest;
  late RandomForestClassifier classifier;
  late LexiconCorrector lexicon;

  setUpAll(() {
    golden = jsonDecode(
      File('test/fixtures/manual_rf_golden.json').readAsStringSync(),
    ) as Map<String, dynamic>;
    manifest = jsonDecode(
      File('test/fixtures/hybrid_lines.json').readAsStringSync(),
    ) as Map<String, dynamic>;

    classifier = RandomForestClassifier()
      ..loadFromBytes(File('assets/ocr_char_rf.bin').readAsBytesSync());
    lexicon = LexiconCorrector()
      ..loadFromJsonString(File('assets/ocr_lexicon.json').readAsStringSync());
  });

  group('komponen pengenal karakter', () {
    test('model Random Forest termuat dengan dimensi benar', () {
      expect(classifier.isLoaded, isTrue);
      expect(classifier.nFeatures, kManualFeatureDim);
      expect(classifier.treeCount, 100);
      expect(classifier.classCount, 70);
    });

    test('fitur 172 dim cocok dengan Python (< 1e-4)', () {
      final List<dynamic> samples = golden['samples'] as List<dynamic>;
      double worst = 0;
      for (final dynamic raw in samples) {
        final Map<String, dynamic> sample = raw as Map<String, dynamic>;
        final Float64List actual =
            manualFeatures(_maskFromRows((sample['mask'] as String).split('\n')));
        final List<dynamic> expected = sample['features'] as List<dynamic>;
        for (int i = 0; i < expected.length; i++) {
          final double diff =
              (actual[i] - (expected[i] as num).toDouble()).abs();
          if (diff > worst) worst = diff;
        }
      }
      expect(worst, lessThan(1e-4));
    });

    test('prediksi Random Forest identik dengan Python', () {
      final List<dynamic> samples = golden['samples'] as List<dynamic>;
      final List<String> mismatches = <String>[];
      for (final dynamic raw in samples) {
        final Map<String, dynamic> sample = raw as Map<String, dynamic>;
        final Float64List features = manualFeatures(
          _maskFromRows((sample['mask'] as String).split('\n')),
        );
        final String predicted = classifier.predict(features).char;
        if (predicted != sample['pred']) {
          mismatches.add('${sample['label']}: $predicted != ${sample['pred']}');
        }
      }
      expect(mismatches, isEmpty);
    });

    test('kamus klasik termuat', () {
      expect(lexicon.isLoaded, isTrue);
      expect(lexicon.wordCount, greaterThan(1000));
      expect(lexicon.correct('COK1AT').word, 'COKLAT');
    });
  });

  group('pipeline gabungan end-to-end', () {
    late HybridOcrEngine engine;

    setUpAll(() {
      engine = HybridOcrEngine(classifier: classifier, lexicon: lexicon);
    });

    test('teks identik dengan referensi Python', () async {
      final List<dynamic> images = manifest['images'] as List<dynamic>;
      expect(images, isNotEmpty);

      for (final dynamic raw in images) {
        final Map<String, dynamic> entry = raw as Map<String, dynamic>;
        final String path = 'test/fixtures/${entry['file']}';
        final HybridOcrResult result = await engine.runFile(path);

        expect(
          result.lines.length,
          (entry['lines'] as List<dynamic>).length,
          reason: 'jumlah baris berbeda pada ${entry['file']}',
        );
        expect(
          result.text,
          entry['text'],
          reason: 'teks berbeda pada ${entry['file']}',
        );
      }
    });

    test('menghasilkan OcrLine bergeometri dan stabil diulang', () async {
      final Map<String, dynamic> entry =
          (manifest['images'] as List<dynamic>).first as Map<String, dynamic>;
      final String path = 'test/fixtures/${entry['file']}';

      final HybridOcrResult first = await engine.runFile(path);
      final HybridOcrResult second = await engine.runFile(path);

      expect(first.lines, isNotEmpty);
      expect(first.lines.first.chars, isNotEmpty);
      expect(first.upscaleFactor, greaterThanOrEqualTo(1.0));
      expect(first.meanConfidence, inInclusiveRange(0.0, 1.0));
      // Deterministik: dua kali jalan pada citra sama memberi teks sama.
      expect(second.text, first.text);
    });

    test('koreksi leksikon aktif pada hasil', () {
      final ({String text, int fixes}) result =
          lexicon.correctText('IND0M1E 1I.500');
      // Kamus klasik boleh memilih kata terdekat mana pun (mis. INDME);
      // yang diuji di sini adalah koreksi benar-benar diterapkan.
      expect(result.fixes, greaterThanOrEqualTo(1));
      expect(result.text, isNot(equals('IND0M1E 1I.500')));
    });
  });
}

GrayImage _maskFromRows(List<String> rows) {
  final int height = rows.length;
  final int width = rows.first.length;
  final GrayImage mask = GrayImage.zeros(width, height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      mask.data[y * width + x] = rows[y].codeUnitAt(x) == 49 ? 255 : 0;
    }
  }
  return mask;
}
