import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:smart_receipt/ocr/classical/gray_image.dart';
import 'package:smart_receipt/ocr/manual/manual_config.dart';
import 'package:smart_receipt/ocr/manual/manual_features.dart';
import 'package:smart_receipt/ocr/manual/manual_lexicon.dart';
import 'package:smart_receipt/ocr/manual/manual_ocr_engine.dart';
import 'package:smart_receipt/ocr/manual/manual_preprocess.dart';
import 'package:smart_receipt/ocr/manual/pil_resize.dart';
import 'package:smart_receipt/ocr/manual/rf_classifier.dart';

/// Uji paritas port Dart terhadap pipeline OCR manual versi Python.
///
/// Golden vector dibuat oleh `ML/manual_ocr_export/export_manual_assets.py`
/// yang menjalankan fitur 172 dimensi dan Random Forest versi Python pada
/// potongan glyph asli. Bila ada yang berbeda, port Dart tidak setia dan
/// akurasi aplikasi menyimpang dari hasil laporan.
void main() {
  late Map<String, dynamic> golden;
  late Map<String, dynamic> manifest;
  late RandomForestClassifier classifier;
  late ManualLexicon lexicon;

  setUpAll(() {
    golden = jsonDecode(
      File('test/fixtures/manual_rf_golden.json').readAsStringSync(),
    ) as Map<String, dynamic>;
    manifest = jsonDecode(
      File('test/fixtures/manual_lines.json').readAsStringSync(),
    ) as Map<String, dynamic>;

    classifier = RandomForestClassifier()
      ..loadFromBytes(File('assets/ocr_char_rf.bin').readAsBytesSync());
    lexicon = ManualLexicon()
      ..loadFromJsonString(
        File('assets/ocr_manual_lexicon.json').readAsStringSync(),
      );
  });

  group('model Random Forest', () {
    test('aset model termuat dengan dimensi yang benar', () {
      expect(classifier.isLoaded, isTrue);
      expect(classifier.nFeatures, kManualFeatureDim);
      expect(classifier.treeCount, 100);
      expect(classifier.classCount, greaterThan(60));
      expect(classifier.classes, contains('A'));
      expect(classifier.classes, contains('0'));
    });

    test('menolak vektor fitur dengan dimensi salah', () {
      expect(
        () => classifier.predict(Float64List(10)),
        throwsA(isA<ArgumentError>()),
      );
    });
  });

  group('paritas fitur 172 dimensi terhadap Python', () {
    test('dimensi vektor sesuai', () {
      expect(golden['n_features'], kManualFeatureDim);
      final List<dynamic> samples = golden['samples'] as List<dynamic>;
      expect(samples.length, greaterThan(30));
    });

    test('nilai fitur cocok dengan toleransi 1e-4', () {
      final List<dynamic> samples = golden['samples'] as List<dynamic>;
      double worst = 0;
      String worstLabel = '';

      for (final dynamic raw in samples) {
        final Map<String, dynamic> sample = raw as Map<String, dynamic>;
        final GrayImage mask = _maskFromRows(
          (sample['mask'] as String).split('\n'),
        );
        final Float64List actual = manualFeatures(mask);
        final List<dynamic> expected = sample['features'] as List<dynamic>;

        expect(actual.length, expected.length);
        for (int i = 0; i < expected.length; i++) {
          final double diff =
              (actual[i] - (expected[i] as num).toDouble()).abs();
          if (diff > worst) {
            worst = diff;
            worstLabel = '${sample['label']}[$i]';
          }
        }
      }

      expect(worst, lessThan(1e-4), reason: 'fitur terjauh: $worstLabel');
    });

    test('prediksi Random Forest identik dengan Python', () {
      final List<dynamic> samples = golden['samples'] as List<dynamic>;
      int agree = 0;
      final List<String> mismatches = <String>[];

      for (final dynamic raw in samples) {
        final Map<String, dynamic> sample = raw as Map<String, dynamic>;
        final GrayImage mask = _maskFromRows(
          (sample['mask'] as String).split('\n'),
        );
        final ManualCharPrediction prediction =
            classifier.predict(manualFeatures(mask));
        if (prediction.char == sample['pred']) {
          agree++;
        } else {
          mismatches.add(
            '${sample['label']}: Dart="${prediction.char}" '
            'Python="${sample['pred']}"',
          );
        }
      }

      expect(mismatches, isEmpty);
      expect(agree, samples.length);
    });
  });

  group('pra-proses', () {
    test('Otsu memisahkan tinta dari kertas pada citra struk nyata', () {
      final img.Image? decoded = img.decodeImage(
        File('test/fixtures/manual_line_0.png').readAsBytesSync(),
      );
      final GrayImage gray = GrayImage.fromImage(img.bakeOrientation(decoded!));

      final int threshold = otsuThreshold(gray);
      expect(threshold, greaterThan(0));
      expect(threshold, lessThan(255));

      int ink = 0;
      for (int i = 0; i < gray.length; i++) {
        if (gray.data[i] < threshold) ink++;
      }
      final double ratio = ink / gray.length;
      // Cetakan struk biasanya menutupi sebagian kecil kertas.
      expect(ratio, greaterThan(0.01));
      expect(ratio, lessThan(0.6));
    });

    test('garis tabel pada citra penuh ikut dihapus', () {
      final GrayImage mask = GrayImage.zeros(50, 40);
      // Satu baris penuh tinta (garis horizontal) + satu huruf kecil.
      mask.data.fillRange(10 * 50, 11 * 50, 255);
      mask.set(20, 25, 255);
      mask.set(21, 25, 255);

      final GrayImage cleaned = removeRules(mask, minRun: 0.9);
      expect(cleaned.data[10 * 50], 0);
      expect(cleaned.data[25 * 50 + 20], 255);
    });

    test('resize PIL mengembalikan salinan saat ukuran sama', () {
      final GrayImage image = GrayImage.zeros(8, 6, 128);
      final GrayImage out = pilResize(image, 8, 6, PilFilter.bilinear);
      expect(out.data, image.data);
    });
  });

  group('leksikon manual', () {
    test('memuat kamus dari aset', () {
      expect(lexicon.isLoaded, isTrue);
      expect(lexicon.wordCount, greaterThan(300));
    });

    test('token angka: huruf mirip digit menjadi digit', () {
      final ManualCorrection c = correctToken('1I,800');
      expect(c.corrected, '11,800');
      expect(c.changed, isTrue);
    });

    test('token kata: digit di antara huruf menjadi huruf', () {
      final ManualCorrection c = correctToken('T0TAL');
      expect(c.corrected, 'TOTAL');
    });

    test('kamus memperbaiki satu huruf yang salah baca', () {
      final ({String text, List<ManualCorrection> changes}) r =
          lexicon.correctLine('IND0M1E');
      expect(r.text, 'INDOMIE');
    });

    test('jarak edit dihitung benar', () {
      expect(editDistance('COKLAT', 'COKLAT'), 0);
      expect(editDistance('COK1AT', 'COKLAT'), 1);
      expect(editDistance('', 'ABC'), 3);
    });
  });

  group('pipeline end-to-end pada citra uji', () {
    late ManualOcrEngine engine;

    setUpAll(() {
      engine = ManualOcrEngine(classifier: classifier, lexicon: lexicon);
    });

    test('teks identik dengan referensi Python', () async {
      final List<dynamic> images = manifest['images'] as List<dynamic>;
      expect(images, isNotEmpty);

      for (final dynamic raw in images) {
        final Map<String, dynamic> entry = raw as Map<String, dynamic>;
        final String path = 'test/fixtures/${entry['file']}';
        final ManualOcrResult result = await engine.runFile(path);

        final List<dynamic> expectedLines = entry['lines'] as List<dynamic>;
        expect(
          result.lines.length,
          expectedLines.length,
          reason: 'jumlah baris berbeda pada ${entry['file']}',
        );
        expect(
          result.text,
          entry['text'],
          reason: 'teks berbeda pada ${entry['file']}',
        );
      }
    });

    test('hasil memuat baris dengan geometri untuk parser', () async {
      final Map<String, dynamic> entry =
          (manifest['images'] as List<dynamic>).first as Map<String, dynamic>;
      final ManualOcrResult result =
          await engine.runFile('test/fixtures/${entry['file']}');
      expect(result.lines, isNotEmpty);
      expect(result.lines.first.chars, isNotEmpty);
      expect(result.upscaleFactor, greaterThanOrEqualTo(1.0));
      expect(result.meanConfidence, inInclusiveRange(0.0, 1.0));
    });
  });
}

GrayImage _maskFromRows(List<String> rows) {
  final int height = rows.length;
  final int width = rows.first.length;
  final GrayImage mask = GrayImage.zeros(width, height);
  for (int y = 0; y < height; y++) {
    final String row = rows[y];
    for (int x = 0; x < width; x++) {
      mask.data[y * width + x] = row.codeUnitAt(x) == 49 ? 255 : 0;
    }
  }
  return mask;
}
