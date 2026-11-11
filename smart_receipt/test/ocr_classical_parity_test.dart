import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:smart_receipt/ocr/classical/char_classifier.dart';
import 'package:smart_receipt/ocr/classical/char_normalizer.dart';
import 'package:smart_receipt/ocr/classical/classical_config.dart';
import 'package:smart_receipt/ocr/classical/gray_image.dart';
import 'package:smart_receipt/ocr/classical/hog.dart';

/// Uji paritas port Dart terhadap implementasi Python.
///
/// Golden vectors dibuat oleh `ML/ocr_classical/export_golden_vectors.py`
/// yang menjalankan HOG dan Linear SVM versi Python. Bila hasil Dart
/// berbeda, berarti port ke Dart tidak setia dan akurasi aplikasi akan
/// menyimpang dari hasil laporan.
void main() {
  late Map<String, dynamic> golden;
  late CharClassifier classifier;

  setUpAll(() {
    final String raw =
        File('test/fixtures/golden_vectors.json').readAsStringSync();
    golden = jsonDecode(raw) as Map<String, dynamic>;

    final String modelRaw =
        File('assets/ocr_char_svm.json').readAsStringSync();
    classifier = CharClassifier()..loadFromJsonString(modelRaw);
  });

  group('model karakter', () {
    test('aset model termuat dan dimensinya sesuai', () {
      expect(classifier.isLoaded, isTrue);
      expect(classifier.dimensions, kHogDim);
      expect(classifier.classCount, greaterThan(30));
    });
  });

  group('paritas HOG terhadap Python', () {
    test('dimensi vektor HOG sama', () {
      final List<dynamic> samples = golden['samples'] as List<dynamic>;
      expect(golden['hog_dim'], kHogDim);
      expect((samples.first as Map<String, dynamic>)['hog'], isA<List<dynamic>>());
      expect(
        ((samples.first as Map<String, dynamic>)['hog'] as List<dynamic>).length,
        kHogDim,
      );
    });

    test('nilai HOG cocok dengan toleransi 1e-4', () {
      final List<dynamic> samples = golden['samples'] as List<dynamic>;
      double worst = 0;
      int worstIndex = -1;

      for (int s = 0; s < samples.length; s++) {
        final Map<String, dynamic> sample = samples[s] as Map<String, dynamic>;
        final List<dynamic> pixels = sample['pixels'] as List<dynamic>;
        final List<dynamic> expectedHog = sample['hog'] as List<dynamic>;

        final GrayImage patch =
            GrayImage.zeros(kCharSize, kCharSize);
        for (int i = 0; i < pixels.length; i++) {
          patch.data[i] = (pixels[i] as num).toInt();
        }

        final List<double> actual = hog(patch);
        for (int i = 0; i < kHogDim; i++) {
          final double diff =
              (actual[i] - (expectedHog[i] as num).toDouble()).abs();
          if (diff > worst) {
            worst = diff;
            worstIndex = s;
          }
        }
      }

      expect(
        worst,
        lessThan(1e-4),
        reason: 'selisih HOG terbesar $worst pada sampel $worstIndex',
      );
    });

    test('prediksi karakter sama dengan Python', () {
      final List<dynamic> samples = golden['samples'] as List<dynamic>;
      int mismatch = 0;
      final List<String> details = <String>[];

      for (final dynamic raw in samples) {
        final Map<String, dynamic> sample = raw as Map<String, dynamic>;
        final List<dynamic> pixels = sample['pixels'] as List<dynamic>;

        final GrayImage patch = GrayImage.zeros(kCharSize, kCharSize);
        for (int i = 0; i < pixels.length; i++) {
          patch.data[i] = (pixels[i] as num).toInt();
        }

        final CharPrediction pred = classifier.predict(patch);
        final String expected = sample['expected_char'] as String;
        if (pred.char != expected) {
          mismatch++;
          details.add(
            'truth=${sample['truth_char']} python=$expected dart=${pred.char}',
          );
        }
      }

      expect(mismatch, 0, reason: details.join(' | '));
    });

    test('keyakinan prediksi mendekati nilai Python', () {
      final List<dynamic> samples = golden['samples'] as List<dynamic>;
      for (final dynamic raw in samples) {
        final Map<String, dynamic> sample = raw as Map<String, dynamic>;
        final List<dynamic> pixels = sample['pixels'] as List<dynamic>;

        final GrayImage patch = GrayImage.zeros(kCharSize, kCharSize);
        for (int i = 0; i < pixels.length; i++) {
          patch.data[i] = (pixels[i] as num).toInt();
        }

        final CharPrediction pred = classifier.predict(patch);
        final double expected = (sample['confidence'] as num).toDouble();
        expect(
          (pred.confidence - expected).abs(),
          lessThan(1e-4),
          reason: 'keyakinan ${pred.confidence} vs $expected',
        );
      }
    });
  });

  group('normalisasi karakter', () {
    test('selalu menghasilkan kanvas 32x32', () {
      final GrayImage src = GrayImage.zeros(40, 40, 255);
      for (int y = 10; y < 30; y++) {
        for (int x = 12; x < 26; x++) {
          src.set(x, y, 0);
        }
      }
      final GrayImage out = normalizeChar(
        src,
        (left: 12, top: 10, right: 25, bottom: 29),
      );
      expect(out.width, kCharSize);
      expect(out.height, kCharSize);
    });

    test('pembulatan mengikuti Python (setengah ke genap)', () {
      expect(pythonRound(2.5), 2);
      expect(pythonRound(3.5), 4);
      expect(pythonRound(2.4), 2);
      expect(pythonRound(2.6), 3);
    });
  });
}
