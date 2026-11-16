import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:smart_receipt/ocr/classical/char_classifier.dart';
import 'package:smart_receipt/ocr/classical/char_normalizer.dart';
import 'package:smart_receipt/ocr/classical/classical_config.dart';
import 'package:smart_receipt/ocr/classical/gray_image.dart';
import 'package:smart_receipt/ocr/classical/hog.dart';
import 'package:smart_receipt/ocr/classical/zoning.dart';

/// Uji paritas port Dart terhadap implementasi Python.
///
/// Golden vectors dibuat oleh `ML/ocr_classical/export_golden_vectors.py`
/// yang menjalankan HOG, Zoning, StandardScaler, dan Linear SVM versi Python.
/// Bila hasil Dart berbeda, berarti port ke Dart tidak setia dan akurasi
/// aplikasi akan menyimpang dari hasil laporan.
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
      // Tanpa PCA, dimensi masuk SVM = vektor gabungan HOG + Zoning.
      expect(classifier.combinedDimensions, kCombinedDim);
      expect(classifier.dimensions,
          classifier.usesPca ? classifier.pcaComponents : kCombinedDim);
      expect(classifier.usesScaler, isTrue);
      expect(classifier.classCount, greaterThan(30));
    });

    test('konfigurasi zoning model sama dengan konstanta aplikasi', () {
      expect(golden['zoning_grid'], kZoningGrid);
      expect(golden['zoning_dim'], kZoningDim);
      expect(golden['combined_dim'], kCombinedDim);
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
  });

  group('paritas Zoning terhadap Python', () {
    test('dimensi vektor zoning sama', () {
      final List<dynamic> samples = golden['samples'] as List<dynamic>;
      final List<dynamic> zoningVec =
          (samples.first as Map<String, dynamic>)['zoning'] as List<dynamic>;
      expect(zoningVec.length, kZoningDim);
      expect(kZoningGrid * kZoningGrid, kZoningDim);
    });

    test('nilai zoning cocok dengan toleransi 1e-4', () {
      final List<dynamic> samples = golden['samples'] as List<dynamic>;
      double worst = 0;
      for (final dynamic raw in samples) {
        final Map<String, dynamic> sample = raw as Map<String, dynamic>;
        final List<dynamic> pixels = sample['pixels'] as List<dynamic>;
        final List<dynamic> expected = sample['zoning'] as List<dynamic>;

        final GrayImage patch = GrayImage.zeros(kCharSize, kCharSize);
        for (int i = 0; i < pixels.length; i++) {
          patch.data[i] = (pixels[i] as num).toInt();
        }

        final List<double> actual = zoning(patch, grid: kZoningGrid);
        expect(actual.length, expected.length);
        for (int i = 0; i < expected.length; i++) {
          final double diff = (actual[i] - (expected[i] as num).toDouble()).abs();
          if (diff > worst) worst = diff;
        }
      }
      expect(worst, lessThan(1e-4), reason: 'selisih zoning terbesar $worst');
    });

    test('zoning merespons sebaran tinta, bukan hanya arah tepi', () {
      // Huruf kotak di kiri-atas vs kanan-bawah: densitas zonanya harus
      // berbeda, meski bentuknya sama.
      final GrayImage kiriAtas = GrayImage.zeros(kCharSize, kCharSize, 255);
      final GrayImage kananBawah = GrayImage.zeros(kCharSize, kCharSize, 255);
      for (int y = 2; y < 14; y++) {
        for (int x = 2; x < 14; x++) {
          kiriAtas.set(x, y, 0);
        }
      }
      for (int y = 18; y < 30; y++) {
        for (int x = 18; x < 30; x++) {
          kananBawah.set(x, y, 0);
        }
      }
      final List<double> a = zoning(kiriAtas, grid: kZoningGrid);
      final List<double> b = zoning(kananBawah, grid: kZoningGrid);
      expect(a.first, greaterThan(0.0));
      expect(b.first, 0.0);
      expect(b.last, greaterThan(0.0));
      expect(a.last, 0.0);
    });

    test('vektor gabungan = HOG ++ Zoning dengan urutan benar', () {
      final List<dynamic> samples = golden['samples'] as List<dynamic>;
      double worst = 0;
      for (final dynamic raw in samples) {
        final Map<String, dynamic> sample = raw as Map<String, dynamic>;
        final List<dynamic> pixels = sample['pixels'] as List<dynamic>;
        final List<dynamic> expectedCombined =
            sample['combined'] as List<dynamic>;

        final GrayImage patch = GrayImage.zeros(kCharSize, kCharSize);
        for (int i = 0; i < pixels.length; i++) {
          patch.data[i] = (pixels[i] as num).toInt();
        }

        final List<double> actual = classifier.combinedFeatures(patch);
        expect(actual.length, kCombinedDim);
        for (int i = 0; i < expectedCombined.length; i++) {
          final double diff =
              (actual[i] - (expectedCombined[i] as num).toDouble()).abs();
          if (diff > worst) worst = diff;
        }
      }
      expect(worst, lessThan(1e-4),
          reason: 'selisih vektor gabungan terbesar $worst');
    });
  });

  group('paritas prediksi terhadap Python', () {

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
