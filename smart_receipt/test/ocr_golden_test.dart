/// Uji paritas port Dart terhadap rujukan Python.
///
/// `golden_vectors.json` berisi glyph 32x32, vektor fitur hasil Python, dan
/// probabilitas classifier versi Python. Dua hal diuji terpisah:
///
/// 1. `extractFeatures` menghasilkan angka yang sama dengan Python.
/// 2. `LinearCharModel.predictProba` menghasilkan probabilitas yang sama
///    dengan Python untuk fitur yang sama.
///
/// Kalau salah satu berbeda, port Dart tidak boleh dipakai: teks hasilnya
/// akan menyimpang tanpa terlihat.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:smart_receipt/ocr/classical/char_classifier.dart';
import 'package:smart_receipt/ocr/classical/features.dart';
import 'package:smart_receipt/ocr/classical/params.dart';

void main() {
  late List<Map<String, dynamic>> vectors;
  late LinearCharModel model;

  setUpAll(() {
    final String golden =
        File('test/fixtures/ocr_golden/golden_vectors.json').readAsStringSync();
    vectors = ((jsonDecode(golden) as Map<String, dynamic>)['vectors']
            as List<dynamic>)
        .cast<Map<String, dynamic>>();

    final String modelJson =
        File('assets/ocr/char_clf.json').readAsStringSync();
    model = LinearCharModel.fromJsonString(modelJson);
  });

  test('model memuat 42 kelas dengan 1844 fitur', () {
    expect(model.classCount, 42);
    expect(model.featureDim, P.featureDim);
    expect(model.charset, P.charset);
  });

  test('fitur Dart sama dengan fitur Python', () {
    expect(vectors, isNotEmpty);
    for (final Map<String, dynamic> vector in vectors) {
      final List<dynamic> rawImage = vector['image'] as List<dynamic>;
      final Uint8List image = Uint8List.fromList(
        rawImage.map((dynamic v) => (v as num).toInt()).toList(),
      );
      final List<dynamic> rawFeature = vector['feature'] as List<dynamic>;
      final Float32List actual = extractFeatures(image);

      double maxDiff = 0;
      for (int i = 0; i < rawFeature.length; i++) {
        final double expected = (rawFeature[i] as num).toDouble();
        final double diff = (actual[i] - expected).abs();
        if (diff > maxDiff) maxDiff = diff;
      }
      expect(
        maxDiff,
        lessThan(1e-4),
        reason: 'label ${vector['label']}: selisih fitur maksimum $maxDiff',
      );
    }
  });

  test('probabilitas classifier Dart sama dengan Python', () {
    for (final Map<String, dynamic> vector in vectors) {
      final List<dynamic> rawFeature = vector['feature'] as List<dynamic>;
      final Float32List features = Float32List.fromList(
        rawFeature.map((dynamic v) => (v as num).toDouble()).toList(),
      );
      final List<dynamic> rawProbs = vector['probs'] as List<dynamic>;
      final Float64List actual = model.predictProba(features, temperature: 1.0);

      double maxDiff = 0;
      for (int i = 0; i < rawProbs.length; i++) {
        final double expected = (rawProbs[i] as num).toDouble();
        final double diff = (actual[i] - expected).abs();
        if (diff > maxDiff) maxDiff = diff;
      }
      // Golden dihitung dengan suhu 1.0 dan fitur presisi penuh; fitur di
      // berkas dibulatkan 5 desimal, jadi selisih kecil wajar.
      expect(
        maxDiff,
        lessThan(1e-4),
        reason: 'label ${vector['label']}: selisih probabilitas maksimum '
            '$maxDiff',
      );

      final int best = model.argmax(actual);
      expect(model.charAt(best), vector['pred']);
    }
  });
}
