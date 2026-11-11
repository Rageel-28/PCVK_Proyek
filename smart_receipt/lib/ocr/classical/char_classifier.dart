import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;

import 'classical_config.dart';
import 'gray_image.dart';
import 'hog.dart';

/// Hasil klasifikasi satu karakter.
class CharPrediction {
  const CharPrediction({
    required this.char,
    required this.confidence,
    required this.margin,
  });

  /// Karakter hasil prediksi.
  final String char;

  /// Keyakinan 0..1 (softmax atas skor linear).
  final double confidence;

  /// Selisih skor juara 1 dan 2. Semakin besar semakin yakin.
  final double margin;

  bool get isConfident => confidence >= kConfidenceThreshold;
}

/// Raised when the character model asset cannot be loaded or is malformed.
class CharModelException implements Exception {
  CharModelException(this.message);

  final String message;

  @override
  String toString() => 'CharModelException: $message';
}

/// Pengklasifikasi karakter: Linear SVM atas fitur HOG.
///
/// Model dilatih di Python (`ML/ocr_classical/train_char.py`) lalu
/// diekspor ke JSON. Inferensi hanya berupa:
///
///   skor[kelas] = coef[kelas] . hog + intercept[kelas]
///   karakter    = argmax(skor)
///
/// Tidak memakai library machine learning apa pun di sisi aplikasi.
class CharClassifier {
  CharClassifier();

  bool _loaded = false;

  late List<String> _charset;
  late List<int> _classes;
  late Float64List _coef; // datar: kelas x dimensi
  late Float64List _intercept;
  late int _dims;
  late double _threshold;

  bool get isLoaded => _loaded;
  int get dimensions => _dims;
  int get classCount => _classes.length;
  double get threshold => _threshold;

  /// Muat model dari aset aplikasi.
  Future<void> load({String assetPath = kModelAsset}) async {
    if (_loaded) return;
    final String raw;
    try {
      raw = await rootBundle.loadString(assetPath);
    } catch (e) {
      throw CharModelException('Gagal membaca model "$assetPath": $e');
    }
    loadFromJsonString(raw);
  }

  /// Muat model dari string JSON (dipakai unit test).
  void loadFromJsonString(String rawJson) {
    final Map<String, dynamic> json;
    try {
      json = jsonDecode(rawJson) as Map<String, dynamic>;
    } catch (e) {
      throw CharModelException('Model bukan JSON yang valid: $e');
    }
    loadFromMap(json);
  }

  void loadFromMap(Map<String, dynamic> json) {
    final Object? charset = json['charset'];
    final Object? svm = json['svm'];

    if (charset is! String || charset.isEmpty) {
      throw CharModelException('Model tidak punya "charset".');
    }
    if (svm is! Map) {
      throw CharModelException('Model tidak punya blok "svm".');
    }

    final Object? classes = svm['classes'];
    final Object? coef = svm['coef'];
    final Object? intercept = svm['intercept'];

    if (classes is! List || coef is! List || intercept is! List) {
      throw CharModelException(
        'Blok svm harus memuat classes, coef, dan intercept.',
      );
    }

    _charset = charset.split('');
    _classes = classes.map((Object? e) => (e as num).toInt()).toList();
    _intercept =
        Float64List.fromList(intercept.map((Object? e) => (e as num).toDouble()).toList());

    if (coef.isEmpty) {
      throw CharModelException('coef kosong.');
    }
    final List<Object?> firstRow = coef.first as List<Object?>;
    _dims = firstRow.length;
    if (_dims != kHogDim) {
      throw CharModelException(
        'Dimensi fitur model ($_dims) tidak sama dengan kHogDim ($kHogDim).',
      );
    }
    if (_classes.length != coef.length || _classes.length != _intercept.length) {
      throw CharModelException(
        'Jumlah classes (${_classes.length}), coef (${coef.length}), dan '
        'intercept (${_intercept.length}) harus sama.',
      );
    }

    _coef = Float64List(_classes.length * _dims);
    for (int c = 0; c < coef.length; c++) {
      final List<Object?> row = coef[c] as List<Object?>;
      if (row.length != _dims) {
        throw CharModelException(
          'Baris coef ke-$c punya ${row.length} nilai, seharusnya $_dims.',
        );
      }
      final int base = c * _dims;
      for (int i = 0; i < _dims; i++) {
        _coef[base + i] = (row[i] as num).toDouble();
      }
    }

    final Object? threshold = json['confidence_threshold'];
    _threshold = threshold is num ? threshold.toDouble() : kConfidenceThreshold;

    _loaded = true;
  }

  /// Klasifikasi satu patch karakter 32x32.
  CharPrediction predict(GrayImage patch) {
    if (!_loaded) {
      throw CharModelException('Model belum dimuat.');
    }
    final Float32List features = hog(patch);
    return predictFromFeatures(features);
  }

  /// Klasifikasi dari vektor HOG yang sudah dihitung.
  CharPrediction predictFromFeatures(Float32List features) {
    if (features.length != _dims) {
      throw ArgumentError(
        'Panjang fitur ${features.length} tidak sama dengan $_dims.',
      );
    }

    final int n = _classes.length;
    final Float64List scores = Float64List(n);

    for (int c = 0; c < n; c++) {
      final int base = c * _dims;
      double sum = _intercept[c];
      for (int i = 0; i < _dims; i++) {
        sum += _coef[base + i] * features[i];
      }
      scores[c] = sum;
    }

    // Cari skor tertinggi dan kedua tertinggi.
    int best = 0;
    double bestScore = scores[0];
    for (int c = 1; c < n; c++) {
      if (scores[c] > bestScore) {
        bestScore = scores[c];
        best = c;
      }
    }

    double second = double.negativeInfinity;
    for (int c = 0; c < n; c++) {
      if (c == best) continue;
      if (scores[c] > second) second = scores[c];
    }
    final double margin =
        second.isFinite ? bestScore - second : double.infinity;

    // Softmax stabil untuk keyakinan.
    double maxScore = scores[0];
    for (int c = 1; c < n; c++) {
      if (scores[c] > maxScore) maxScore = scores[c];
    }
    double denom = 0;
    for (int c = 0; c < n; c++) {
      denom += math.exp(scores[c] - maxScore);
    }
    final double confidence =
        denom > 0 ? math.exp(bestScore - maxScore) / denom : 0.0;

    final int classIndex = _classes[best];
    final String char = classIndex >= 0 && classIndex < _charset.length
        ? _charset[classIndex]
        : '?';

    return CharPrediction(
      char: char,
      confidence: confidence,
      margin: margin,
    );
  }
}
