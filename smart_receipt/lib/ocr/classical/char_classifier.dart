/// Classifier karakter linear (LinearSVC + softmax bersuhu).
///
/// Port dari `ML/ocr/model.py`. Bobot dibaca dari `char_clf.json` yang
/// diekspor skrip Python, jadi tidak ada pelatihan di perangkat.
library;

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

class LinearCharModel {
  LinearCharModel._(
    this._coef,
    this._intercept, {
    required this.charset,
    required this.charSize,
    required this.pad,
    required this.featureDim,
    required this.temperature,
  });

  final String charset;
  final int charSize;
  final int pad;
  final int featureDim;
  final double temperature;
  final Float32List _coef;
  final Float32List _intercept;

  int get classCount => charset.length;

  static LinearCharModel fromJsonString(String source) {
    final Map<String, dynamic> payload =
        jsonDecode(source) as Map<String, dynamic>;
    final String charset = payload['charset'] as String;
    final int featureDim = payload['feature_dim'] as int;
    final int nClasses = charset.length;

    final List<dynamic> rawCoef = payload['coef'] as List<dynamic>;
    if (rawCoef.length != nClasses * featureDim) {
      throw FormatException(
        'panjang koefisien ${rawCoef.length} != $nClasses x $featureDim',
      );
    }
    final Float32List coef = Float32List(rawCoef.length);
    for (int i = 0; i < rawCoef.length; i++) {
      coef[i] = (rawCoef[i] as num).toDouble();
    }

    final List<dynamic> rawIntercept = payload['intercept'] as List<dynamic>;
    final Float32List intercept = Float32List(rawIntercept.length);
    for (int i = 0; i < rawIntercept.length; i++) {
      intercept[i] = (rawIntercept[i] as num).toDouble();
    }

    return LinearCharModel._(
      coef,
      intercept,
      charset: charset,
      charSize: payload['char_size'] as int,
      pad: payload['pad'] as int,
      featureDim: featureDim,
      temperature: (payload['temperature'] as num).toDouble(),
    );
  }

  /// Skor mentah `decision_function` untuk satu vektor fitur.
  Float64List decisionFunction(Float32List features) {
    final int dim = featureDim;
    final Float64List scores = Float64List(classCount);
    for (int c = 0; c < classCount; c++) {
      final int offset = c * dim;
      double sum = 0;
      for (int d = 0; d < dim; d++) {
        sum += _coef[offset + d] * features[d];
      }
      scores[c] = sum + _intercept[c];
    }
    return scores;
  }

  /// Probabilitas kelas lewat softmax bersuhu.
  ///
  /// [temperature] boleh ditimpa; uji paritas memakai suhu 1.0 karena
  /// `golden_vectors.json` dihitung dengan suhu netral.
  Float64List predictProba(Float32List features, {double? temperature}) {
    final Float64List scores = decisionFunction(features);
    double best = scores[0];
    for (int c = 1; c < scores.length; c++) {
      if (scores[c] > best) best = scores[c];
    }
    final double temp = temperature ?? this.temperature;
    double total = 0;
    for (int c = 0; c < scores.length; c++) {
      final double e = math.exp((scores[c] - best) / temp);
      scores[c] = e;
      total += e;
    }
    if (total > 0) {
      for (int c = 0; c < scores.length; c++) {
        scores[c] /= total;
      }
    }
    return scores;
  }

  int argmax(Float64List probs) {
    int best = 0;
    for (int c = 1; c < probs.length; c++) {
      if (probs[c] > probs[best]) best = c;
    }
    return best;
  }

  String charAt(int index) => charset[index];
}
