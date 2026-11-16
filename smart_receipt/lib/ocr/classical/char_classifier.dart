import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;

import 'classical_config.dart';
import 'gray_image.dart';
import 'hog.dart';
import 'zoning.dart';

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

/// Pengklasifikasi karakter: HOG + Zoning -> StandardScaler -> (PCA) -> SVM.
///
/// Model dilatih di Python (`ML/ocr_classical/train_char.py`) lalu diekspor
/// ke JSON. Inferensi meniru urutan latih langkah demi langkah:
///
///   1. gabungan = HOG(324) ++ Zoning(16)              -> 340 dimensi
///   2. gabungan = (gabungan - mean) / scale           -> StandardScaler
///   3. bila model memakai PCA:
///        p = (gabungan - pca_mean) @ components.T
///   4. skor[kelas] = coef[kelas] . p + intercept[kelas]
///      karakter    = argmax(skor)
///
/// Scaler dan PCA dibiarkan eksplisit (tidak dilipat ke bobot SVM) supaya
/// tiap tahap bisa diperiksa dan dijelaskan. Melipat secara matematis
/// mungkin untuk model linear, tetapi menyembunyikan tahap yang justru
/// ingin ditunjukkan.
///
/// Tidak memakai library machine learning apa pun di sisi aplikasi.
class CharClassifier {
  CharClassifier();

  bool _loaded = false;

  late List<String> _charset;
  late List<int> _classes;
  late Float64List _coef; // datar: kelas x dimensi_masuk_svm
  late Float64List _intercept;

  /// Dimensi vektor yang masuk SVM (setelah PCA bila dipakai).
  late int _dims;

  /// Dimensi vektor gabungan sebelum PCA.
  late int _combinedDim;

  late int _hogDim;
  late int _zoningDim;
  late int _zoningGrid;
  late int _charSize;

  Float64List? _scalerMean;
  Float64List? _scalerScale;

  Float64List? _pcaMean;
  Float64List? _pcaComponents; // datar: komponen x combinedDim
  int _pcaComponentsCount = 0;

  late double _threshold;

  bool get isLoaded => _loaded;

  /// Dimensi yang masuk SVM (setelah PCA bila ada).
  int get dimensions => _dims;

  /// Dimensi vektor gabungan HOG + Zoning.
  int get combinedDimensions => _combinedDim;

  int get classCount => _classes.length;
  double get threshold => _threshold;
  bool get usesScaler => _scalerMean != null;
  bool get usesPca => _pcaComponents != null;
  int get pcaComponents => _pcaComponentsCount;

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

    _charset = charset.split('');
    _charSize = (json['char_size'] as num?)?.toInt() ?? kCharSize;

    // --- Konfigurasi fitur ------------------------------------------------
    final Object? hogCfg = json['hog'];
    if (hogCfg is Map) {
      final int cell = (hogCfg['cell'] as num?)?.toInt() ?? kHogCell;
      final int bins = (hogCfg['bins'] as num?)?.toInt() ?? kHogBins;
      final int block = (hogCfg['block'] as num?)?.toInt() ?? kHogBlock;
      if (cell != kHogCell || bins != kHogBins || block != kHogBlock) {
        throw CharModelException(
          'Parameter HOG model (cell=$cell, bins=$bins, block=$block) tidak '
          'sama dengan konstanta aplikasi ($kHogCell, $kHogBins, $kHogBlock).',
        );
      }
    }

    final Object? zoningCfg = json['zoning'];
    if (zoningCfg is Map) {
      _zoningGrid = (zoningCfg['grid'] as num?)?.toInt() ?? kZoningGrid;
      _zoningDim = (zoningCfg['dim'] as num?)?.toInt() ?? kZoningDim;
    } else {
      throw CharModelException(
        'Model tidak punya blok "zoning". Latih ulang dengan train_char.py.',
      );
    }
    if (_zoningGrid != kZoningGrid || _zoningDim != kZoningDim) {
      throw CharModelException(
        'Konfigurasi zoning model (grid=$_zoningGrid, dim=$_zoningDim) tidak '
        'sama dengan konstanta aplikasi ($kZoningGrid, $kZoningDim).',
      );
    }

    final Object? featuresCfg = json['features'];
    _hogDim = kHogDim;
    _combinedDim = kHogDim + _zoningDim;
    if (featuresCfg is Map) {
      _hogDim = (featuresCfg['hog_dim'] as num?)?.toInt() ?? _hogDim;
      _combinedDim =
          (featuresCfg['combined_dim'] as num?)?.toInt() ?? _combinedDim;
    }
    if (_hogDim != kHogDim || _combinedDim != kCombinedDim) {
      throw CharModelException(
        'Model mengharapkan $_hogDim dimensi HOG dan $_combinedDim gabungan, '
        'sedangkan aplikasi memakai $kHogDim dan $kCombinedDim.',
      );
    }

    // --- StandardScaler ---------------------------------------------------
    final Object? scaler = json['scaler'];
    if (scaler is Map) {
      final List<double> mean = _toDoubleList(scaler['mean'], 'scaler.mean');
      final List<double> scale =
          _toDoubleList(scaler['scale'], 'scaler.scale');
      if (mean.length != _combinedDim || scale.length != _combinedDim) {
        throw CharModelException(
          'Panjang scaler (${mean.length}/${scale.length}) harus '
          '$_combinedDim.',
        );
      }
      _scalerMean = Float64List.fromList(mean);
      _scalerScale = Float64List.fromList(scale);
    } else {
      _scalerMean = null;
      _scalerScale = null;
    }

    // --- PCA (opsional) ---------------------------------------------------
    final Object? pca = json['pca'];
    if (pca is Map) {
      final List<double> pcaMean =
          _toDoubleList(pca['mean'], 'pca.mean');
      final List<List<double>> comps =
          _toMatrix(pca['components'], 'pca.components');
      if (pcaMean.length != _combinedDim) {
        throw CharModelException(
          'pca.mean panjang ${pcaMean.length}, seharusnya $_combinedDim.',
        );
      }
      if (comps.isEmpty || comps.first.length != _combinedDim) {
        throw CharModelException(
          'pca.components harus berbentuk n x $_combinedDim.',
        );
      }
      _pcaMean = Float64List.fromList(pcaMean);
      _pcaComponentsCount = comps.length;
      _pcaComponents = Float64List(comps.length * _combinedDim);
      for (int i = 0; i < comps.length; i++) {
        if (comps[i].length != _combinedDim) {
          throw CharModelException('Baris pca.components ke-$i tidak konsisten.');
        }
        final int base = i * _combinedDim;
        for (int j = 0; j < _combinedDim; j++) {
          _pcaComponents![base + j] = comps[i][j];
        }
      }
    } else {
      _pcaMean = null;
      _pcaComponents = null;
      _pcaComponentsCount = 0;
    }

    // --- SVM linear -------------------------------------------------------
    final Object? classes = svm['classes'];
    final Object? coef = svm['coef'];
    final Object? intercept = svm['intercept'];

    if (classes is! List || coef is! List || intercept is! List) {
      throw CharModelException(
        'Blok svm harus memuat classes, coef, dan intercept.',
      );
    }

    _classes = classes.map((Object? e) => (e as num).toInt()).toList();
    _intercept = Float64List.fromList(
      intercept.map((Object? e) => (e as num).toDouble()).toList(),
    );

    if (coef.isEmpty) {
      throw CharModelException('coef kosong.');
    }
    final List<Object?> firstRow = coef.first as List<Object?>;
    _dims = firstRow.length;

    // Dimensi masuk SVM harus sama dengan keluaran PCA (bila ada) atau
    // vektor gabungan (bila tanpa PCA).
    final int expected = _pcaComponents != null ? _pcaComponentsCount : _combinedDim;
    if (_dims != expected) {
      throw CharModelException(
        'Dimensi fitur model ($_dims) tidak sama dengan yang diharapkan '
        '($expected).',
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

  /// Klasifikasi satu patch karakter.
  CharPrediction predict(GrayImage patch) {
    if (!_loaded) {
      throw CharModelException('Model belum dimuat.');
    }
    return predictFromCombined(combinedFeatures(patch));
  }

  /// Vektor gabungan HOG ++ Zoning untuk satu patch karakter.
  ///
  /// Diekspor agar dapat diuji terpisah: paritas tiap kelompok fitur dengan
  /// Python diperiksa sebelum paritas prediksi akhir.
  Float64List combinedFeatures(GrayImage patch) {
    final List<double> hogVec = hog(patch);
    final List<double> zoneVec = zoning(
      patch,
      grid: _zoningGrid,
      charSize: _charSize,
    );

    if (hogVec.length != _hogDim) {
      throw CharModelException(
        'HOG menghasilkan ${hogVec.length} dimensi, seharusnya $_hogDim.',
      );
    }

    final Float64List out = Float64List(_combinedDim);
    for (int i = 0; i < _hogDim; i++) {
      out[i] = hogVec[i];
    }
    for (int i = 0; i < _zoningDim; i++) {
      out[_hogDim + i] = zoneVec[i];
    }
    return out;
  }

  /// Klasifikasi dari vektor gabungan (340 dimensi, belum diskalakan).
  CharPrediction predictFromCombined(Float64List combined) {
    if (!_loaded) {
      throw CharModelException('Model belum dimuat.');
    }
    if (combined.length != _combinedDim) {
      throw ArgumentError(
        'Panjang fitur ${combined.length} tidak sama dengan $_combinedDim.',
      );
    }

    // Langkah 2: StandardScaler.
    Float64List scaled;
    final Float64List? mean = _scalerMean;
    final Float64List? scale = _scalerScale;
    if (mean != null && scale != null) {
      scaled = Float64List(_combinedDim);
      for (int i = 0; i < _combinedDim; i++) {
        scaled[i] = (combined[i] - mean[i]) / scale[i];
      }
    } else {
      scaled = combined;
    }

    // Langkah 3: PCA (bila model memakainya).
    Float64List features;
    final Float64List? pcaMean = _pcaMean;
    final Float64List? components = _pcaComponents;
    if (pcaMean != null && components != null) {
      features = Float64List(_pcaComponentsCount);
      for (int c = 0; c < _pcaComponentsCount; c++) {
        final int base = c * _combinedDim;
        double sum = 0;
        for (int j = 0; j < _combinedDim; j++) {
          sum += (scaled[j] - pcaMean[j]) * components[base + j];
        }
        features[c] = sum;
      }
    } else {
      features = scaled;
    }

    // Langkah 4: skor linear SVM.
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

  static List<double> _toDoubleList(Object? value, String name) {
    if (value is! List) {
      throw CharModelException('$name harus berupa array.');
    }
    return value.map((Object? e) => (e as num).toDouble()).toList();
  }

  static List<List<double>> _toMatrix(Object? value, String name) {
    if (value is! List) {
      throw CharModelException('$name harus berupa array 2 dimensi.');
    }
    return value
        .map((Object? row) => _toDoubleList(row, name))
        .toList();
  }
}
