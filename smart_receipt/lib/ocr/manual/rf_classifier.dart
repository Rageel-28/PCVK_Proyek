import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;

import 'manual_config.dart';

/// Dilempar ketika aset model karakter tidak dapat dibaca atau rusak.
class RandomForestModelException implements Exception {
  RandomForestModelException(this.message);

  final String message;

  @override
  String toString() => 'RandomForestModelException: $message';
}

/// Hasil klasifikasi satu glyph.
class ManualCharPrediction {
  const ManualCharPrediction({
    required this.char,
    required this.confidence,
    required this.margin,
    required this.votes,
  });

  final String char;

  /// Fraksi pohon yang memilih kelas ini (0..1).
  final double confidence;

  /// Selisih fraksi suara juara 1 dan 2. Semakin besar semakin yakin.
  final double margin;

  /// Jumlah pohon yang memilih kelas ini.
  final int votes;

  bool get isConfident => confidence >= kManualConfidenceThreshold;
}

/// Random Forest 100 pohon yang dieksekusi manual di Dart.
///
/// Model dilatih di Python (`ML/Percobaan OCR Manual/tools/train_all_rf.py`)
/// lalu diekspor ke biner oleh `ML/manual_ocr_export/export_manual_assets.py`.
///
/// Inferensi hanya berupa traversal pohon:
///
///   node = 0
///   while bukan daun:
///       node = (x[feature[node]] <= threshold[node]) ? left : right
///   suara[kelas[node]]++
///
/// Tidak memakai library machine learning apa pun di sisi aplikasi.
class RandomForestClassifier {
  RandomForestClassifier();

  bool _loaded = false;

  late Uint8List _feature;
  late Float32List _threshold;
  late Int32List _left;
  late Int32List _right;
  late Uint8List _value;
  late Int32List _treeStart;

  List<String> _classes = const <String>[];
  int _nFeatures = 0;
  int _nTrees = 0;

  bool get isLoaded => _loaded;
  int get classCount => _classes.length;
  int get nFeatures => _nFeatures;
  int get treeCount => _nTrees;
  List<String> get classes => List<String>.unmodifiable(_classes);

  /// Muat model dari aset aplikasi. Aman dipanggil berulang.
  Future<void> load({String assetPath = kManualRfAsset}) async {
    if (_loaded) return;
    final ByteData data;
    try {
      data = await rootBundle.load(assetPath);
    } catch (e) {
      throw RandomForestModelException('Gagal membaca model "$assetPath": $e');
    }
    loadFromBytes(
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
    );
  }

  /// Muat model dari buffer (dipakai unit test).
  void loadFromBytes(Uint8List bytes) {
    if (bytes.length < 16) {
      throw RandomForestModelException('Berkas model terlalu pendek.');
    }
    final ByteData view =
        ByteData.view(bytes.buffer, bytes.offsetInBytes, bytes.lengthInBytes);

    final String magic = String.fromCharCodes(bytes.sublist(0, 4));
    if (magic != kManualRfMagic) {
      throw RandomForestModelException('Magic model tidak dikenal: "$magic".');
    }

    final int nTrees = view.getUint32(4, Endian.little);
    final int nClasses = view.getUint16(8, Endian.little);
    final int nFeatures = view.getUint16(10, Endian.little);

    // Tabel kelas: u8 panjang + byte label.
    int pos = 12;
    final List<String> classes = <String>[];
    for (int c = 0; c < nClasses; c++) {
      final int length = bytes[pos++];
      // Label kelas bisa berupa simbol non-ASCII (¢, ¥, ¬), jadi didekode
      // sebagai UTF-8, bukan dipetakan byte-per-byte.
      classes.add(utf8.decode(bytes.sublist(pos, pos + length)));
      pos += length;
    }

    // Lintasan pertama: hitung total node supaya array bisa dialokasikan
    // sekali (831.562 node pada model produksi — hindari list growable).
    final int nodesOffset = pos;
    int total = 0;
    for (int t = 0; t < nTrees; t++) {
      if (pos + 4 > bytes.length) {
        throw RandomForestModelException('Berkas model terpotong.');
      }
      final int count = view.getUint32(pos, Endian.little);
      total += count;
      pos += 4 + count * 14;
      if (pos > bytes.length) {
        throw RandomForestModelException('Berkas model terpotong.');
      }
    }

    final Uint8List feature = Uint8List(total);
    final Float32List threshold = Float32List(total);
    final Int32List left = Int32List(total);
    final Int32List right = Int32List(total);
    final Uint8List value = Uint8List(total);
    final Int32List treeStart = Int32List(nTrees + 1);

    pos = nodesOffset;
    int index = 0;
    for (int t = 0; t < nTrees; t++) {
      final int count = view.getUint32(pos, Endian.little);
      pos += 4;
      treeStart[t] = index;
      final int treeBase = index;
      for (int i = 0; i < count; i++) {
        feature[index] = bytes[pos];
        threshold[index] = view.getFloat32(pos + 1, Endian.little);
        // Indeks anak di berkas bersifat RELATIF terhadap awal pohon, jadi
        // harus digeser ke posisi global di array gabungan.
        final int rawLeft = view.getInt32(pos + 5, Endian.little);
        final int rawRight = view.getInt32(pos + 9, Endian.little);
        left[index] = rawLeft < 0 ? -1 : treeBase + rawLeft;
        right[index] = rawRight < 0 ? -1 : treeBase + rawRight;
        value[index] = bytes[pos + 13];
        pos += 14;
        index++;
      }
    }
    treeStart[nTrees] = index;

    _feature = feature;
    _threshold = threshold;
    _left = left;
    _right = right;
    _value = value;
    _treeStart = treeStart;
    _classes = classes;
    _nFeatures = nFeatures;
    _nTrees = nTrees;
    _loaded = true;
  }

  /// Klasifikasikan satu vektor fitur (panjang [nFeatures]).
  ManualCharPrediction predict(Float64List vector) {
    if (!_loaded) {
      throw RandomForestModelException('Model belum dimuat.');
    }
    if (vector.length != _nFeatures) {
      throw ArgumentError(
        'Vektor fitur harus $nFeatures dimensi, dapat ${vector.length}.',
      );
    }

    final Int32List votes = Int32List(_classes.length);
    for (int t = 0; t < _nTrees; t++) {
      int node = _treeStart[t];
      while (_feature[node] != kManualLeafSentinel) {
        final int f = _feature[node];
        node = vector[f] <= _threshold[node] ? _left[node] : _right[node];
      }
      votes[_value[node]]++;
    }

    int best = 0;
    for (int i = 1; i < votes.length; i++) {
      if (votes[i] > votes[best]) best = i;
    }
    int second = -1;
    for (int i = 0; i < votes.length; i++) {
      if (i == best) continue;
      if (second < 0 || votes[i] > votes[second]) second = i;
    }

    final int secondVotes = second < 0 ? 0 : votes[second];
    final double confidence = votes[best] / _nTrees;
    final double margin = (votes[best] - secondVotes) / _nTrees;

    return ManualCharPrediction(
      char: _classes[best],
      confidence: confidence,
      margin: margin,
      votes: votes[best],
    );
  }
}
