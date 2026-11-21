/// Klasifikasi pita baris: teks vs garis vs barcode.
///
/// Port dari `ML/ocr/rules.py`.
library;

import 'dart:math' as math;

import 'gray_image.dart';
import 'lines.dart';

/// Panjang setiap rentetan true di satu baris.
List<int> _runLengths(List<bool> row) {
  final List<int> lengths = <int>[];
  int current = 0;
  for (final bool value in row) {
    if (value) {
      current++;
    } else if (current > 0) {
      lengths.add(current);
      current = 0;
    }
  }
  if (current > 0) lengths.add(current);
  return lengths;
}

double _median(List<int> values) {
  if (values.isEmpty) return 0.0;
  final List<int> sorted = List<int>.from(values)..sort();
  final int mid = sorted.length ~/ 2;
  if (sorted.length.isOdd) return sorted[mid].toDouble();
  return (sorted[mid - 1] + sorted[mid]) / 2.0;
}

/// True bila pita ini garis horizontal (putus-putus atau padat).
bool isRule(BinaryImage binary, LineBox band, {double medianHeight = 0.0}) {
  final int w = band.width;
  final int h = band.height;
  if (w <= 0 || h <= 0) return false;

  int rowsWithInk = 0;
  int minCol = w;
  int maxCol = -1;
  for (int y = band.top; y < band.bottom; y++) {
    bool rowHasInk = false;
    final int row = y * binary.width;
    for (int x = band.left; x < band.right; x++) {
      if (binary.data[row + x] != 0) {
        rowHasInk = true;
        if (x < minCol) minCol = x;
        if (x > maxCol) maxCol = x;
      }
    }
    if (rowHasInk) rowsWithInk++;
  }
  if (rowsWithInk == 0 || maxCol < 0) return false;

  final double span = (maxCol - minCol + 1) / math.max(1, w);
  if (span <= 0.6) return false;

  // Baris dengan tinta paling banyak.
  int bestRow = band.top;
  int bestCount = -1;
  for (int y = band.top; y < band.bottom; y++) {
    final int row = y * binary.width;
    int count = 0;
    for (int x = band.left; x < band.right; x++) {
      if (binary.data[row + x] != 0) count++;
    }
    if (count > bestCount) {
      bestCount = count;
      bestRow = y;
    }
  }
  final List<bool> rowMask = List<bool>.generate(
    w,
    (int i) => binary.data[bestRow * binary.width + band.left + i] != 0,
  );
  final List<int> runs = _runLengths(rowMask);
  if (runs.isEmpty) return false;

  final bool dashed = runs.length >= 3 && _median(runs) < 0.25 * w;
  final bool solid = runs.reduce(math.max) > 0.5 * w;
  if (!dashed && !solid) return false;

  if (medianHeight > 0) return h <= 0.55 * medianHeight;
  return rowsWithInk <= math.max(6, (h * 0.35).floor());
}

/// True bila pita berisi gambar/logo, bukan teks.
///
/// Logo toko (wordmark, blok abu-abu, kotak promo) menghasilkan rentetan
/// tinta horizontal yang sangat panjang: satu baris piksel bisa menyala
/// sampai setengah lebar pita. Huruf tidak pernah seperti itu — pada baris
/// teks rentetan terpanjang biasanya di bawah 10% lebar, bahkan untuk huruf
/// tebal yang menyatu.
bool isGraphic(BinaryImage binary, LineBox band) {
  final int w = band.width;
  final int h = band.height;
  if (w <= 0 || h <= 0) return false;
  if (w < 40) return false;

  int longest = 0;
  for (int y = band.top; y < band.bottom; y++) {
    final int row = y * binary.width;
    int current = 0;
    for (int x = band.left; x < band.right; x++) {
      if (binary.data[row + x] != 0) {
        current++;
        if (current > longest) longest = current;
      } else {
        current = 0;
      }
    }
  }
  return longest > 0.25 * w;
}

/// True bila pita ini barcode: bar vertikal seragam dan pita sangat tinggi.
bool isBarcode(BinaryImage binary, LineBox band, {double medianHeight = 0.0}) {
  final int w = band.width;
  final int h = band.height;
  if (w <= 0 || h <= 0) return false;
  if (w < 40) return false;
  if (medianHeight > 0 && h < 1.6 * medianHeight) return false;

  final List<double> perRow = <double>[];
  for (int y = band.top; y < band.bottom; y++) {
    final int row = y * binary.width;
    double count = 0;
    for (int x = band.left; x < band.right; x++) {
      if (binary.data[row + x] != 0) count++;
    }
    if (count > 0) perRow.add(count);
  }
  if (perRow.length < 10) return false;

  final double mean = perRow.reduce((double a, double b) => a + b) / perRow.length;
  if (mean <= 0) return false;
  double variance = 0;
  for (final double v in perRow) {
    variance += (v - mean) * (v - mean);
  }
  variance /= perRow.length;
  final double cv = math.sqrt(variance) / mean;
  if (cv > 0.25) return false;

  final int midTop = band.top + h ~/ 4;
  final int midBottom = math.max(midTop + 1, band.top + (3 * h) ~/ 4);
  double transitionTotal = 0;
  int transitionRows = 0;
  for (int y = midTop; y < midBottom; y++) {
    final int row = y * binary.width;
    int transitions = 0;
    bool prev = binary.data[row + band.left] != 0;
    for (int x = band.left + 1; x < band.right; x++) {
      final bool cur = binary.data[row + x] != 0;
      if (cur != prev) transitions++;
      prev = cur;
    }
    transitionTotal += transitions;
    transitionRows++;
  }
  final double meanTransitions =
      transitionRows > 0 ? transitionTotal / transitionRows : 0.0;
  return meanTransitions > 0.05 * w;
}

/// Kembalikan `'rule'`, `'barcode'`, `'graphic'`, atau `'text'`.
String classifyBand(BinaryImage binary, LineBox band, {double medianHeight = 0.0}) {
  if (isRule(binary, band, medianHeight: medianHeight)) return 'rule';
  if (isGraphic(binary, band)) return 'graphic';
  if (isBarcode(binary, band, medianHeight: medianHeight)) return 'barcode';
  return 'text';
}

/// Buang pita yang bukan teks, pertahankan urutan.
List<LineBox> filterBands(
  BinaryImage binary,
  List<LineBox> bands, {
  double medianHeight = 0.0,
}) {
  if (bands.isEmpty) return <LineBox>[];
  double median = medianHeight;
  if (median <= 0) {
    final List<int> heights = bands.map((LineBox b) => b.height).toList()..sort();
    median = _median(heights);
  }
  return bands
      .where((LineBox b) => classifyBand(binary, b, medianHeight: median) == 'text')
      .toList();
}
