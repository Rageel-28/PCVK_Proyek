/// Deteksi baris teks dari citra biner struk.
///
/// Port dari `ML/ocr/lines.py`: proyeksi horizontal jumlah tinta per baris,
/// ambang relatif lebar citra, pemecahan pita yang terlalu tinggi, dan
/// penggabungan fragmen yang berdekatan.
library;

import 'dart:typed_data';

import 'gray_image.dart';
import 'params.dart';

/// Kotak baris: `bottom` dan `right` eksklusif.
class LineBox {
  const LineBox(this.top, this.bottom, this.left, this.right);

  final int top;
  final int bottom;
  final int left;
  final int right;

  int get height => bottom - top;
  int get width => right - left;

  @override
  String toString() => 'LineBox($top,$bottom,$left,$right)';
}

/// Rentetan nilai true sebagai pasangan (awal, akhir-eksklusif).
List<List<int>> runsOf(List<bool> mask) {
  final List<List<int>> out = <List<int>>[];
  int? start;
  for (int i = 0; i < mask.length; i++) {
    if (mask[i] && start == null) {
      start = i;
    } else if (!mask[i] && start != null) {
      out.add(<int>[start, i]);
      start = null;
    }
  }
  if (start != null) out.add(<int>[start, mask.length]);
  return out;
}

/// Tinggi baris teks acuan, tahan terhadap garis dan pita tergabung.
///
/// Buang pencilan jauh dengan patokan persentil ke-75, lalu ambil median
/// separuh bawah dari sisanya.
double referenceHeight(List<int> heights) {
  if (heights.isEmpty) return 0.0;
  final List<int> sorted = List<int>.from(heights)..sort();
  final double high = _percentile(sorted, 75);
  if (high <= 0) return 0.0;

  final List<int> kept = sorted
      .where((int h) => h >= 0.5 * high && h <= 1.5 * high)
      .toList();
  if (kept.isEmpty) return high;
  final double mid = _percentile(kept, 50);
  final List<int> lower = kept.where((int h) => h <= mid).toList();
  return _percentile(lower, 50);
}

double _percentile(List<int> sorted, double p) {
  if (sorted.isEmpty) return 0.0;
  final double index = (sorted.length - 1) * p / 100.0;
  final int lo = index.floor();
  final int hi = index.ceil();
  final double frac = index - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

/// Pecah pita yang jauh lebih tinggi dari acuan di lembah terdalamnya.
List<LineBox> splitTallBands(
  List<LineBox> bands,
  Int32List profile,
  double medianHeight,
) {
  if (medianHeight <= 0) return bands;
  final double limit = medianHeight * P.lineSplitRatio;

  final List<LineBox> out = <LineBox>[];
  for (final LineBox band in bands) {
    if (band.height <= limit) {
      out.add(band);
      continue;
    }
    final int segmentSize = band.bottom - band.top;
    if (segmentSize < 6) {
      out.add(band);
      continue;
    }

    final int margin = segmentSize ~/ 5 < 1 ? 1 : segmentSize ~/ 5;
    final int innerStart = band.top + margin;
    final int innerEnd = band.bottom - margin;
    if (innerEnd <= innerStart) {
      out.add(band);
      continue;
    }

    int cut = innerStart;
    int lowest = profile[innerStart];
    for (int i = innerStart + 1; i < innerEnd; i++) {
      if (profile[i] < lowest) {
        lowest = profile[i];
        cut = i;
      }
    }

    // Median segmen.
    final List<int> segment = <int>[];
    for (int i = band.top; i < band.bottom; i++) {
      segment.add(profile[i]);
    }
    segment.sort();
    final double segmentMedian = _percentile(segment, 50);
    if (lowest > 0.75 * segmentMedian) {
      out.add(band);
      continue;
    }

    final LineBox top = LineBox(band.top, cut, band.left, band.right);
    final LineBox bottom = LineBox(cut, band.bottom, band.left, band.right);
    out.addAll(splitTallBands(<LineBox>[top], profile, medianHeight));
    out.addAll(splitTallBands(<LineBox>[bottom], profile, medianHeight));
  }
  return out;
}

/// Deteksi pita baris teks.
List<LineBox> detectLines(
  BinaryImage binary, {
  int? minInk,
  int? minHeight,
}) {
  if (binary.width == 0 || binary.height == 0) return <LineBox>[];
  if (!binary.hasInk) return <LineBox>[];

  final int h = binary.height;
  final int w = binary.width;
  final int minH = minHeight ?? (h * 0.004).round().clamp(4, 1 << 30);

  final Int32List profile = Int32List(h);
  for (int y = 0; y < h; y++) {
    final int row = y * w;
    int count = 0;
    for (int x = 0; x < w; x++) {
      if (binary.data[row + x] != 0) count++;
    }
    profile[y] = count;
  }

  final int minInkValue = minInk ?? (w * 0.008).round().clamp(2, 1 << 30);

  List<bool> active =
      List<bool>.generate(h, (int y) => profile[y] >= minInkValue);
  active = _fillSmallGaps(active, (minH ~/ 3) < 1 ? 1 : minH ~/ 3);

  final List<LineBox> raw = <LineBox>[];
  for (final List<int> run in runsOf(active)) {
    final int start = run[0];
    final int end = run[1];
    if (end - start < minH) continue;

    int left = w;
    int right = 0;
    for (int y = start; y < end; y++) {
      final int row = y * w;
      for (int x = 0; x < w; x++) {
        if (binary.data[row + x] != 0) {
          if (x < left) left = x;
          if (x + 1 > right) right = x + 1;
        }
      }
    }
    if (right <= left) continue;
    raw.add(LineBox(start, end, left, right));
  }

  if (raw.isEmpty) return <LineBox>[];

  final List<int> heights = raw.map((LineBox b) => b.height).toList();
  final List<int> sorted = List<int>.from(heights)..sort();
  final double medianHeight = _percentile(sorted, 50);
  final double reference = referenceHeight(heights);

  List<LineBox> bands = splitTallBands(raw, profile, reference);
  bands = _mergeClose(
    bands,
    (medianHeight * 0.15).round() < 1 ? 1 : (medianHeight * 0.15).round(),
  );
  return bands.where((LineBox b) => b.height >= minH).toList();
}

List<bool> _fillSmallGaps(List<bool> mask, int maxGap) {
  final List<bool> out = List<bool>.from(mask);
  for (final List<int> gap in runsOf(mask.map((bool v) => !v).toList())) {
    final int start = gap[0];
    final int end = gap[1];
    if (start > 0 && end < mask.length && (end - start) <= maxGap) {
      for (int i = start; i < end; i++) {
        out[i] = true;
      }
    }
  }
  return out;
}

List<LineBox> _mergeClose(List<LineBox> bands, int maxGap) {
  if (bands.isEmpty) return <LineBox>[];
  final List<LineBox> sorted = List<LineBox>.from(bands)
    ..sort((LineBox a, LineBox b) => a.top.compareTo(b.top));
  final List<int> heights = sorted.map((LineBox b) => b.height).toList()..sort();
  final double median = _percentile(heights, 50);

  final List<LineBox> merged = <LineBox>[sorted[0]];
  for (int i = 1; i < sorted.length; i++) {
    final LineBox band = sorted[i];
    final LineBox prev = merged.last;
    final int gap = band.top - prev.bottom;
    final bool isFragment =
        (prev.height < band.height ? prev.height : band.height) < 0.6 * median;
    if (gap <= maxGap && isFragment) {
      merged[merged.length - 1] = LineBox(
        prev.top,
        band.bottom > prev.bottom ? band.bottom : prev.bottom,
        prev.left < band.left ? prev.left : band.left,
        prev.right > band.right ? prev.right : band.right,
      );
    } else {
      merged.add(band);
    }
  }
  return merged;
}
