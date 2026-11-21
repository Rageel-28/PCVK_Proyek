/// Segmentasi kata di dalam satu pita baris.
///
/// Port dari `ML/ocr/words.py`. Ambang celah kata dinyatakan relatif
/// terhadap lebar sel huruf, bukan terhadap sebaran celah.
library;

import 'dart:math' as math;

import 'gray_image.dart';
import 'params.dart';

/// Kotak kata di koordinat citra penuh.
class Word {
  const Word(this.left, this.top, this.right, this.bottom);

  final int left;
  final int top;
  final int right;
  final int bottom;

  int get width => right - left;
  int get height => bottom - top;

  @override
  String toString() => 'Word($left,$top,$right,$bottom)';
}

double _median(List<int> values) {
  if (values.isEmpty) return 0.0;
  final List<int> sorted = List<int>.from(values)..sort();
  final int mid = sorted.length ~/ 2;
  if (sorted.length.isOdd) return sorted[mid].toDouble();
  return (sorted[mid - 1] + sorted[mid]) / 2.0;
}

/// Ambang celah batas kata dari lebar glyph baris itu.
double wordGapThreshold(List<List<int>> runs) {
  if (runs.isEmpty) return 2.0;
  final List<int> widths =
      runs.map((List<int> r) => r[1] - r[0]).toList();
  final double medianWidth = _median(widths);
  if (medianWidth <= 0) return 2.0;

  final List<int> gaps = <int>[];
  for (int i = 0; i + 1 < runs.length; i++) {
    gaps.add(runs[i + 1][0] - runs[i][1]);
  }
  final double medianGap = gaps.isEmpty ? 0.0 : _median(gaps);
  final double advance = medianWidth + math.max(0.0, medianGap);
  return math.max(2.0, advance * P.wordGapToWidth);
}

/// Pecah pita baris menjadi kata.
List<Word> segmentWords(
  BinaryImage band, {
  int bandTop = 0,
  int bandLeft = 0,
}) {
  final int w = band.width;
  final int h = band.height;
  if (w == 0 || h == 0 || !band.hasInk) return <Word>[];

  // Rentetan kolom bertinta.
  final List<List<int>> runs = <List<int>>[];
  int? start;
  for (int x = 0; x < w; x++) {
    bool hasInk = false;
    for (int y = 0; y < h; y++) {
      if (band.data[y * w + x] != 0) {
        hasInk = true;
        break;
      }
    }
    if (hasInk && start == null) {
      start = x;
    } else if (!hasInk && start != null) {
      runs.add(<int>[start, x]);
      start = null;
    }
  }
  if (start != null) runs.add(<int>[start, w]);
  if (runs.isEmpty) return <Word>[];

  final double threshold = wordGapThreshold(runs);

  int top = h;
  int bottom = 0;
  for (int y = 0; y < h; y++) {
    final int row = y * w;
    for (int x = 0; x < w; x++) {
      if (band.data[row + x] != 0) {
        if (y < top) top = y;
        if (y + 1 > bottom) bottom = y + 1;
        break;
      }
    }
  }

  final List<Word> words = <Word>[];
  int currentLeft = runs[0][0];
  for (int i = 0; i + 1 < runs.length; i++) {
    final List<int> run = runs[i];
    final int nextStart = runs[i + 1][0];
    if (nextStart - run[1] > threshold) {
      words.add(Word(
        bandLeft + currentLeft,
        bandTop + top,
        bandLeft + run[1],
        bandTop + bottom,
      ));
      currentLeft = nextStart;
    }
  }
  words.add(Word(
    bandLeft + currentLeft,
    bandTop + top,
    bandLeft + runs.last[1],
    bandTop + bottom,
  ));
  return words;
}
