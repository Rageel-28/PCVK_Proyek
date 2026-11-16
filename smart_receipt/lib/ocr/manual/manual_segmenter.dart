import 'dart:math' as math;
import 'dart:typed_data';

import '../classical/gray_image.dart';
import 'manual_config.dart';

/// Baris teks hasil deteksi. Koordinat relatif terhadap citra mask.
typedef ManualLine = ({int top, int bottom, int left, int right});

/// Satu glyph hasil segmentasi. Koordinat relatif terhadap potongan baris.
typedef ManualGlyph = ({int top, int left, int height, int width});

/// Pisah citra menjadi baris memakai horizontal projection profile,
/// seperti `detect_lines()` di `annotate_lines.py`.
///
/// Ambang rendah (0,2% lebar) supaya baris dengan sedikit tinta
/// (mis. `Rp` atau garis pemisah) tetap ikut terdeteksi.
List<ManualLine> manualDetectLines(GrayImage mask) {
  final int h = mask.height;
  final int w = mask.width;
  if (h == 0 || w == 0) return const <ManualLine>[];

  final List<int> profile = List<int>.filled(h, 0);
  for (int y = 0; y < h; y++) {
    final int base = y * w;
    int count = 0;
    for (int x = 0; x < w; x++) {
      if (mask.data[base + x] != 0) count++;
    }
    profile[y] = count;
  }

  final int threshold = math.max(2, (w * kManualLineInkRatio).toInt());

  final List<({int top, int bottom})> bands = <({int top, int bottom})>[];
  int? start;
  int gap = 0;
  for (int y = 0; y < h; y++) {
    if (profile[y] > threshold) {
      start ??= y;
      gap = 0;
    } else if (start != null) {
      gap++;
      if (gap >= kManualRowGap) {
        bands.add((top: start, bottom: y - gap));
        start = null;
        gap = 0;
      }
    }
  }
  if (start != null) bands.add((top: start, bottom: h - 1));

  final List<ManualLine> lines = <ManualLine>[];
  for (final ({int top, int bottom}) band in bands) {
    final int height = band.bottom - band.top + 1;
    if (height < kManualMinLineHeight || height > kManualMaxLineHeight) {
      continue;
    }

    int left = w;
    int right = -1;
    for (int y = band.top; y <= band.bottom; y++) {
      final int base = y * w;
      for (int x = 0; x < w; x++) {
        if (mask.data[base + x] != 0) {
          if (x < left) left = x;
          if (x > right) right = x;
        }
      }
    }
    if (right < 0) continue;

    lines.add((
      top: band.top,
      bottom: band.bottom,
      left: left,
      right: right,
    ));
  }
  return lines;
}

/// Kotak kerja yang bisa digabung di tempat.
class _Box {
  _Box({
    required this.top,
    required this.left,
    required this.height,
    required this.width,
    required this.order,
  });

  int top;
  int left;
  int height;
  int width;

  /// Urutan asli, dipakai agar pengurutan stabil seperti Python.
  final int order;

  int get right => left + width;
  int get bottom => top + height;
}

/// Pisah satu baris menjadi glyph memakai connected components
/// 4-konektivitas, seperti `scipy.ndimage.label` versi bawaan.
///
/// Catatan: garis tabel TIDAK dibuang di sini; penghapusan dilakukan
/// sekali pada citra penuh oleh [removeRules].
List<ManualGlyph> manualSegmentGlyphs(
  GrayImage lineMask, {
  int minWidth = kManualGlyphMinWidth,
}) {
  final int w = lineMask.width;
  final int h = lineMask.height;
  if (w == 0 || h == 0) return const <ManualGlyph>[];

  final Int32List labels = Int32List(w * h);
  final List<int> stack = <int>[];
  final List<_Box> boxes = <_Box>[];

  int current = 0;
  for (int i = 0; i < labels.length; i++) {
    if (lineMask.data[i] == 0 || labels[i] != 0) continue;
    current++;

    int minX = w;
    int maxX = -1;
    int minY = h;
    int maxY = -1;

    labels[i] = current;
    stack.add(i);
    while (stack.isNotEmpty) {
      final int p = stack.removeLast();
      final int y = p ~/ w;
      final int x = p - y * w;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;

      if (x > 0) {
        final int q = p - 1;
        if (labels[q] == 0 && lineMask.data[q] != 0) {
          labels[q] = current;
          stack.add(q);
        }
      }
      if (x < w - 1) {
        final int q = p + 1;
        if (labels[q] == 0 && lineMask.data[q] != 0) {
          labels[q] = current;
          stack.add(q);
        }
      }
      if (y > 0) {
        final int q = p - w;
        if (labels[q] == 0 && lineMask.data[q] != 0) {
          labels[q] = current;
          stack.add(q);
        }
      }
      if (y < h - 1) {
        final int q = p + w;
        if (labels[q] == 0 && lineMask.data[q] != 0) {
          labels[q] = current;
          stack.add(q);
        }
      }
    }

    final int bw = maxX - minX + 1;
    final int bh = maxY - minY + 1;
    if (bw >= minWidth) {
      boxes.add(_Box(
        top: minY,
        left: minX,
        height: bh,
        width: bw,
        order: boxes.length,
      ));
    }
  }

  final List<_Box> merged = _mergeStacked(boxes);
  return <ManualGlyph>[
    for (final _Box b in merged)
      (top: b.top, left: b.left, height: b.height, width: b.width),
  ];
}

List<_Box> _sortByLeft(List<_Box> boxes) {
  // Python `sorted()` stabil; List.sort Dart tidak, jadi urutan asli
  // dijadikan kunci kedua agar hasil gabungan deterministik.
  final List<_Box> sorted = List<_Box>.of(boxes)
    ..sort((_Box a, _Box b) {
      final int byLeft = a.left.compareTo(b.left);
      return byLeft != 0 ? byLeft : a.order.compareTo(b.order);
    });
  return sorted;
}

/// Gabungkan komponen yang sebenarnya bagian dari satu huruf.
///
/// Titik pada `i`/`j` adalah komponen terpisah dari batangnya. Tanpa
/// penggabungan, satu huruf terhitung dua komponen sehingga jumlah
/// glyph tidak cocok dengan jumlah karakter — pada pipeline pelatihan
/// ini membuat SELURUH kata dibuang.
///
/// Dua komponen digabung bila rentang horizontalnya bertumpuk >= 50%
/// lebar yang lebih kecil dan jarak vertikalnya <= 25% tinggi gabungan.
List<_Box> _mergeStacked(List<_Box> boxes) {
  if (boxes.length < 2) return boxes;

  List<_Box> merged = _sortByLeft(boxes);
  bool changed = true;
  while (changed) {
    changed = false;
    final List<_Box> out = <_Box>[];
    for (final _Box box in merged) {
      bool joined = false;
      for (int i = 0; i < out.length; i++) {
        final _Box other = out[i];

        final int overlap =
            math.min(box.right, other.right) - math.max(box.left, other.left);
        if (overlap <= 0) continue;

        final int smallerWidth = math.min(box.width, other.width);
        if (smallerWidth <= 0 || overlap / smallerWidth < 0.5) continue;

        final int gap = math.max(box.top, other.top) - math.min(box.bottom, other.bottom);
        final int unionHeight = math.max(box.bottom, other.bottom) - math.min(box.top, other.top);
        if (gap > unionHeight * 0.25) continue;

        final int newTop = math.min(box.top, other.top);
        final int newLeft = math.min(box.left, other.left);
        final int newBottom = math.max(box.bottom, other.bottom);
        final int newRight = math.max(box.right, other.right);
        out[i] = _Box(
          top: newTop,
          left: newLeft,
          height: newBottom - newTop,
          width: newRight - newLeft,
          order: math.min(box.order, other.order),
        );
        joined = true;
        changed = true;
        break;
      }
      if (!joined) out.add(box);
    }
    merged = out;
  }
  return merged;
}
