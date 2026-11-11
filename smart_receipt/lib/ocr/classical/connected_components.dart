import 'dart:typed_data';

import 'gray_image.dart';

/// Satu komponen terhubung pada citra biner.
class InkComponent {
  InkComponent({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
    required this.area,
    required this.label,
  });

  int left;
  int top;
  int right;
  int bottom;
  int area;
  final int label;

  int get width => right - left + 1;
  int get height => bottom - top + 1;
}

/// Hasil pelabelan komponen terhubung.
class LabeledComponents {
  LabeledComponents(this.labels, this.components);

  /// Label per piksel; 0 = latar, 1..n = komponen.
  final Int32List labels;

  /// Daftar komponen, label = indeks + 1.
  final List<InkComponent> components;
}

/// Pelabelan komponen terhubung 8-konektivitas pada mask tinta.
///
/// Mask dianggap biner: nilai > 0 berarti tinta. Flood fill iteratif
/// dengan stack eksplisit (bukan rekursi) agar aman pada citra besar.
LabeledComponents labelComponents(GrayImage mask) {
  final int w = mask.width;
  final int h = mask.height;
  final int total = w * h;

  final Int32List labels = Int32List(total);
  final List<InkComponent> components = <InkComponent>[];
  final List<int> stack = <int>[];
  final List<int> pixels = <int>[];

  int nextLabel = 1;

  for (int start = 0; start < total; start++) {
    if (labels[start] != 0 || mask.data[start] == 0) {
      continue;
    }

    final int label = nextLabel++;
    pixels.clear();
    stack.clear();
    labels[start] = label;
    stack.add(start);

    int minX = w, minY = h, maxX = -1, maxY = -1;

    while (stack.isNotEmpty) {
      final int index = stack.removeLast();
      pixels.add(index);
      final int cx = index % w;
      final int cy = index ~/ w;

      if (cx < minX) minX = cx;
      if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy;
      if (cy > maxY) maxY = cy;

      for (int dy = -1; dy <= 1; dy++) {
        final int ny = cy + dy;
        if (ny < 0 || ny >= h) continue;
        final int rowBase = ny * w;
        for (int dx = -1; dx <= 1; dx++) {
          if (dx == 0 && dy == 0) continue;
          final int nx = cx + dx;
          if (nx < 0 || nx >= w) continue;
          final int n = rowBase + nx;
          if (labels[n] != 0 || mask.data[n] == 0) continue;
          labels[n] = label;
          stack.add(n);
        }
      }
    }

    components.add(
      InkComponent(
        left: minX,
        top: minY,
        right: maxX,
        bottom: maxY,
        area: pixels.length,
        label: label,
      ),
    );
  }

  return LabeledComponents(labels, components);
}

/// Komponen dengan `area >= minArea`, diurutkan kiri ke kanan.
///
/// Urutan horizontal dipakai agar pembacaan mengikuti arah baca.
List<InkComponent> connectedComponents(GrayImage mask, {int minArea = 1}) {
  final LabeledComponents result = labelComponents(mask);
  final List<InkComponent> kept = result.components
      .where((InkComponent c) => c.area >= minArea)
      .toList()
    ..sort((InkComponent a, InkComponent b) => a.left.compareTo(b.left));
  return kept;
}

/// Hapus komponen yang areanya lebih kecil dari [minArea].
GrayImage removeSmallComponents(GrayImage mask, int minArea) {
  if (minArea <= 1) {
    return mask.clone();
  }
  final LabeledComponents labeled = labelComponents(mask);
  final Uint8List keep = Uint8List(labeled.components.length + 1);
  for (final InkComponent c in labeled.components) {
    if (c.area >= minArea) {
      keep[c.label] = 1;
    }
  }

  final GrayImage out = GrayImage.zeros(mask.width, mask.height);
  for (int i = 0; i < labeled.labels.length; i++) {
    final int label = labeled.labels[i];
    if (label != 0 && keep[label] != 0) {
      out.data[i] = 255;
    }
  }
  return out;
}
