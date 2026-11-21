/// Normalisasi crop glyph ke kanvas 32x32.
///
/// Port dari `normalize_char` di `ML/ocr/render_dataset.py`. Data latih
/// dirender dengan fungsi yang sama, jadi inferensi harus identik.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'gray_image.dart';
import 'params.dart';
import 'preprocess.dart' show resizeArea;

/// Skalakan crop glyph ke kanvas 32x32, rasio aspek dijaga, lalu geser agar
/// pusat massa tinta tepat di tengah kanvas.
Uint8List normalizeChar(BinaryImage crop) {
  final Uint8List canvas = Uint8List(P.charSize * P.charSize);
  if (crop.width == 0 || crop.height == 0) return canvas;

  const int size = P.charSize;
  const int inner = size - 2 * P.pad;

  final double scale = inner / math.max(crop.width, crop.height);
  int newW = math.max(1, (crop.width * scale).round());
  int newH = math.max(1, (crop.height * scale).round());
  newW = math.min(newW, inner);
  newH = math.min(newH, inner);

  // Crop adalah citra biner; bungkus sebagai GrayImage agar bisa memakai
  // resizeArea yang sama dengan praproses.
  final GrayImage source = GrayImage(crop.width, crop.height, crop.data);
  final GrayImage resized = resizeArea(source, newW, newH);

  final int x0 = (size - newW) ~/ 2;
  final int y0 = (size - newH) ~/ 2;
  for (int y = 0; y < newH; y++) {
    final int dstRow = (y0 + y) * size + x0;
    canvas.setRange(dstRow, dstRow + newW, resized.data, y * newW);
  }

  // Geser agar pusat massa tinta tepat di tengah. Tanpa ini, fitur
  // proyeksi/zoning sangat sensitif terhadap jitter bbox 1-2 piksel.
  double sumX = 0;
  double sumY = 0;
  int count = 0;
  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      if (canvas[y * size + x] != 0) {
        sumX += x;
        sumY += y;
        count++;
      }
    }
  }
  if (count == 0) return canvas;

  final double target = (size - 1) / 2.0;
  final double dx = target - sumX / count;
  final double dy = target - sumY / count;
  if (dx.abs() <= 0.05 && dy.abs() <= 0.05) return canvas;

  final Uint8List shifted = Uint8List(size * size);
  for (int y = 0; y < size; y++) {
    final double sy = y - dy;
    for (int x = 0; x < size; x++) {
      final double sx = x - dx;
      shifted[y * size + x] = _bilinearZero(canvas, size, sx, sy);
    }
  }
  return shifted;
}

int _bilinearZero(Uint8List src, int size, double sx, double sy) {
  final int x0 = sx.floor();
  final int y0 = sy.floor();
  final double fx = sx - x0;
  final double fy = sy - y0;
  int sample(int x, int y) {
    if (x < 0 || y < 0 || x >= size || y >= size) return 0;
    return src[y * size + x];
  }

  final int v00 = sample(x0, y0);
  final int v10 = sample(x0 + 1, y0);
  final int v01 = sample(x0, y0 + 1);
  final int v11 = sample(x0 + 1, y0 + 1);
  final double top = v00 * (1 - fx) + v10 * fx;
  final double bottom = v01 * (1 - fx) + v11 * fx;
  return (top * (1 - fy) + bottom * fy).round().clamp(0, 255);
}
