import 'dart:math' as math;
import 'dart:typed_data';

import '../classical/gray_image.dart';
import 'manual_config.dart';
import 'pil_resize.dart';

/// Potong area bertinta saja (seperti `trim()` di pipeline Python).
GrayImage _trim(GrayImage mask) {
  int minX = mask.width;
  int maxX = -1;
  int minY = mask.height;
  int maxY = -1;

  for (int y = 0; y < mask.height; y++) {
    final int base = y * mask.width;
    for (int x = 0; x < mask.width; x++) {
      if (mask.data[base + x] != 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return mask;
  return mask.crop(minX, minY, maxX, maxY);
}

void _binMeans(List<double> values, int bins, Float64List out, int offset) {
  if (values.isEmpty) {
    for (int i = 0; i < bins; i++) {
      out[offset + i] = 0.0;
    }
    return;
  }
  final int length = values.length;
  for (int i = 0; i < bins; i++) {
    final int lo = i * length ~/ bins;
    final int hi = math.max(lo + 1, (i + 1) * length ~/ bins);
    double sum = 0;
    int count = 0;
    for (int k = lo; k < hi && k < length; k++) {
      sum += values[k];
      count++;
    }
    out[offset + i] = count == 0 ? 0.0 : sum / count;
  }
}

/// Vektor fitur 172 dimensi untuk satu glyph, identik dengan `feat_g12()`
/// di `ML/Percobaan OCR Manual/tools/rebuild_feature_cache.py`.
///
/// Kelompok fitur:
///   - grid 12x12 (144): rata-rata intensitas per sel — distribusi tinta
///   - kontur atas & bawah, 12 bin masing-masing (24): tinggi huruf dan ekor
///   - bentuk global (4): rasio aspek kanvas, kepadatan, transisi vertikal
///
/// Glyph dinormalkan ke 64x48 dengan rasio aspek TIDAK dijaga (stretch),
/// karena model dilatih dengan cara itu. Crop diambil dari mask biner,
/// bukan dari grayscale.
Float64List manualFeatures(GrayImage glyphMask) {
  final GrayImage trimmed = _trim(glyphMask);
  final GrayImage resized = pilResize(
    trimmed,
    kManualCharWidth,
    kManualCharHeight,
    PilFilter.bilinear,
  );

  const int w = kManualCharWidth;
  const int h = kManualCharHeight;

  final List<double> values = List<double>.filled(w * h, 0);
  final Uint8List binary = Uint8List(w * h);
  for (int i = 0; i < values.length; i++) {
    final int v = resized.data[i];
    values[i] = v / 255.0;
    binary[i] = v > 127 ? 1 : 0; // > 0.5 pada skala 0..255
  }

  final Float64List out = Float64List(kManualFeatureDim);
  int offset = 0;

  // 1. Grid 12x12.
  const int grid = 12;
  for (int r = 0; r < grid; r++) {
    final int y0 = r * h ~/ grid;
    final int y1 = (r + 1) * h ~/ grid;
    for (int c = 0; c < grid; c++) {
      final int x0 = c * w ~/ grid;
      final int x1 = (c + 1) * w ~/ grid;
      double sum = 0;
      int count = 0;
      for (int y = y0; y < y1; y++) {
        final int base = y * w;
        for (int x = x0; x < x1; x++) {
          sum += values[base + x];
          count++;
        }
      }
      out[offset++] = count == 0 ? 0.0 : sum / count;
    }
  }

  // 2. Kontur atas & bawah per kolom, diringkas 12 bin.
  final List<double> topProfile = List<double>.filled(w, 0);
  final List<double> bottomProfile = List<double>.filled(w, 0);
  for (int x = 0; x < w; x++) {
    int first = -1;
    int last = -1;
    for (int y = 0; y < h; y++) {
      if (binary[y * w + x] != 0) {
        if (first < 0) first = y;
        last = y;
      }
    }
    if (first < 0) {
      topProfile[x] = 1.0;
      bottomProfile[x] = 0.0;
    } else {
      topProfile[x] = first / h;
      bottomProfile[x] = 1.0 - last / h;
    }
  }
  _binMeans(topProfile, 12, out, offset);
  offset += 12;
  _binMeans(bottomProfile, 12, out, offset);
  offset += 12;

  // 3. Bentuk global.
  double valueSum = 0;
  for (final double v in values) {
    valueSum += v;
  }
  final double mean = valueSum / values.length;

  final List<int> transitions = List<int>.filled(w, 0);
  for (int x = 0; x < w; x++) {
    int count = 0;
    for (int y = 1; y < h; y++) {
      if (binary[y * w + x] != binary[(y - 1) * w + x]) count++;
    }
    transitions[x] = count;
  }
  double transitionSum = 0;
  int transitionCount = 0;
  for (final int v in transitions) {
    transitionSum += v;
    if (v > 0) transitionCount++;
  }

  out[offset++] = math.min(w / math.max(h, 1), 2.0) / 2.0;
  out[offset++] = mean;
  out[offset++] = transitionSum / w;
  out[offset++] = transitionCount / w;

  return out;
}
