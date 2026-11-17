import 'dart:math' as math;
import 'dart:typed_data';

import 'classical_config.dart';
import 'connected_components.dart';
import 'gray_image.dart';

/// Konvensi seluruh pipeline: `255 = tinta`, `0 = kertas`.
///
/// Konvensi ini membuat "kepadatan tinta" intuitif dan memudahkan
/// perhitungan proyeksi baris/karakter.

/// Median blur 3x3 (radius 1). Menghapus noise titik tanpa mengaburkan
/// tepi huruf. Median dipilih karena tidak terpengaruh nilai ekstrem.
GrayImage medianBlur3(GrayImage src) {
  final int w = src.width;
  final int h = src.height;
  if (w < 3 || h < 3) {
    return src.clone();
  }
  final GrayImage out = GrayImage.zeros(w, h);
  final List<int> window = List<int>.filled(9, 0);

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      int n = 0;
      for (int dy = -1; dy <= 1; dy++) {
        final int ny = (y + dy).clamp(0, h - 1);
        for (int dx = -1; dx <= 1; dx++) {
          final int nx = (x + dx).clamp(0, w - 1);
          window[n++] = src.data[ny * w + nx];
        }
      }
      window.sort();
      out.data[y * w + x] = window[4];
    }
  }
  return out;
}

/// Ambang Otsu global untuk citra grayscale.
///
/// Mencari nilai ambang yang memaksimalkan variansi antar dua kelas
/// (tinta dan kertas). Dihitung manual dari histogram 256 bin.
int otsuThreshold(List<int> histogram, int total) {
  if (total <= 0) {
    return 128;
  }
  double sumAll = 0;
  for (int i = 0; i < 256; i++) {
    sumAll += i * histogram[i];
  }

  double weightBg = 0;
  double sumBg = 0;
  double bestVariance = -1;
  int bestThreshold = 0;

  for (int t = 0; t < 256; t++) {
    weightBg += histogram[t];
    if (weightBg == 0) continue;
    final double weightFg = total - weightBg;
    if (weightFg == 0) break;

    sumBg += t * histogram[t];
    final double meanBg = sumBg / weightBg;
    final double meanFg = (sumAll - sumBg) / weightFg;
    final double variance =
        weightBg * weightFg * (meanBg - meanFg) * (meanBg - meanFg);

    if (variance > bestVariance) {
      bestVariance = variance;
      bestThreshold = t;
    }
  }
  return bestThreshold;
}

/// Otsu global pada citra grayscale -> mask tinta.
GrayImage binarizeOtsu(GrayImage gray) {
  final List<int> hist = gray.histogram();
  final int threshold = otsuThreshold(hist, gray.length);
  final GrayImage out = GrayImage.zeros(gray.width, gray.height);
  for (int i = 0; i < gray.length; i++) {
    // Tinta = lebih gelap dari ambang (inversi).
    out.data[i] = gray.data[i] <= threshold ? 255 : 0;
  }
  return out;
}

/// Integral image dari citra yang sudah di-pad dengan replikasi tepi.
///
/// OpenCV `boxFilter` memakai `BORDER_REPLICATE`: piksel di luar batas
/// diulang dari piksel tepi, dan rata-rata tetap dibagi luas jendela
/// penuh (win x win). Tanpa padding ini, piksel di tepi memakai pembagi
/// lebih kecil sehingga ambang lokalnya berbeda dan binarisasi menyimpang
/// dari versi Python.
class _PaddedIntegral {
  _PaddedIntegral(GrayImage src, int pad)
      : pad = pad,
        width = src.width,
        height = src.height,
        stride = src.width + 2 * pad + 1 {
    final int rows = src.height + 2 * pad + 1;
    integral = Int64List(stride * rows);
    integralSq = Int64List(stride * rows);

    for (int py = 0; py < src.height + 2 * pad; py++) {
      final int sy = (py - pad).clamp(0, src.height - 1);
      int rowSum = 0;
      int rowSumSq = 0;
      final int srcBase = sy * src.width;
      final int curBase = (py + 1) * stride;
      final int prevBase = py * stride;
      for (int px = 0; px < src.width + 2 * pad; px++) {
        final int sx = (px - pad).clamp(0, src.width - 1);
        final int v = src.data[srcBase + sx];
        rowSum += v;
        rowSumSq += v * v;
        integral[curBase + px + 1] = integral[prevBase + px + 1] + rowSum;
        integralSq[curBase + px + 1] = integralSq[prevBase + px + 1] + rowSumSq;
      }
    }
  }

  final int pad;
  final int width;
  final int height;
  final int stride;
  late final Int64List integral;
  late final Int64List integralSq;

  /// Jumlah nilai pada jendela [x0,x1) x [y0,y1) dalam koordinat asli.
  int sum(int x0, int y0, int x1, int y1) {
    final int px0 = x0 + pad;
    final int py0 = y0 + pad;
    final int px1 = x1 + pad;
    final int py1 = y1 + pad;
    return integral[py1 * stride + px1] -
        integral[py0 * stride + px1] -
        integral[py1 * stride + px0] +
        integral[py0 * stride + px0];
  }

  int sumSq(int x0, int y0, int x1, int y1) {
    final int px0 = x0 + pad;
    final int py0 = y0 + pad;
    final int px1 = x1 + pad;
    final int py1 = y1 + pad;
    return integralSq[py1 * stride + px1] -
        integralSq[py0 * stride + px1] -
        integralSq[py1 * stride + px0] +
        integralSq[py0 * stride + px0];
  }
}

/// Sauvola: ambang lokal berbasis rata-rata dan simpangan baku lokal.
///
///   T(x,y) = m(x,y) * (1 + k * (s(x,y)/R - 1))
///
/// Lebih baik daripada Otsu pada struk karena kertas struk sering
/// memiliki bayangan dan noda dengan intensitas berbeda-beda.
GrayImage binarizeSauvola(
  GrayImage gray, {
  int window = kSauvolaWindow,
  double k = kSauvolaK,
  double r = kSauvolaR,
}) {
  final int w = gray.width;
  final int h = gray.height;
  int win = window;
  if (win % 2 == 0) win++;
  final int half = win ~/ 2;

  final _PaddedIntegral acc = _PaddedIntegral(gray, half);
  final GrayImage out = GrayImage.zeros(w, h);
  final double area = (win * win).toDouble();

  for (int y = 0; y < h; y++) {
    final int y0 = y - half;
    final int y1 = y0 + win;
    for (int x = 0; x < w; x++) {
      final int x0 = x - half;
      final int x1 = x0 + win;

      final double mean = acc.sum(x0, y0, x1, y1) / area;
      final double variance = (acc.sumSq(x0, y0, x1, y1) / area) - (mean * mean);
      final double std = variance > 0 ? math.sqrt(variance) : 0.0;

      final double threshold = mean * (1.0 + k * (std / r - 1.0));
      out.data[y * w + x] = gray.data[y * w + x] < threshold ? 255 : 0;
    }
  }
  return out;
}

/// Adaptive mean threshold: tahan terhadap bayangan bertahap.
GrayImage binarizeAdaptiveMean(
  GrayImage gray, {
  int block = 31,
  int c = 10,
}) {
  final int w = gray.width;
  final int h = gray.height;
  int size = block;
  if (size % 2 == 0) size++;
  final int half = size ~/ 2;

  final _PaddedIntegral acc = _PaddedIntegral(gray, half);
  final GrayImage out = GrayImage.zeros(w, h);
  final double area = (size * size).toDouble();

  for (int y = 0; y < h; y++) {
    final int y0 = y - half;
    final int y1 = y0 + size;
    for (int x = 0; x < w; x++) {
      final int x0 = x - half;
      final int x1 = x0 + size;
      final double mean = acc.sum(x0, y0, x1, y1) / area;
      out.data[y * w + x] = gray.data[y * w + x] < mean - c ? 255 : 0;
    }
  }
  return out;
}

/// Pilihan metode binarisasi.
enum BinarizationMethod { otsu, sauvola, adaptiveMean }

GrayImage binarize(GrayImage gray, BinarizationMethod method) {
  switch (method) {
    case BinarizationMethod.otsu:
      return binarizeOtsu(gray);
    case BinarizationMethod.sauvola:
      return binarizeSauvola(gray);
    case BinarizationMethod.adaptiveMean:
      return binarizeAdaptiveMean(gray);
  }
}

/// Ratakan bayangan dan pencahayaan tak merata sebelum deteksi kertas / Sauvola.
///
/// Hanya diterapkan pada citra dokumen utuh (`min(h, w) >= 200`) dengan
/// max-filter separabel + jumlah kotak padded integer (paritas penuh dengan Python).
GrayImage normalizeIllumination(GrayImage gray) {
  final int h = gray.height;
  final int w = gray.width;
  if (math.min(h, w) < 200) {
    return gray;
  }
  final int rawK = (math.min(h, w) ~/ 25) | 1;
  final int k = math.min(51, math.max(15, rawK));
  final int half = k ~/ 2;

  final GrayImage hmax = GrayImage.zeros(w, h);
  final List<int> dq = List<int>.filled(w + k, 0);

  // 1D sliding maximum horizontal.
  for (int y = 0; y < h; y++) {
    final int rowBase = y * w;
    int head = 0;
    int tail = 0;
    for (int i = 0; i < w + 2 * half; i++) {
      final int sx = (i - half).clamp(0, w - 1);
      final int val = gray.data[rowBase + sx];
      while (tail > head) {
        final int lastI = dq[tail - 1];
        final int lastX = (lastI - half).clamp(0, w - 1);
        if (gray.data[rowBase + lastX] <= val) {
          tail--;
        } else {
          break;
        }
      }
      dq[tail++] = i;
      if (dq[head] <= i - k) {
        head++;
      }
      if (i >= k - 1) {
        final int topX = (dq[head] - half).clamp(0, w - 1);
        hmax.data[rowBase + (i - k + 1)] = gray.data[rowBase + topX];
      }
    }
  }

  // 1D sliding maximum vertikal.
  final GrayImage bg = GrayImage.zeros(w, h);
  final List<int> vdq = List<int>.filled(h + k, 0);
  for (int x = 0; x < w; x++) {
    int head = 0;
    int tail = 0;
    for (int i = 0; i < h + 2 * half; i++) {
      final int sy = (i - half).clamp(0, h - 1);
      final int val = hmax.data[sy * w + x];
      while (tail > head) {
        final int lastI = vdq[tail - 1];
        final int lastY = (lastI - half).clamp(0, h - 1);
        if (hmax.data[lastY * w + x] <= val) {
          tail--;
        } else {
          break;
        }
      }
      vdq[tail++] = i;
      if (vdq[head] <= i - k) {
        head++;
      }
      if (i >= k - 1) {
        final int topY = (vdq[head] - half).clamp(0, h - 1);
        bg.data[(i - k + 1) * w + x] = hmax.data[topY * w + x];
      }
    }
  }

  final _PaddedIntegral acc = _PaddedIntegral(bg, half);
  final int area = k * k;
  final GrayImage out = GrayImage.zeros(w, h);
  for (int y = 0; y < h; y++) {
    final int y0 = y - half;
    final int y1 = y0 + k;
    final int rowBase = y * w;
    for (int x = 0; x < w; x++) {
      final int bsum = math.max(1, acc.sum(x - half, y0, x - half + k, y1));
      final int v = (gray.data[rowBase + x] * area * 255) ~/ bsum;
      out.data[rowBase + x] = v > 255 ? 255 : v;
    }
  }
  return out;
}

/// Rangkaian awal: median blur -> binarisasi -> buang bintik kecil.
GrayImage prepareInk(
  GrayImage gray, {
  BinarizationMethod method = BinarizationMethod.sauvola,
  int minInkArea = kCharMinInk,
}) {
  final GrayImage denoised = medianBlur3(gray);
  final GrayImage ink = binarize(denoised, method);
  return removeSmallComponents(ink, minInkArea);
}
