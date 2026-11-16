import 'dart:math' as math;
import 'dart:typed_data';

import '../classical/gray_image.dart';

/// Filter resampling yang dipakai pipeline manual.
enum PilFilter { bilinear, lanczos }

double _bilinearWeight(double x) {
  final double a = x.abs();
  return a < 1.0 ? 1.0 - a : 0.0;
}

double _lanczosWeight(double x) {
  final double a = x.abs();
  if (a >= 3.0) return 0.0;
  if (a < 1e-12) return 1.0;
  return (math.sin(math.pi * a) * math.sin(math.pi * a / 3.0)) /
      (math.pi * math.pi * a * a / 3.0);
}

/// Pillow menyimpan koefisien sebagai fixed-point 22 bit
/// (`PRECISION_BITS = 32 - 8 - 2`). Akumulasi memakai integer lalu
/// dibulatkan dengan `+ (1 << 21)` sebelum digeser.
const int _precisionBits = 22;

class _Coefficients {
  const _Coefficients(this.start, this.weights);

  /// Indeks piksel sumber pertama.
  final int start;

  /// Bobot terkuantisasi (skala 2^22).
  final Int32List weights;
}

List<_Coefficients> _precompute(
  int inSize,
  int outSize,
  double Function(double) filter,
  double support,
) {
  final double scale = inSize / outSize;
  final double filterscale = scale < 1.0 ? 1.0 : scale;
  final double supportScaled = support * filterscale;
  final double ss = 1.0 / filterscale;
  final int quantScale = 1 << _precisionBits;

  final List<_Coefficients> out = <_Coefficients>[];
  for (int i = 0; i < outSize; i++) {
    final double center = (i + 0.5) * scale;

    // (int) di C memotong ke arah nol, sama seperti double.toInt().
    int xmin = (center - supportScaled + 0.5).toInt();
    if (xmin < 0) xmin = 0;
    int xmax = (center + supportScaled + 0.5).toInt();
    if (xmax > inSize) xmax = inSize;

    final int n = xmax - xmin;
    final Int32List weights = Int32List(n < 0 ? 0 : n);
    if (n > 0) {
      final List<double> raw = List<double>.filled(n, 0);
      double total = 0.0;
      for (int k = 0; k < n; k++) {
        final double w = filter((k + xmin - center + 0.5) * ss);
        raw[k] = w;
        total += w;
      }
      if (total != 0.0) {
        // Pillow menormalkan dengan mengalikan 1/ww (bukan membagi).
        final double inv = 1.0 / total;
        for (int k = 0; k < n; k++) {
          weights[k] = (raw[k] * inv * quantScale).round();
        }
      }
    }
    out.add(_Coefficients(xmin, weights));
  }
  return out;
}

/// Resize dua lintasan (horizontal lalu vertikal) yang meniru
/// `PIL.Image.resize` untuk citra 8-bit.
///
/// Ditulis ulang, bukan memakai interpolasi bilinear biasa, karena hasil
/// PIL berbeda 1 LSB pada sebagian piksel. Selisih itu cukup untuk
/// menggeser fitur grid 12x12 dan membuat prediksi Random Forest berubah.
GrayImage pilResize(GrayImage src, int outWidth, int outHeight, PilFilter filter) {
  if (outWidth <= 0 || outHeight <= 0) {
    return GrayImage.zeros(math.max(1, outWidth), math.max(1, outHeight));
  }
  if (src.width == outWidth && src.height == outHeight) {
    return src.clone();
  }

  final double Function(double) kernel =
      filter == PilFilter.bilinear ? _bilinearWeight : _lanczosWeight;
  final double support = filter == PilFilter.bilinear ? 1.0 : 3.0;
  final int half = 1 << (_precisionBits - 1);

  // Lintasan horizontal: hasil antara disimpan sebagai uint8, seperti PIL.
  final GrayImage tmp = GrayImage.zeros(outWidth, src.height);
  final List<_Coefficients> horizontal =
      _precompute(src.width, outWidth, kernel, support);
  for (int x = 0; x < outWidth; x++) {
    final _Coefficients c = horizontal[x];
    final int n = c.weights.length;
    for (int y = 0; y < src.height; y++) {
      final int base = y * src.width + c.start;
      int acc = 0;
      for (int k = 0; k < n; k++) {
        acc += c.weights[k] * src.data[base + k];
      }
      int v = (acc + half) >> _precisionBits;
      if (v < 0) {
        v = 0;
      } else if (v > 255) {
        v = 255;
      }
      tmp.data[y * outWidth + x] = v;
    }
  }

  // Lintasan vertikal.
  final GrayImage out = GrayImage.zeros(outWidth, outHeight);
  final List<_Coefficients> vertical =
      _precompute(src.height, outHeight, kernel, support);
  for (int y = 0; y < outHeight; y++) {
    final _Coefficients c = vertical[y];
    final int n = c.weights.length;
    for (int x = 0; x < outWidth; x++) {
      int acc = 0;
      for (int k = 0; k < n; k++) {
        acc += c.weights[k] * tmp.data[(c.start + k) * outWidth + x];
      }
      int v = (acc + half) >> _precisionBits;
      if (v < 0) {
        v = 0;
      } else if (v > 255) {
        v = 255;
      }
      out.data[y * outWidth + x] = v;
    }
  }
  return out;
}
