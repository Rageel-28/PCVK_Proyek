/// Praproses citra struk: grayscale → denoise → binarisasi Sauvola → upscale.
///
/// Port langsung dari `ML/ocr/preprocess.py`. Urutannya penting: kemiringan
/// dikoreksi **sebelum** upscale, lalu upscale dihitung dari tinggi baris
/// hasil binarisasi (bukan ditebak).
///
/// Semua fungsi di sini bekerja pada [GrayImage]/[BinaryImage] milik sendiri
/// supaya tidak bergantung pada paket gambar mana pun.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'gray_image.dart';
import 'params.dart';

/// Hasil praproses lengkap beserta faktor skala yang dipakai.
class Prepared {
  Prepared({
    required this.gray,
    required this.binary,
    required this.scale,
    this.skew = 0.0,
  });

  final GrayImage gray;
  final BinaryImage binary;
  final double scale;
  final double skew;
}

// ---------------------------------------------------------------------------
// Filter dasar
// ---------------------------------------------------------------------------

/// Box filter rata-rata yang dapat dipisah (separable), O(N).
///
/// Padanan `cv2.blur` dengan jendela `2*radius+1`. Tepi memakai piksel
/// terdekat (replicate), bukan cermin seperti OpenCV — bedanya hanya di
/// beberapa piksel terluar dan tidak memengaruhi teks.
Float32List boxBlur(Float32List src, int width, int height, int radius) {
  final int win = 2 * radius + 1;
  final Float32List tmp = Float32List(width * height);
  final Float32List out = Float32List(width * height);

  for (int y = 0; y < height; y++) {
    final int row = y * width;
    double sum = 0;
    for (int d = -radius; d <= radius; d++) {
      sum += src[row + d.clamp(0, width - 1)];
    }
    for (int x = 0; x < width; x++) {
      tmp[row + x] = sum / win;
      final int xOut = (x - radius).clamp(0, width - 1);
      final int xIn = (x + radius + 1).clamp(0, width - 1);
      sum += src[row + xIn] - src[row + xOut];
    }
  }

  for (int x = 0; x < width; x++) {
    double sum = 0;
    for (int d = -radius; d <= radius; d++) {
      sum += tmp[d.clamp(0, height - 1) * width + x];
    }
    for (int y = 0; y < height; y++) {
      out[y * width + x] = sum / win;
      final int yOut = (y - radius).clamp(0, height - 1);
      final int yIn = (y + radius + 1).clamp(0, height - 1);
      sum += tmp[yIn * width + x] - tmp[yOut * width + x];
    }
  }
  return out;
}

/// Median 3x3 (padanan `cv2.medianBlur(gray, 3)`).
GrayImage median3(GrayImage src) {
  final int w = src.width;
  final int h = src.height;
  final GrayImage out = GrayImage.zeros(w, h);
  final List<int> window = List<int>.filled(9, 0);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      int n = 0;
      for (int dy = -1; dy <= 1; dy++) {
        final int yy = (y + dy).clamp(0, h - 1);
        for (int dx = -1; dx <= 1; dx++) {
          final int xx = (x + dx).clamp(0, w - 1);
          window[n++] = src.data[yy * w + xx];
        }
      }
      // Insertion sort 9 elemen: lebih cepat daripada sort() karena tidak
      // mengalokasi dan datanya hampir terurut.
      for (int i = 1; i < 9; i++) {
        final int value = window[i];
        int j = i - 1;
        while (j >= 0 && window[j] > value) {
          window[j + 1] = window[j];
          j--;
        }
        window[j + 1] = value;
      }
      out.data[y * w + x] = window[4];
    }
  }
  return out;
}

/// Cermin BORDER_REFLECT_101: `-1 -> 1`, `n -> n-2`.
int _reflect101(int p, int n) {
  if (n <= 1) return 0;
  int value = p;
  while (value < 0 || value >= n) {
    value = value < 0 ? -value : 2 * (n - 1) - value;
  }
  return value;
}

/// Bilateral filter 5x5, padanan `cv2.bilateralFilter(d=5, 40, 40)`.
///
/// Dua detail OpenCV yang wajib ditiru:
/// * kernel-nya **lingkaran** — offset dengan jarak > radius dibuang
///   (13 piksel untuk radius 2, bukan 25);
/// * tepi memakai `BORDER_REFLECT_101`.
///
/// Tanpa keduanya, hasilnya meleset beberapa level abu-abu dan binarisasi
/// memunculkan bintik-bintik palsu.
GrayImage bilateral5(GrayImage src,
    {double sigmaColor = 40, double sigmaSpace = 40}) {
  final int w = src.width;
  final int h = src.height;
  final GrayImage out = GrayImage.zeros(w, h);

  final Float32List rangeLut = Float32List(256);
  final double rangeScale = -0.5 / (sigmaColor * sigmaColor);
  for (int i = 0; i < 256; i++) {
    rangeLut[i] = math.exp(i * i * rangeScale).toDouble();
  }

  const int radius = 2;
  final double spaceScale = -0.5 / (sigmaSpace * sigmaSpace);
  final List<int> offsets = <int>[];
  final List<double> weights = <double>[];
  for (int dy = -radius; dy <= radius; dy++) {
    for (int dx = -radius; dx <= radius; dx++) {
      final double r = math.sqrt((dx * dx + dy * dy).toDouble());
      if (r > radius) continue;
      offsets.add(dx);
      offsets.add(dy);
      weights.add(math.exp(r * r * spaceScale).toDouble());
    }
  }

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final int center = src.data[y * w + x];
      double weighted = 0;
      double total = 0;
      for (int k = 0; k < weights.length; k++) {
        final int xx = _reflect101(x + offsets[k * 2], w);
        final int yy = _reflect101(y + offsets[k * 2 + 1], h);
        final int value = src.data[yy * w + xx];
        final double weight = weights[k] * rangeLut[(value - center).abs()];
        weighted += weight * value;
        total += weight;
      }
      out.data[y * w + x] = (weighted / total).round().clamp(0, 255);
    }
  }
  return out;
}

/// Denoise: median 3x3 lalu bilateral halus.
GrayImage denoise(GrayImage gray) => bilateral5(median3(gray));

// ---------------------------------------------------------------------------
// Binarisasi Sauvola
// ---------------------------------------------------------------------------

/// Nilai persentil dari histogram 256 bin, interpolasi linear seperti numpy.
double _percentile(GrayImage img, double p) {
  final Int32List hist = Int32List(256);
  final Uint8List data = img.data;
  for (int i = 0; i < data.length; i++) {
    hist[data[i]]++;
  }
  final int n = data.length;
  if (n == 0) return 0.0;

  int valueAtRank(int rank) {
    int seen = 0;
    for (int v = 0; v < 256; v++) {
      seen += hist[v];
      if (rank < seen) return v;
    }
    return 255;
  }

  final double index = (n - 1) * p / 100.0;
  final int lo = index.floor();
  final int hi = index.ceil();
  final double frac = index - lo;
  return valueAtRank(lo) * (1 - frac) + valueAtRank(hi) * frac;
}

/// Binarisasi Sauvola: ambang lokal dari rata-rata dan simpangan baku jendela.
BinaryImage sauvola(
  GrayImage gray, {
  int window = P.sauvolaWindow,
  double k = P.sauvolaK,
  double r = P.sauvolaR,
}) {
  final int w = gray.width;
  final int h = gray.height;
  if (w < 3 || h < 3) return BinaryImage.zeros(w, h);

  int win = window;
  if (win.isEven) win += 1;
  final int maxSide = math.min(w, h);
  win = math.max(3, math.min(win, maxSide.isOdd ? maxSide : maxSide - 1));
  final int radius = (win - 1) ~/ 2;

  final Float32List values = Float32List(w * h);
  final Float32List squares = Float32List(w * h);
  for (int i = 0; i < values.length; i++) {
    final int v = gray.data[i];
    values[i] = v.toDouble();
    squares[i] = (v * v).toDouble();
  }

  final Float32List mean = boxBlur(values, w, h, radius);
  final Float32List meanSq = boxBlur(squares, w, h, radius);

  final BinaryImage out = BinaryImage.zeros(w, h);
  for (int i = 0; i < out.length; i++) {
    final double m = mean[i];
    double variance = meanSq[i] - m * m;
    if (variance < 0) variance = 0;
    final double std = math.sqrt(variance);
    final double threshold = m * (1.0 + k * (std / r - 1.0));
    out.data[i] = values[i] < threshold ? 255 : 0;
  }

  // Sauvola gagal di blok gelap seragam (barcode, tinta padat): rata-rata
  // lokal ikut turun sehingga ambangnya berada di bawah piksel itu sendiri.
  // Tambal dengan ambang global dari kertas (persentil terang) dan tinta
  // (persentil gelap).
  final double paper = _percentile(gray, 80);
  final double inkRef = _percentile(gray, 2);
  if (paper - inkRef > 20) {
    final double solidThreshold = (paper + inkRef) / 2.0;
    for (int i = 0; i < out.length; i++) {
      if (out.data[i] == 0 && values[i] < solidThreshold) {
        out.data[i] = 255;
      }
    }
  }
  return out;
}

/// Denoise + Sauvola.
BinaryImage binarize(GrayImage gray) => sauvola(denoise(gray));

// ---------------------------------------------------------------------------
// Resize & rotasi
// ---------------------------------------------------------------------------

/// Kernel cubic Catmull-Rom dengan a = -0.75 (sama seperti OpenCV).
double _cubic(double x) {
  const double a = -0.75;
  final double ax = x.abs();
  if (ax <= 1.0) {
    return ((a + 2) * ax - (a + 3)) * ax * ax + 1;
  }
  if (ax < 2.0) {
    return (((ax - 5) * ax + 8) * ax - 4) * a;
  }
  return 0.0;
}

/// Resize grayscale dengan interpolasi bicubic (padanan `cv2.INTER_CUBIC`).
GrayImage resizeBicubic(GrayImage src, int dstW, int dstH) {
  if (dstW <= 0 || dstH <= 0) return GrayImage.zeros(1, 1);
  if (src.width == dstW && src.height == dstH) return src.clone();

  final int sw = src.width;
  final int sh = src.height;
  final double scaleX = sw / dstW;
  final double scaleY = sh / dstH;

  // Lalu lintas vertikal dulu ke buffer selebar citra sumber, lalu
  // horizontal ke lebar tujuan. Dengan urutan ini buffer antara hanya
  // sebesar sw x dstH, bukan dstW x sh.
  final Float32List tmp = Float32List(sw * dstH);
  final List<int> ys = List<int>.filled(4, 0);
  final List<double> wy = List<double>.filled(4, 0);
  for (int y = 0; y < dstH; y++) {
    final double sy = (y + 0.5) * scaleY - 0.5;
    final int base = sy.floor();
    for (int t = 0; t < 4; t++) {
      wy[t] = _cubic((sy - (base - 1 + t)).abs());
      ys[t] = (base - 1 + t).clamp(0, sh - 1);
    }
    final int row = y * sw;
    for (int x = 0; x < sw; x++) {
      double sum = 0;
      for (int t = 0; t < 4; t++) {
        sum += wy[t] * src.data[ys[t] * sw + x];
      }
      tmp[row + x] = sum;
    }
  }

  final GrayImage out = GrayImage.zeros(dstW, dstH);
  final List<int> xs = List<int>.filled(4, 0);
  final List<double> wx = List<double>.filled(4, 0);
  for (int x = 0; x < dstW; x++) {
    final double sx = (x + 0.5) * scaleX - 0.5;
    final int base = sx.floor();
    for (int t = 0; t < 4; t++) {
      wx[t] = _cubic((sx - (base - 1 + t)).abs());
      xs[t] = (base - 1 + t).clamp(0, sw - 1);
    }
    for (int y = 0; y < dstH; y++) {
      final int row = y * sw;
      double sum = 0;
      for (int t = 0; t < 4; t++) {
        sum += wx[t] * tmp[row + xs[t]];
      }
      out.data[y * dstW + x] = sum.round().clamp(0, 255);
    }
  }
  return out;
}

/// Resize dengan rata-rata area (padanan `cv2.INTER_AREA`).
///
/// Dipakai `normalize_char`: glyph besar diperkecil ke kanvas 32x32. Saat
/// memperbesar, OpenCV INTER_AREA berperilaku seperti nearest; itu ditiru.
GrayImage resizeArea(GrayImage src, int dstW, int dstH) {
  if (dstW <= 0 || dstH <= 0) return GrayImage.zeros(1, 1);
  final GrayImage out = GrayImage.zeros(dstW, dstH);

  if (dstW >= src.width && dstH >= src.height) {
    for (int y = 0; y < dstH; y++) {
      final int sy = (y * src.height ~/ dstH).clamp(0, src.height - 1);
      for (int x = 0; x < dstW; x++) {
        final int sx = (x * src.width ~/ dstW).clamp(0, src.width - 1);
        out.data[y * dstW + x] = src.data[sy * src.width + sx];
      }
    }
    return out;
  }

  final double scaleX = src.width / dstW;
  final double scaleY = src.height / dstH;
  for (int y = 0; y < dstH; y++) {
    final double sy0 = y * scaleY;
    final double sy1 = (y + 1) * scaleY;
    final int iy0 = sy0.floor();
    final int iy1 = math.max(iy0 + 1, sy1.ceil());
    for (int x = 0; x < dstW; x++) {
      final double sx0 = x * scaleX;
      final double sx1 = (x + 1) * scaleX;
      final int ix0 = sx0.floor();
      final int ix1 = math.max(ix0 + 1, sx1.ceil());
      double sum = 0;
      double weight = 0;
      for (int yy = iy0; yy < iy1 && yy < src.height; yy++) {
        final double wy = math.min(sy1, yy + 1.0) - math.max(sy0, yy.toDouble());
        if (wy <= 0) continue;
        for (int xx = ix0; xx < ix1 && xx < src.width; xx++) {
          final double wx = math.min(sx1, xx + 1.0) - math.max(sx0, xx.toDouble());
          if (wx <= 0) continue;
          sum += src.data[yy * src.width + xx] * wx * wy;
          weight += wx * wy;
        }
      }
      out.data[y * dstW + x] = weight > 0 ? (sum / weight).round().clamp(0, 255) : 0;
    }
  }
  return out;
}

/// Rotasi grayscale dengan interpolasi bicubic (padanan `warpAffine`
/// `INTER_CUBIC`), tepi `BORDER_REPLICATE`.
GrayImage rotateGray(GrayImage src, double angleDegrees) {
  final int w = src.width;
  final int h = src.height;
  final double cx = w / 2 - 0.5;
  final double cy = h / 2 - 0.5;
  final double a = angleDegrees * math.pi / 180.0;
  final double alpha = math.cos(a);
  final double beta = math.sin(a);

  final GrayImage out = GrayImage.zeros(w, h);
  final List<double> wx = List<double>.filled(4, 0);
  final List<double> wy = List<double>.filled(4, 0);
  for (int y = 0; y < h; y++) {
    final double dy = y - cy;
    for (int x = 0; x < w; x++) {
      final double dx = x - cx;
      // Invers rotasi: src = R(-a) * (dst - center) + center.
      final double sx = alpha * dx - beta * dy + cx;
      final double sy = beta * dx + alpha * dy + cy;

      final int x0 = sx.floor();
      final int y0 = sy.floor();
      final double fx = sx - x0;
      final double fy = sy - y0;
      for (int t = 0; t < 4; t++) {
        wx[t] = _cubic((fx - (t - 1)).abs());
        wy[t] = _cubic((fy - (t - 1)).abs());
      }

      double sum = 0;
      for (int ty = 0; ty < 4; ty++) {
        final int yy = (y0 - 1 + ty).clamp(0, h - 1);
        double rowSum = 0;
        for (int tx = 0; tx < 4; tx++) {
          final int xx = (x0 - 1 + tx).clamp(0, w - 1);
          rowSum += wx[tx] * src.data[yy * w + xx];
        }
        sum += wy[ty] * rowSum;
      }
      out.data[y * w + x] = sum.round().clamp(0, 255);
    }
  }
  return out;
}

/// Rotasi biner dengan nearest neighbor (padanan `INTER_NEAREST`).
BinaryImage rotateBinaryNearest(BinaryImage src, double angleDegrees) {
  final int w = src.width;
  final int h = src.height;
  final double cx = w / 2 - 0.5;
  final double cy = h / 2 - 0.5;
  final double a = angleDegrees * math.pi / 180.0;
  final double alpha = math.cos(a);
  final double beta = math.sin(a);

  final BinaryImage out = BinaryImage.zeros(w, h);
  for (int y = 0; y < h; y++) {
    final double dy = y - cy;
    for (int x = 0; x < w; x++) {
      final double dx = x - cx;
      final int sx = (alpha * dx - beta * dy + cx).round();
      final int sy = (beta * dx + alpha * dy + cy).round();
      if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
      out.data[y * w + x] = src.data[sy * w + sx];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Estimasi kemiringan & tinggi baris
// ---------------------------------------------------------------------------

/// Ketajaman profil horizontal; maksimum saat baris sejajar sumbu-x.
double _profileScore(BinaryImage binary) {
  final int h = binary.height;
  if (h < 3) return 0.0;
  final Int32List profile = Int32List(h);
  for (int y = 0; y < h; y++) {
    final int row = y * binary.width;
    int count = 0;
    for (int x = 0; x < binary.width; x++) {
      if (binary.data[row + x] != 0) count++;
    }
    profile[y] = count;
  }
  double score = 0;
  for (int y = 1; y < h; y++) {
    final double diff = (profile[y] - profile[y - 1]).toDouble();
    score += diff * diff;
  }
  return score;
}

/// Taksir sudut kemiringan halaman dari citra biner (projection profile).
double estimateSkew(
  BinaryImage binary, {
  double maxAngle = P.deskewMaxAngle,
  double coarseStep = 0.5,
  double fineStep = 0.1,
}) {
  if (!binary.hasInk) return 0.0;

  // Estimasi pada citra kecil: cukup akurat, jauh lebih cepat.
  final double scale =
      math.min(1.0, P.deskewWorkingHeight / math.max(1, binary.height));
  final int sw = math.max(1, (binary.width * scale).round());
  final int sh = math.max(1, (binary.height * scale).round());

  final BinaryImage small = BinaryImage.zeros(sw, sh);
  for (int y = 0; y < sh; y++) {
    final int sy = (y / scale).floor().clamp(0, binary.height - 1);
    for (int x = 0; x < sw; x++) {
      final int sx = (x / scale).floor().clamp(0, binary.width - 1);
      small.data[y * sw + x] = binary.data[sy * binary.width + sx];
    }
  }

  double scoreAt(double angle) =>
      _profileScore(rotateBinaryNearest(small, angle));

  double bestCoarse = -maxAngle;
  double bestScore = -1;
  for (double a = -maxAngle; a <= maxAngle + 1e-9; a += coarseStep) {
    final double s = scoreAt(a);
    if (s > bestScore) {
      bestScore = s;
      bestCoarse = a;
    }
  }

  double bestFine = bestCoarse;
  bestScore = -1;
  for (double a = bestCoarse - coarseStep;
      a <= bestCoarse + coarseStep + 1e-9;
      a += fineStep) {
    final double s = scoreAt(a);
    if (s > bestScore) {
      bestScore = s;
      bestFine = a;
    }
  }
  return bestFine;
}

/// Perkiraan tinggi baris dari persentil ke-75 tinggi komponen tinta.
///
/// Padanan `estimate_line_height`: persentil, bukan median, karena garis
/// pemisah `-----` menyumbang ratusan komponen 2-4 px yang menyeret median.
double estimateLineHeight(BinaryImage binary) {
  final List<int> heights = <int>[];
  for (final GlyphComponent c in connectedComponents(binary)) {
    // cv2.contourArea untuk kotak w x h kira-kira (w-1)*(h-1); komponen yang
    // sangat tipis bernilai 0 dan dibuang, sama seperti di Python.
    if ((c.width - 1) * (c.height - 1) < 3) continue;
    heights.add(c.height);
  }
  if (heights.isEmpty) return 0.0;
  heights.sort();
  final double pos = (heights.length - 1) * 0.75;
  final int lo = pos.floor();
  final int hi = pos.ceil();
  final double frac = pos - lo;
  return heights[lo] * (1 - frac) + heights[hi] * frac;
}

// ---------------------------------------------------------------------------
// Deteksi daerah kertas (foto struk di atas latar gelap)
// ---------------------------------------------------------------------------

/// Jumlah piksel "set" (255) dalam jendela kxk di sekitar (x, y), tepi di-clamp.
int _windowInkCount(Int32List prefix, int w, int h, int x, int y, int r) {
  final int x0 = (x - r).clamp(0, w - 1);
  final int x1 = (x + r).clamp(0, w - 1);
  final int y0 = (y - r).clamp(0, h - 1);
  final int y1 = (y + r).clamp(0, h - 1);
  final int w2 = w + 1;
  return prefix[(y1 + 1) * w2 + (x1 + 1)] -
      prefix[y0 * w2 + (x1 + 1)] -
      prefix[(y1 + 1) * w2 + x0] +
      prefix[y0 * w2 + x0];
}

/// Gambar integral (prefix sum) dari piksel "set" — O(N) sekali bangun.
Int32List _prefixSum(BinaryImage src) {
  final int w = src.width;
  final int h = src.height;
  final int w2 = w + 1;
  final Int32List prefix = Int32List(w2 * (h + 1));
  for (int y = 0; y < h; y++) {
    int rowSum = 0;
    final int srcRow = y * w;
    final int prefRow = (y + 1) * w2;
    final int prefAbove = y * w2;
    for (int x = 0; x < w; x++) {
      rowSum += src.data[srcRow + x] != 0 ? 1 : 0;
      prefix[prefRow + x + 1] = prefix[prefAbove + x + 1] + rowSum;
    }
  }
  return prefix;
}

/// Erosi/dilasi dengan kernel persegi kxk (padanan `morphologyEx` RECT).
///
/// Dipakai gambar integral supaya O(N), bukan O(N·k²) — penting untuk erosi
/// besar di langkah akhir pada citra yang sudah di-upscale.
BinaryImage _morphRect(BinaryImage src, int k, bool erode) {
  final int w = src.width;
  final int h = src.height;
  final int r = k ~/ 2;
  final Int32List prefix = _prefixSum(src);
  final BinaryImage out = BinaryImage.zeros(w, h);
  for (int y = 0; y < h; y++) {
    final int y0 = (y - r).clamp(0, h - 1);
    final int y1 = (y + r).clamp(0, h - 1);
    final int winH = y1 - y0 + 1;
    for (int x = 0; x < w; x++) {
      final int x0 = (x - r).clamp(0, w - 1);
      final int x1 = (x + r).clamp(0, w - 1);
      final int area = (x1 - x0 + 1) * winH;
      final int count = _windowInkCount(prefix, w, h, x, y, r);
      out.data[y * w + x] = (erode ? count == area : count > 0) ? 255 : 0;
    }
  }
  return out;
}

/// Blob kertas (terang) terbesar beserta kotak dan luasnya.
class _PaperRegion {
  _PaperRegion(this.mask, this.left, this.top, this.right, this.bottom, this.area);

  final BinaryImage mask;
  final int left;
  final int top;
  final int right;
  final int bottom;
  final int area;
}

/// Cari blob terang terbesar; ``null`` bila struk memenuhi bingkai (menyentuh
/// keempat tepi) atau tidak ada kertas yang jelas.
///
/// Foto struk yang dipegang di atas kain/meja gelap membuat patch solid global
/// di [sauvola] salah: persentil-2 jatuh ke latar, bukan tinta, sehingga
/// seluruh latar jadi tinta dan semua baris menyatu. Memotong ke blob kertas
/// dan membuang tinta di luarnya menyelesaikan masalah itu.
_PaperRegion? _paperMask(GrayImage gray) {
  final int w = gray.width;
  final int h = gray.height;
  if (w == 0 || h == 0) return null;

  final double paperTone = _percentile(gray, 90);
  final double threshold = paperTone * P.paperMaskThresholdFrac;
  final BinaryImage raw = BinaryImage.zeros(w, h);
  for (int i = 0; i < gray.length; i++) {
    if (gray.data[i] > threshold) raw.data[i] = 255;
  }

  // Open (buang jembatan terang tipis ke kilau latar) lalu close (tutup lubang
  // teks) supaya kertas menjadi satu blob utuh.
  int openK = P.paperMaskOpen;
  if (openK.isEven) openK += 1;
  int closeK = P.paperMaskClose;
  if (closeK.isEven) closeK += 1;
  final BinaryImage opened = _morphRect(_morphRect(raw, openK, true), openK, false);
  final BinaryImage closed = _morphRect(_morphRect(opened, closeK, false), closeK, true);

  // Lintasan pertama: cari seed dan luas komponen terbesar.
  final Uint8List visited = Uint8List(w * h);
  final Int32List stack = Int32List(w * h);
  int bestStart = -1;
  int bestArea = 0;
  for (int start = 0; start < w * h; start++) {
    if (closed.data[start] == 0 || visited[start] != 0) continue;
    int sp = 0;
    stack[sp++] = start;
    visited[start] = 1;
    int area = 0;
    while (sp > 0) {
      final int index = stack[--sp];
      area++;
      final int x = index % w;
      final int y = index ~/ w;
      for (int dy = -1; dy <= 1; dy++) {
        final int yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (int dx = -1; dx <= 1; dx++) {
          final int xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          final int next = yy * w + xx;
          if (closed.data[next] != 0 && visited[next] == 0) {
            visited[next] = 1;
            stack[sp++] = next;
          }
        }
      }
    }
    if (area > bestArea) {
      bestArea = area;
      bestStart = start;
    }
  }
  if (bestStart < 0) return null;
  if (bestArea < w * h * P.paperMaskMinAreaFrac) return null;

  // Lintasan kedua: bangun mask komponen terbesar + bbox.
  final BinaryImage mask = BinaryImage.zeros(w, h);
  visited.fillRange(0, w * h, 0);
  int sp = 0;
  stack[sp++] = bestStart;
  visited[bestStart] = 1;
  int left = w, top = h, right = 0, bottom = 0;
  while (sp > 0) {
    final int index = stack[--sp];
    mask.data[index] = 255;
    final int x = index % w;
    final int y = index ~/ w;
    if (x < left) left = x;
    if (x + 1 > right) right = x + 1;
    if (y < top) top = y;
    if (y + 1 > bottom) bottom = y + 1;
    for (int dy = -1; dy <= 1; dy++) {
      final int yy = y + dy;
      if (yy < 0 || yy >= h) continue;
      for (int dx = -1; dx <= 1; dx++) {
        final int xx = x + dx;
        if (xx < 0 || xx >= w) continue;
        final int next = yy * w + xx;
        if (closed.data[next] != 0 && visited[next] == 0) {
          visited[next] = 1;
          stack[sp++] = next;
        }
      }
    }
  }

  // Menyentuh keempat tepi = struk memenuhi bingkai, tidak ada latar.
  bool touches(bool Function(int, int) test) {
    for (int i = 0; i < w * h; i++) {
      if (mask.data[i] != 0 && test(i % w, i ~/ w)) return true;
    }
    return false;
  }
  final bool touchesAll =
      touches((x, y) => y == 0) &&
      touches((x, y) => y == h - 1) &&
      touches((x, y) => x == 0) &&
      touches((x, y) => x == w - 1);
  if (touchesAll) return null;

  return _PaperRegion(mask, left, top, right, bottom, bestArea);
}

/// Salin sub-citra grayscale.
GrayImage _cropGray(GrayImage src, int left, int top, int right, int bottom) {
  final int w = right - left;
  final int h = bottom - top;
  final GrayImage out = GrayImage.zeros(w, h);
  for (int y = 0; y < h; y++) {
    out.data.setRange(y * w, y * w + w, src.data, (top + y) * src.width + left);
  }
  return out;
}

/// Resize biner nearest neighbor (padanan `INTER_NEAREST`).
BinaryImage _resizeNearest(BinaryImage src, int dstW, int dstH) {
  final BinaryImage out = BinaryImage.zeros(dstW, dstH);
  for (int y = 0; y < dstH; y++) {
    final int sy = (y * src.height ~/ dstH).clamp(0, src.height - 1);
    for (int x = 0; x < dstW; x++) {
      final int sx = (x * src.width ~/ dstW).clamp(0, src.width - 1);
      out.data[y * dstW + x] = src.data[sy * src.width + sx];
    }
  }
  return out;
}

/// Jalur praproses lengkap: [potong kertas] → deskew → binarisasi → upscale.
///
/// Urutan penting: kemiringan dikoreksi **sebelum** upscale. Bila struk berada
/// di atas latar gelap, citra dipotong dulu ke daerah kertas (lihat
/// [_paperMask]) supaya patch solid Sauvola tidak salah menandai latar sebagai
/// tinta.
Prepared prepare(GrayImage gray, {double targetLineHeight = P.targetLineHeight}) {
  GrayImage base = gray;
  final _PaperRegion? region = _paperMask(base);
  BinaryImage? paper = region?.mask;

  // 0. Potong ke daerah kertas bila ada latar gelap di sekelilingnya.
  if (region != null) {
    final int bw = region.right - region.left;
    final int bh = region.bottom - region.top;
    final int margin = math.max(4, (0.015 * math.max(bw, bh)).round());
    final int x0 = (region.left - margin).clamp(0, base.width);
    final int y0 = (region.top - margin).clamp(0, base.height);
    final int x1 = (region.right + margin).clamp(0, base.width);
    final int y1 = (region.bottom + margin).clamp(0, base.height);
    if (x1 - x0 >= 60 && y1 - y0 >= 60) {
      base = _cropGray(base, x0, y0, x1, y1);
      paper = paper!.crop(x0, y0, x1, y1);
    } else {
      paper = null;
    }
  }

  // 1. Binarisasi awal hanya untuk menaksir sudut.
  final BinaryImage probe = binarize(base);
  final double angle = estimateSkew(probe);

  // 2. Putar grayscale (bukan biner) supaya tepi tetap halus.
  if (angle.abs() >= P.deskewMinAngle) {
    base = rotateGray(base, angle);
    if (paper != null) paper = rotateBinaryNearest(paper, angle);
  }

  // 3. Binarisasi final.
  BinaryImage binary = binarize(base);

  final double height = estimateLineHeight(binary);
  double scale = 1.0;
  if (height > 0) {
    scale = (targetLineHeight / height).clamp(1.0, P.maxUpscale);
  }

  if (scale > 1.01) {
    final int sizeW = (base.width * scale).round();
    final int sizeH = (base.height * scale).round();
    final GrayImage up = resizeBicubic(base, sizeW, sizeH);
    binary = binarize(up);
    if (paper != null) paper = _resizeNearest(paper, sizeW, sizeH);
  }

  // 4. Buang tinta di luar kertas: erode mask dulu agar garis transisi di tepi
  //    (yang Sauvola tandai sebagai tinta) ikut hilang, lalu nol-kan daerah di
  //    luar mask. Tanpa ini garis tepi vertikal menjaga profil horizontal tetap
  //    di atas ambang dan semua baris menyatu.
  if (paper != null) {
    int k = math.max(3,
        (P.paperMaskErodeFrac * math.min(binary.width, binary.height)).round());
    if (k.isEven) k += 1;
    paper = _morphRect(paper, k, true);
    for (int i = 0; i < binary.length; i++) {
      if (paper.data[i] == 0) binary.data[i] = 0;
    }
  }
  return Prepared(gray: base, binary: binary, scale: scale, skew: angle);
}

/// Komponen tinta hasil connected components.
class GlyphComponent {
  GlyphComponent(this.left, this.top, this.right, this.bottom);

  final int left;
  final int top;
  final int right;
  final int bottom;

  int get width => right - left;
  int get height => bottom - top;
}

/// Connected components 8-konektivitas pada citra biner.
///
/// Dipakai baik oleh `estimateLineHeight` (bbox komponen) maupun segmentasi
/// glyph. Implementasi iteratif dengan stack eksplisit agar tidak meluap
/// pada komponen besar.
List<GlyphComponent> connectedComponents(BinaryImage binary) {
  final int w = binary.width;
  final int h = binary.height;
  final Uint8List visited = Uint8List(w * h);
  final List<GlyphComponent> out = <GlyphComponent>[];
  final Int32List stack = Int32List(w * h);
  final Uint8List data = binary.data;

  for (int start = 0; start < data.length; start++) {
    if (data[start] == 0 || visited[start] != 0) continue;
    int top = h, bottom = 0, left = w, right = 0;
    int sp = 0;
    stack[sp++] = start;
    visited[start] = 1;
    int area = 0;
    while (sp > 0) {
      final int index = stack[--sp];
      final int x = index % w;
      final int y = index ~/ w;
      area++;
      if (x < left) left = x;
      if (x + 1 > right) right = x + 1;
      if (y < top) top = y;
      if (y + 1 > bottom) bottom = y + 1;
      for (int dy = -1; dy <= 1; dy++) {
        final int yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (int dx = -1; dx <= 1; dx++) {
          final int xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          final int next = yy * w + xx;
          if (data[next] != 0 && visited[next] == 0) {
            visited[next] = 1;
            stack[sp++] = next;
          }
        }
      }
    }
    if (area < 2) continue;
    out.add(GlyphComponent(left, top, right, bottom));
  }
  return out;
}
