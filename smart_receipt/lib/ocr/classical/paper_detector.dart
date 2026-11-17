import 'dart:math' as math;

import 'classical_config.dart';
import 'connected_components.dart';
import 'gray_image.dart';

/// Area persegi panjang pada citra.
class ImageBox {
  const ImageBox(this.left, this.top, this.width, this.height);

  final int left;
  final int top;
  final int width;
  final int height;

  int get right => left + width - 1;
  int get bottom => top + height - 1;
}

/// Deteksi area kertas struk pada foto.
///
/// Foto struk biasanya memuat latar gelap (meja, keyboard, tangan). Bila
/// latar ikut diproses, tepi kertas meninggalkan tinta tipis di setiap
/// baris sehingga segmentasi baris gagal.
///
/// Pendekatan (CV klasik):
///   1. Otsu pada kecerahan -> kertas menjadi putih, latar hitam
///   2. Komponen terhubung -> ambil yang terbesar
///   3. Periksa kelayakan: luas, rasio aspek, kerapatan isi
///   4. Potong dengan margin kecil
ImageBox? findPaperRegion(GrayImage gray) {
  final int w = gray.width;
  final int h = gray.height;
  if (w < 8 || h < 8) return null;

  // Ambang terang: Otsu awal, lalu fallback ambang lebih tinggi bila latar
  // terang ikut menyatu dengan kertas (>90% bingkai).
  final List<int> hist = gray.histogram();
  final int otsuT = otsuThresholdFromHistogram(hist, gray.length);
  final List<int> thresholds = <int>[otsuT, 195, 205, 185];

  for (int ti = 0; ti < thresholds.length; ti++) {
    final int threshold = thresholds[ti];
    final GrayImage bright = GrayImage.zeros(w, h);
    for (int i = 0; i < gray.length; i++) {
      bright.data[i] = gray.data[i] > threshold ? 255 : 0;
    }

    final List<InkComponent> comps = connectedComponents(bright, minArea: 1);
    if (comps.isEmpty) continue;

    comps.sort((InkComponent a, InkComponent b) => b.area.compareTo(a.area));
    final InkComponent best = comps.first;

    final double boxFraction = (best.width * best.height) / (w * h);
    if (ti == 0 && boxFraction > 0.90) continue;
    if (boxFraction > 0.92) continue;

    final double areaFraction = best.area / (w * h);
    if (areaFraction < 0.15) continue;

    final double aspect = best.height / math.max(best.width, 1);
    if (aspect < kPaperMinAspect || aspect > kPaperMaxAspect) continue;

    final double fill = best.area / math.max(best.width * best.height, 1);
    if (fill < 0.50) continue;

    final int margin = (0.01 * math.max(best.width, best.height)).round();
    final int x0 = math.max(0, best.left - margin);
    final int y0 = math.max(0, best.top - margin);
    final int x1 = math.min(w - 1, best.right + margin);
    final int y1 = math.min(h - 1, best.bottom + margin);

    return ImageBox(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
  }
  return null;
}

/// Otsu dari histogram (dipakai juga oleh binarisasi).
int otsuThresholdFromHistogram(List<int> hist, int total) {
  if (total <= 0) return 128;
  double sumAll = 0;
  for (int i = 0; i < 256; i++) {
    sumAll += i * hist[i];
  }
  double weightBg = 0;
  double sumBg = 0;
  double bestVariance = -1;
  int bestThreshold = 0;

  for (int t = 0; t < 256; t++) {
    weightBg += hist[t];
    if (weightBg == 0) continue;
    final double weightFg = total - weightBg;
    if (weightFg == 0) break;
    sumBg += t * hist[t];
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

/// Hilangkan segitiga latar gelap di tepi kertas miring dengan banjir tepi.
GrayImage _cleanPaperBorders(GrayImage crop) {
  final int w = crop.width;
  final int h = crop.height;
  final int total = w * h;
  if (w < 8 || h < 8) return crop;

  final int tBg = otsuThresholdFromHistogram(crop.histogram(), total);
  final List<int> borderMask = List<int>.filled(total, 0);
  final List<int> stack = <int>[];

  void seed(int x, int y) {
    final int idx = y * w + x;
    if (borderMask[idx] == 0 && crop.data[idx] <= tBg) {
      borderMask[idx] = 1;
      stack.add(idx);
    }
  }

  for (int x = 0; x < w; x++) {
    seed(x, 0);
    seed(x, h - 1);
  }
  for (int y = 0; y < h; y++) {
    seed(0, y);
    seed(w - 1, y);
  }

  while (stack.isNotEmpty) {
    final int idx = stack.removeLast();
    final int x = idx % w;
    final int y = idx ~/ w;
    if (x > 0) seed(x - 1, y);
    if (x + 1 < w) seed(x + 1, y);
    if (y > 0) seed(x, y - 1);
    if (y + 1 < h) seed(x, y + 1);
  }

  if (stack.isEmpty && !borderMask.any((int v) => v == 1)) {
    return crop;
  }

  // Dilasi 5x5 untuk membersihkan bayangan transisi di tepi kertas.
  final List<int> dilated = List<int>.filled(total, 0);
  int bgCount = 0;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (borderMask[y * w + x] == 0) continue;
      final int y0 = math.max(0, y - 2);
      final int y1 = math.min(h - 1, y + 2);
      final int x0 = math.max(0, x - 2);
      final int x1 = math.min(w - 1, x + 2);
      for (int ny = y0; ny <= y1; ny++) {
        final int rowBase = ny * w;
        for (int nx = x0; nx <= x1; nx++) {
          if (dilated[rowBase + nx] == 0) {
            dilated[rowBase + nx] = 1;
            bgCount++;
          }
        }
      }
    }
  }

  if (bgCount <= 0 || bgCount >= total * 0.40) {
    return crop;
  }

  final List<int> remHist = List<int>.filled(256, 0);
  for (int i = 0; i < total; i++) {
    if (dilated[i] == 0) {
      remHist[crop.data[i]]++;
    }
  }
  final int remTotal = total - bgCount;
  final int target = (remTotal + 1) ~/ 2;
  int acc = 0;
  int med = 255;
  for (int v = 0; v < 256; v++) {
    acc += remHist[v];
    if (acc >= target) {
      med = v;
      break;
    }
  }

  final GrayImage out = crop.clone();
  for (int i = 0; i < total; i++) {
    if (dilated[i] != 0) {
      out.data[i] = med;
    }
  }
  return out;
}

/// Potong area kertas bila terdeteksi; jika tidak, kembalikan aslinya.
///
/// `crop` juga mengembalikan box agar koordinat hasil OCR dapat
/// dipetakan kembali ke citra asli bila diperlukan.
({GrayImage image, ImageBox? box}) cropPaper(GrayImage gray) {
  final ImageBox? box = findPaperRegion(gray);
  if (box == null) {
    return (image: gray, box: null);
  }
  // Jangan potong bila hasilnya nyaris sama dengan gambar asli.
  if (box.width * box.height > 0.98 * gray.width * gray.height) {
    return (image: gray, box: null);
  }
  final GrayImage cropped =
      gray.crop(box.left, box.top, box.right, box.bottom);
  return (image: _cleanPaperBorders(cropped), box: box);
}
