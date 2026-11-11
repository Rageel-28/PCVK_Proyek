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

  // Ambang terang: kertas lebih terang daripada latar.
  final List<int> hist = gray.histogram();
  final int threshold = otsuThresholdFromHistogram(hist, gray.length);

  final GrayImage bright = GrayImage.zeros(w, h);
  for (int i = 0; i < gray.length; i++) {
    bright.data[i] = gray.data[i] > threshold ? 255 : 0;
  }

  final List<InkComponent> comps = connectedComponents(bright, minArea: 1);
  if (comps.isEmpty) return null;

  comps.sort((InkComponent a, InkComponent b) => b.area.compareTo(a.area));
  final InkComponent best = comps.first;

  final double areaFraction = best.area / (w * h);
  if (areaFraction < kPaperMinAreaFraction) return null;

  final double aspect = best.height / math.max(best.width, 1);
  if (aspect < kPaperMinAspect || aspect > kPaperMaxAspect) return null;

  final double fill = best.area / math.max(best.width * best.height, 1);
  if (fill < kPaperMinFill) return null;

  final int margin = (0.01 * math.max(best.width, best.height)).round();
  final int x0 = math.max(0, best.left - margin);
  final int y0 = math.max(0, best.top - margin);
  final int x1 = math.min(w - 1, best.right + margin);
  final int y1 = math.min(h - 1, best.bottom + margin);

  return ImageBox(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
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
  return (image: cropped, box: box);
}
