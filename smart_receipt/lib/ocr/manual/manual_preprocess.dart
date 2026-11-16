import 'dart:math' as math;

import '../classical/gray_image.dart';
import 'manual_config.dart';
import 'pil_resize.dart';

/// Ambang Otsu global dari histogram, persis seperti
/// `otsu_threshold()` di `annotate_lines.py`.
///
/// Dipilih karena terbukti mengalahkan adaptive threshold pada dokumen
/// dua-tone seperti struk (89,4% vs 47,2% pada uji yang terdokumentasi).
int otsuThreshold(GrayImage gray) {
  final List<int> hist = List<int>.filled(256, 0);
  for (int i = 0; i < gray.length; i++) {
    hist[gray.data[i]]++;
  }

  final double total = gray.length.toDouble();
  double sumAll = 0;
  for (int i = 0; i < 256; i++) {
    sumAll += i * hist[i];
  }

  double omega = 0;
  double mu = 0;
  double best = -1;
  int bestThreshold = 0;
  for (int t = 0; t < 256; t++) {
    omega += hist[t];
    if (omega == 0) continue;
    final double weightFg = total - omega;
    if (weightFg == 0) break;

    mu += t * hist[t];
    final double meanBg = mu / omega;
    final double meanFg = (sumAll - mu) / weightFg;
    final double sigma = omega * weightFg * (meanBg - meanFg) * (meanBg - meanFg);
    if (sigma > best) {
      best = sigma;
      bestThreshold = t;
    }
  }
  return bestThreshold;
}

/// Mask biner dengan konvensi seluruh pipeline: 255 = tinta, 0 = kertas.
GrayImage binarizeMask(GrayImage gray) {
  final int threshold = otsuThreshold(gray);
  final GrayImage out = GrayImage.zeros(gray.width, gray.height);
  for (int i = 0; i < gray.length; i++) {
    out.data[i] = gray.data[i] < threshold ? 255 : 0;
  }
  return out;
}

/// Hapus garis tabel pada citra penuh.
///
/// Sebuah garis horizontal adalah baris yang hampir seluruh lebarnya
/// bertinta (>= [minRun]); garis vertikal serupa pada sumbu kolom.
/// Dijalankan pada citra penuh, bukan per baris: pada potongan sempit
/// deret digit padat (`081584471843`) bisa terlihat seperti garis
/// vertikal dan ikut terhapus, sehingga satu digit pecah menjadi
/// beberapa komponen.
GrayImage removeRules(GrayImage ink, {double minRun = kManualRuleRunRatio}) {
  final int h = ink.height;
  final int w = ink.width;
  if (h == 0 || w == 0) return ink.clone();

  final GrayImage out = ink.clone();

  // Baris: hitung rasio piksel bertinta, hapus yang melewati ambang.
  final List<int> rowFill = List<int>.filled(h, 0);
  for (int y = 0; y < h; y++) {
    final int base = y * w;
    int count = 0;
    for (int x = 0; x < w; x++) {
      if (out.data[base + x] != 0) count++;
    }
    rowFill[y] = count;
  }
  for (int y = 0; y < h; y++) {
    if (rowFill[y] / w >= minRun) {
      out.data.fillRange(y * w, y * w + w, 0);
    }
  }

  // Kolom dihitung SETELAH baris dibuang (Pillow/numpy juga begitu).
  for (int x = 0; x < w; x++) {
    int count = 0;
    for (int y = 0; y < h; y++) {
      if (out.data[y * w + x] != 0) count++;
    }
    if (count / h >= minRun) {
      for (int y = 0; y < h; y++) {
        out.data[y * w + x] = 0;
      }
    }
  }

  return out;
}

/// Median seperti `numpy.median`: nilai tengah, atau rata-rata dua nilai
/// tengah bila jumlahnya genap.
double medianOf(List<int> values) {
  if (values.isEmpty) return 0.0;
  final List<int> sorted = List<int>.of(values)..sort();
  final int n = sorted.length;
  if (n.isOdd) return sorted[n ~/ 2].toDouble();
  return (sorted[n ~/ 2 - 1] + sorted[n ~/ 2]) / 2.0;
}

/// Perkiraan tinggi baris teks, diukur pada salinan kecil untuk kecepatan.
double estimateLineHeight(GrayImage image) {
  GrayImage probe = image;
  if (image.width > 420) {
    final double scale = 420 / image.width;
    probe = pilResize(
      image,
      420,
      math.max(1, (image.height * scale).toInt()),
      PilFilter.bilinear,
    );
  }

  final GrayImage mask = binarizeMask(probe);
  final List<int> profile = List<int>.filled(mask.height, 0);
  for (int y = 0; y < mask.height; y++) {
    final int base = y * mask.width;
    int count = 0;
    for (int x = 0; x < mask.width; x++) {
      if (mask.data[base + x] != 0) count++;
    }
    profile[y] = count;
  }

  final int threshold = math.max(2, (mask.width * 0.01).toInt());
  final List<int> heights = <int>[];
  int? start;
  for (int y = 0; y < profile.length; y++) {
    if (profile[y] > threshold) {
      start ??= y;
    } else if (start != null) {
      heights.add(y - start);
      start = null;
    }
  }
  if (start != null) heights.add(profile.length - start);
  if (heights.isEmpty) return 0.0;

  final double scaleBack = image.width / probe.width;
  return medianOf(heights) * scaleBack;
}

/// Muat citra dengan batas ukuran + pembesaran adaptif, seperti
/// `load_image()` di `annotate_lines.py`.
///
/// Citra beresolusi rendah membuat tinggi baris jatuh di bawah 18 px
/// sehingga seluruh baris dibuang dan citra dianggap kosong. Untuk
/// dataset render (528x854, glyph ~14 px) ini kasus nyata: tanpa
/// pembesaran, 500 citra menghasilkan NOL data.
GrayImage loadAndUpscale(GrayImage source) {
  GrayImage image = source;

  final int longest = math.max(image.width, image.height);
  if (longest > kManualMaxDimension) {
    final double scale = kManualMaxDimension / longest;
    image = pilResize(
      image,
      math.max(1, (image.width * scale).toInt()),
      math.max(1, (image.height * scale).toInt()),
      PilFilter.bilinear,
    );
  }

  final double lineHeight = estimateLineHeight(image);
  if (lineHeight > 0 && lineHeight < kManualTargetLineHeight) {
    final double factor =
        math.min(kManualTargetLineHeight / lineHeight, kManualMaxUpscale);
    int newWidth = (image.width * factor).toInt();
    int newHeight = (image.height * factor).toInt();

    // Jaga agar tetap di bawah batas memori.
    if (math.max(newWidth, newHeight) > kManualMaxDimension) {
      final double shrink = kManualMaxDimension / math.max(newWidth, newHeight);
      newWidth = (newWidth * shrink).toInt();
      newHeight = (newHeight * shrink).toInt();
    }
    image = pilResize(
      image,
      math.max(1, newWidth),
      math.max(1, newHeight),
      PilFilter.lanczos,
    );
  }

  return image;
}

/// Binerisasi lalu buang garis tabel: mask yang dipakai semua tahap hilir.
GrayImage prepareManualMask(GrayImage image) =>
    removeRules(binarizeMask(image));
