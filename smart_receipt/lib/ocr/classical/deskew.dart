import 'dart:math' as math;

import 'classical_config.dart';
import 'gray_image.dart';

/// Estimasi dan koreksi kemiringan (deskew) berbasis profil proyeksi.
///
/// Ide: bila teks tegak, jumlah piksel tinta per baris membentuk pola
/// puncak-lembah yang tajam. Semakin miring, pola itu semakin kabur.
/// Maka sudut terbaik adalah sudut yang memaksimalkan variansi profil
/// horizontal.
///
/// Diimplementasikan manual agar identik dengan versi Python dan tidak
/// bergantung pada library pengolahan citra apa pun.

/// Variansi profil horizontal pada sudut tertentu.
double _profileScore(GrayImage ink, double angle) {
  final GrayImage rotated = ink.rotated(angle);
  final int w = rotated.width;
  final int h = rotated.height;

  // Profil = jumlah piksel tinta per baris.
  final List<int> profile = List<int>.filled(h, 0);
  for (int y = 0; y < h; y++) {
    int count = 0;
    final int rowBase = y * w;
    for (int x = 0; x < w; x++) {
      if (rotated.data[rowBase + x] != 0) count++;
    }
    profile[y] = count;
  }

  // Variansi profil.
  double sum = 0;
  for (final int v in profile) {
    sum += v;
  }
  final double mean = sum / h;
  double variance = 0;
  for (final int v in profile) {
    final double d = v - mean;
    variance += d * d;
  }
  return variance / h;
}

/// Perkecil citra agar estimasi sudut cepat.
GrayImage _downscale(GrayImage ink, int longest) {
  final int m = math.max(ink.width, ink.height);
  if (m <= longest) {
    return ink;
  }
  final double scale = longest / m;
  final int nw = math.max(1, (ink.width * scale).round());
  final int nh = math.max(1, (ink.height * scale).round());
  return ink.resized(nw, nh);
}

/// Kembalikan sudut (derajat) yang harus diterapkan agar teks tegak.
double estimateSkew(
  GrayImage ink, {
  double maxAngle = kDeskewMaxAngle,
  double coarseStep = kDeskewCoarseStep,
  double fineStep = kDeskewFineStep,
}) {
  if (ink.length == 0 || !_hasInk(ink)) {
    return 0.0;
  }

  final GrayImage small = _downscale(ink, kDeskewMaxDimension);

  double bestAngle = 0.0;
  double bestScore = -1.0;

  for (double a = -maxAngle; a <= maxAngle + 1e-9; a += coarseStep) {
    final double score = _profileScore(small, a);
    if (score > bestScore) {
      bestScore = score;
      bestAngle = a;
    }
  }

  final double lo = bestAngle - coarseStep;
  final double hi = bestAngle + coarseStep;
  for (double a = lo; a <= hi + 1e-9; a += fineStep) {
    final double score = _profileScore(small, a);
    if (score > bestScore) {
      bestScore = score;
      bestAngle = a;
    }
  }

  return bestAngle;
}

bool _hasInk(GrayImage ink) {
  for (final int v in ink.data) {
    if (v != 0) return true;
  }
  return false;
}

/// Estimasi lalu koreksi sekaligus pada grayscale dan mask tinta.
///
/// Keduanya harus memakai sudut yang sama supaya bbox hasil segmentasi
/// pada mask tetap valid ketika dipakai memotong grayscale.
({GrayImage gray, GrayImage ink, double angle}) autoDeskewPair(
  GrayImage gray,
  GrayImage ink, {
  double maxAngle = kDeskewMaxAngle,
}) {
  double angle = estimateSkew(ink, maxAngle: maxAngle);
  if (angle.abs() > maxAngle) {
    angle = 0.0;
  }
  if (angle.abs() < 1e-6) {
    return (gray: gray, ink: ink, angle: 0.0);
  }
  return (
    gray: gray.rotated(angle, borderValue: 255),
    ink: ink.rotated(angle, borderValue: 0),
    angle: angle,
  );
}
