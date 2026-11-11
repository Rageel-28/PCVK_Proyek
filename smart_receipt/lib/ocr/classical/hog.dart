import 'dart:math' as math;
import 'dart:typed_data';

import 'classical_config.dart';
import 'gray_image.dart';

/// Ekstraksi fitur HOG (Histogram of Oriented Gradients).
///
/// Ditulis manual, tanpa library, karena versi ini harus meniru
/// perhitungan Python (`ML/ocr_classical/features.py`) sedekat mungkin.
///
/// Langkah:
///   1. Inversi citra: tinta menjadi terang (255 - piksel)
///   2. Gradien: gx = I[:,x+1] - I[:,x-1], gy = I[y+1,:] - I[y-1,:]
///   3. Magnitudo m = sqrt(gx^2 + gy^2)
///      Orientasi theta = atan2(gy, gx) dalam derajat, dimodulo 180
///   4. Setiap sel 8x8 mengakumulasi m ke 9 bin orientasi
///      (interpolasi bilinear ke dua bin terdekat)
///   5. Blok 2x2 sel (bertumpuk, stride 1) dinormalisasi L2-Hys
///   6. Hasil: 3 x 3 blok x 2 x 2 sel x 9 bin = 324 dimensi
///
/// Normalisasi L2-Hys: bagi dengan norma L2, potong nilai > 0.2, bagi lagi.
/// Ini membuat fitur tahan terhadap perbedaan kontras antar citra.
Float32List hog(
  GrayImage image, {
  int cell = kHogCell,
  int bins = kHogBins,
  int block = kHogBlock,
}) {
  if (image.width != kCharSize || image.height != kCharSize) {
    throw ArgumentError(
      'Dimensi harus ${kCharSize}x$kCharSize, dapat '
      '${image.width}x${image.height}',
    );
  }

  final int size = kCharSize;

  // Inversi dan normalisasi ke 0..1 seperti Python (g = img / 255).
  final Float32List inv = Float32List(size * size);
  for (int i = 0; i < inv.length; i++) {
    inv[i] = (255.0 - image.data[i]) / 255.0;
  }

  // Gradien dengan selisih pusat; tepi dibiarkan nol.
  final Float32List gx = Float32List(size * size);
  final Float32List gy = Float32List(size * size);

  for (int y = 0; y < size; y++) {
    final int rowBase = y * size;
    for (int x = 1; x < size - 1; x++) {
      gx[rowBase + x] = inv[rowBase + x + 1] - inv[rowBase + x - 1];
    }
  }
  for (int y = 1; y < size - 1; y++) {
    final int rowBase = y * size;
    final int above = (y - 1) * size;
    final int below = (y + 1) * size;
    for (int x = 0; x < size; x++) {
      gy[rowBase + x] = inv[below + x] - inv[above + x];
    }
  }

  final Float32List mag = Float32List(size * size);
  final Float32List ang = Float32List(size * size);
  for (int i = 0; i < mag.length; i++) {
    final double dx = gx[i];
    final double dy = gy[i];
    mag[i] = math.sqrt(dx * dx + dy * dy);
    double deg = math.atan2(dy, dx) * 180.0 / math.pi;
    deg = deg % 180.0;
    if (deg < 0) deg += 180.0;
    ang[i] = deg;
  }

  final int nCells = size ~/ cell;
  final double binWidth = 180.0 / bins;

  // Akumulasi histogram per sel: [cy][cx][bin].
  final Float32List hist = Float32List(nCells * nCells * bins);

  for (int cy = 0; cy < nCells; cy++) {
    for (int cx = 0; cx < nCells; cx++) {
      final int cellBase = (cy * nCells + cx) * bins;
      for (int py = cy * cell; py < (cy + 1) * cell; py++) {
        final int rowBase = py * size;
        for (int px = cx * cell; px < (cx + 1) * cell; px++) {
          final int idx = rowBase + px;
          final double m = mag[idx];
          if (m == 0) continue;

          final double pos = ang[idx] / binWidth;
          final int lo = pos.floor() % bins;
          final double frac = pos - pos.floor();
          final int hi = (lo + 1) % bins;

          hist[cellBase + lo] += m * (1.0 - frac);
          hist[cellBase + hi] += m * frac;
        }
      }
    }
  }

  // Blok 2x2 sel bertumpuk, normalisasi L2-Hys.
  final int blocksPerSide = nCells - block + 1;
  final int blockLen = block * block * bins;
  final Float32List out = Float32List(blocksPerSide * blocksPerSide * blockLen);

  int writeIndex = 0;
  for (int by = 0; by < blocksPerSide; by++) {
    for (int bx = 0; bx < blocksPerSide; bx++) {
      final int start = writeIndex;

      // Salin blok ke keluaran.
      for (int dy = 0; dy < block; dy++) {
        for (int dx = 0; dx < block; dx++) {
          final int cellBase = ((by + dy) * nCells + (bx + dx)) * bins;
          for (int b = 0; b < bins; b++) {
            out[writeIndex++] = hist[cellBase + b];
          }
        }
      }

      // L2-Hys: norm -> clip 0.2 -> norm.
      double norm = 0;
      for (int i = start; i < start + blockLen; i++) {
        norm += out[i] * out[i];
      }
      norm = math.sqrt(norm) + 1e-6;
      for (int i = start; i < start + blockLen; i++) {
        double v = out[i] / norm;
        if (v > 0.2) v = 0.2;
        out[i] = v;
      }
      norm = 0;
      for (int i = start; i < start + blockLen; i++) {
        norm += out[i] * out[i];
      }
      norm = math.sqrt(norm) + 1e-6;
      for (int i = start; i < start + blockLen; i++) {
        out[i] = out[i] / norm;
      }
    }
  }

  return out;
}
