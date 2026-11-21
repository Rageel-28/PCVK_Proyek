/// Ekstraksi fitur karakter 1844-d.
///
/// Port dari `ML/ocr/features.py`:
/// HOG 1764 (cell 4, block 2, 9 bin) + zoning 48 + proyeksi 24 + lubang 8.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'params.dart';

const double _eps = 1e-6;

/// Gradien gx, gy dengan beda tengah (sama seperti `np.gradient`).
void _gradients(Uint8List img, Float32List gx, Float32List gy) {
  const int n = P.charSize;
  for (int y = 0; y < n; y++) {
    for (int x = 0; x < n; x++) {
      final int i = y * n + x;
      double dx;
      if (x == 0) {
        dx = (img[i + 1] - img[i]) / 255.0;
      } else if (x == n - 1) {
        dx = (img[i] - img[i - 1]) / 255.0;
      } else {
        dx = (img[i + 1] - img[i - 1]) / (2.0 * 255.0);
      }
      double dy;
      if (y == 0) {
        dy = (img[i + n] - img[i]) / 255.0;
      } else if (y == n - 1) {
        dy = (img[i] - img[i - n]) / 255.0;
      } else {
        dy = (img[i + n] - img[i - n]) / (2.0 * 255.0);
      }
      gx[i] = dx;
      gy[i] = dy;
    }
  }
}

/// HOG manual 1764 dimensi.
Float32List hogFeatures(Uint8List img) {
  const int n = P.charSize;
  final Float32List gx = Float32List(n * n);
  final Float32List gy = Float32List(n * n);
  _gradients(img, gx, gy);

  const int cellsPerSide = n ~/ P.hogCell; // 4
  const int nCells = cellsPerSide * cellsPerSide;
  const double binWidth = 180.0 / P.hogBins;

  final Float32List hist = Float32List(nCells * P.hogBins);
  for (int y = 0; y < n; y++) {
    for (int x = 0; x < n; x++) {
      final int i = y * n + x;
      final double mag = math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
      if (mag == 0) continue;

      double orientation = math.atan2(gy[i], gx[i]) * 180.0 / math.pi;
      orientation = orientation % 180.0;
      if (orientation < 0) orientation += 180.0;

      final double position = orientation / binWidth;
      final int lower = position.floor();
      final double upperWeight = position - lower;
      final double lowerWeight = 1.0 - upperWeight;
      final int bin0 = lower % P.hogBins;
      final int bin1 = (bin0 + 1) % P.hogBins;

      final int cellFlat = (y ~/ P.hogCell) * cellsPerSide + (x ~/ P.hogCell);
      hist[cellFlat * P.hogBins + bin0] += mag * lowerWeight;
      hist[cellFlat * P.hogBins + bin1] += mag * upperWeight;
    }
  }

  const int blocksPerSide = cellsPerSide - P.hogBlock + 1; // 3
  const int blockSize = P.hogBlock * P.hogBlock * P.hogBins;
  final Float32List out = Float32List(blocksPerSide * blocksPerSide * blockSize);
  int pos = 0;
  for (int by = 0; by < blocksPerSide; by++) {
    for (int bx = 0; bx < blocksPerSide; bx++) {
      final Float32List block = Float32List(blockSize);
      int k = 0;
      for (int cy = 0; cy < P.hogBlock; cy++) {
        for (int cx = 0; cx < P.hogBlock; cx++) {
          final int cell = (by + cy) * cellsPerSide + (bx + cx);
          for (int b = 0; b < P.hogBins; b++) {
            block[k++] = hist[cell * P.hogBins + b];
          }
        }
      }
      double sum = 0;
      for (final double v in block) {
        sum += v * v;
      }
      final double norm = math.sqrt(sum + _eps * _eps);
      for (int j = 0; j < blockSize; j++) {
        out[pos + j] = norm > 0 ? block[j] / norm : 0;
      }
      pos += blockSize;
    }
  }
  return out;
}

/// Zoning 48-d: grid 4x4 x (densitas, sentroid-x, sentroid-y).
Float32List zoningFeatures(Uint8List img) {
  const int n = P.charSize;
  const int grid = P.zoningGrid;
  const int cell = n ~/ grid;

  final Float32List out = Float32List(grid * grid * P.zoningStats);
  final Float32List coords = Float32List(cell);
  for (int i = 0; i < cell; i++) {
    coords[i] = i / math.max(1, cell - 1);
  }

  for (int gy = 0; gy < grid; gy++) {
    for (int gx = 0; gx < grid; gx++) {
      double total = 0;
      double sumX = 0;
      double sumY = 0;
      for (int y = 0; y < cell; y++) {
        for (int x = 0; x < cell; x++) {
          final int px = gx * cell + x;
          final int py = gy * cell + y;
          final double value = img[py * n + px] / 255.0;
          total += value;
          sumX += value * coords[x];
          sumY += value * coords[y];
        }
      }
      final double ink = total + _eps;
      final int base = (gy * grid + gx) * P.zoningStats;
      out[base] = total / (cell * cell);
      out[base + 1] = sumX / ink;
      out[base + 2] = sumY / ink;
    }
  }
  return out;
}

/// Distribusi tinta 16 kolom + 8 baris, dinormalisasi agar berjumlah 1.
Float32List projectionFeatures(Uint8List img) {
  const int n = P.charSize;
  final Float32List out =
      Float32List(P.projectionCols + P.projectionRows);

  final int colWidth = n ~/ P.projectionCols;
  final int rowHeight = n ~/ P.projectionRows;

  double colTotal = 0;
  for (int c = 0; c < P.projectionCols; c++) {
    double sum = 0;
    for (int y = 0; y < n; y++) {
      final int row = y * n;
      for (int x = c * colWidth; x < (c + 1) * colWidth; x++) {
        sum += img[row + x] / 255.0;
      }
    }
    out[c] = sum;
    colTotal += sum;
  }
  for (int c = 0; c < P.projectionCols; c++) {
    out[c] = out[c] / (colTotal + _eps);
  }

  double rowTotal = 0;
  for (int r = 0; r < P.projectionRows; r++) {
    double sum = 0;
    for (int y = r * rowHeight; y < (r + 1) * rowHeight; y++) {
      final int row = y * n;
      for (int x = 0; x < n; x++) {
        sum += img[row + x] / 255.0;
      }
    }
    out[P.projectionCols + r] = sum;
    rowTotal += sum;
  }
  for (int r = 0; r < P.projectionRows; r++) {
    out[P.projectionCols + r] =
        out[P.projectionCols + r] / (rowTotal + _eps);
  }
  return out;
}

/// Fitur topologi 8-d: jumlah lubang, luas relatif, dan posisinya.
Float32List holeFeatures(Uint8List img) {
  const int n = P.charSize;
  final Float32List feats = Float32List(P.holeDim);

  // Binerkan + bingkai kertas 1 px. Semua piksel awalnya kertas, lalu
  // piksel tinta di dalamnya dikosongkan. Bingkai ikut kertas supaya
  // flood-fill dari luar benar-benar bisa mencapai tepi.
  const int pn = n + 2;
  final Uint8List paper = Uint8List(pn * pn);
  paper.fillRange(0, paper.length, 1);
  int inkTotal = 0;
  for (int y = 0; y < n; y++) {
    for (int x = 0; x < n; x++) {
      if (img[y * n + x] > 127) {
        inkTotal++;
        paper[(y + 1) * pn + (x + 1)] = 0;
      }
    }
  }
  if (inkTotal == 0) return feats;

  // Label daerah kertas dengan 4-konektivitas.
  final Int32List labels = Int32List(pn * pn);
  int nextLabel = 0;
  final Int32List stack = Int32List(pn * pn);
  for (int start = 0; start < paper.length; start++) {
    if (paper[start] == 0 || labels[start] != 0) continue;
    nextLabel++;
    int sp = 0;
    stack[sp++] = start;
    labels[start] = nextLabel;
    while (sp > 0) {
      final int index = stack[--sp];
      final int x = index % pn;
      final int y = index ~/ pn;
      // 4-konektivitas.
      if (x > 0) {
        final int nb = index - 1;
        if (paper[nb] != 0 && labels[nb] == 0) {
          labels[nb] = nextLabel;
          stack[sp++] = nb;
        }
      }
      if (x + 1 < pn) {
        final int nb = index + 1;
        if (paper[nb] != 0 && labels[nb] == 0) {
          labels[nb] = nextLabel;
          stack[sp++] = nb;
        }
      }
      if (y > 0) {
        final int nb = index - pn;
        if (paper[nb] != 0 && labels[nb] == 0) {
          labels[nb] = nextLabel;
          stack[sp++] = nb;
        }
      }
      if (y + 1 < pn) {
        final int nb = index + pn;
        if (paper[nb] != 0 && labels[nb] == 0) {
          labels[nb] = nextLabel;
          stack[sp++] = nb;
        }
      }
    }
  }

  // Label yang menyentuh bingkai luar = latar.
  final Uint8List outside = Uint8List(nextLabel + 1);
  for (int x = 0; x < pn; x++) {
    outside[labels[x]] = 1;
    outside[labels[(pn - 1) * pn + x]] = 1;
  }
  for (int y = 0; y < pn; y++) {
    outside[labels[y * pn]] = 1;
    outside[labels[y * pn + pn - 1]] = 1;
  }

  final List<List<double>> holes = <List<double>>[];
  final Int32List area = Int32List(nextLabel + 1);
  final Float64List sumX = Float64List(nextLabel + 1);
  final Float64List sumY = Float64List(nextLabel + 1);
  for (int y = 0; y < pn; y++) {
    for (int x = 0; x < pn; x++) {
      final int label = labels[y * pn + x];
      if (label == 0 || outside[label] != 0) continue;
      area[label]++;
      sumX[label] += x;
      sumY[label] += y;
    }
  }
  for (int label = 1; label <= nextLabel; label++) {
    if (area[label] == 0) continue;
    holes.add(<double>[
      area[label].toDouble(),
      sumX[label] / area[label],
      sumY[label] / area[label],
    ]);
  }
  holes.sort((List<double> a, List<double> b) => b[0].compareTo(a[0]));

  feats[0] = math.min(holes.length, P.holeMax) / P.holeMax;
  for (int i = 0; i < P.holeMax; i++) {
    if (i < holes.length) {
      feats[1 + i] = holes[i][0] / inkTotal;
    }
  }
  for (int i = 0; i < 2; i++) {
    final int base = 1 + P.holeMax + 2 * i;
    if (i < holes.length) {
      // Sama seperti Python: centroid diukur pada citra **berbingkai** lalu
      // dibagi (charSize-1). Bingkai 1 px sengaja tidak dikurangi.
      feats[base] = holes[i][1] / (n - 1);
      feats[base + 1] = holes[i][2] / (n - 1);
    }
  }
  return feats;
}

/// Vektor fitur 1844-d untuk satu karakter 32x32.
Float32List extractFeatures(Uint8List charImage) {
  final Float32List hog = hogFeatures(charImage);
  final Float32List zoning = zoningFeatures(charImage);
  final Float32List projection = projectionFeatures(charImage);
  final Float32List holes = holeFeatures(charImage);

  final Float32List out = Float32List(P.featureDim);
  int pos = 0;
  out.setRange(pos, pos + hog.length, hog);
  pos += hog.length;
  out.setRange(pos, pos + zoning.length, zoning);
  pos += zoning.length;
  out.setRange(pos, pos + projection.length, projection);
  pos += projection.length;
  out.setRange(pos, pos + holes.length, holes);
  return out;
}
