import 'dart:math' as math;

import 'classical_config.dart';
import 'connected_components.dart';
import 'gray_image.dart';

/// Segmentasi baris dan karakter pada mask tinta.

// ---------------------------------------------------------------------------
// Utilitas run-length
// ---------------------------------------------------------------------------

/// Ubah array boolean menjadi daftar rentang (start, end) inklusif.
List<({int start, int end})> _runs(List<bool> mask) {
  final List<({int start, int end})> out = <({int start, int end})>[];
  int? start;
  for (int i = 0; i < mask.length; i++) {
    if (mask[i] && start == null) {
      start = i;
    } else if (!mask[i] && start != null) {
      out.add((start: start, end: i - 1));
      start = null;
    }
  }
  if (start != null) {
    out.add((start: start, end: mask.length - 1));
  }
  return out;
}

/// Gabungkan celah antar rentang yang lebih kecil atau sama dengan [maxGap].
List<({int start, int end})> _mergeGaps(
  List<({int start, int end})> runs,
  int maxGap,
) {
  if (runs.isEmpty) return runs;
  final List<({int start, int end})> merged = <({int start, int end})>[
    runs.first,
  ];
  for (int i = 1; i < runs.length; i++) {
    final ({int start, int end}) last = merged.last;
    final ({int start, int end}) cur = runs[i];
    if (cur.start - last.end - 1 <= maxGap) {
      merged[merged.length - 1] = (start: last.start, end: cur.end);
    } else {
      merged.add(cur);
    }
  }
  return merged;
}

/// Otsu 1 dimensi untuk profil baris.
///
/// Dipakai memisahkan baris berisi teks dari baseline noise pada foto
/// struk, di mana ambang absolut tidak pernah tercapai.
double _otsu1d(List<int> values) {
  int maxValue = 0;
  for (final int v in values) {
    if (v > maxValue) maxValue = v;
  }
  if (maxValue == 0) return 0;

  // Skala ke 0..255 seperti np.clip(values, 0, 255).
  final List<int> hist = List<int>.filled(256, 0);
  for (final int v in values) {
    hist[v.clamp(0, 255)]++;
  }

  double total = 0;
  for (final int h in hist) {
    total += h;
  }
  if (total <= 0) return 0;

  double sumAll = 0;
  for (int i = 0; i < 256; i++) {
    sumAll += i * hist[i];
  }

  double weightBg = 0;
  double sumBg = 0;
  double bestVar = -1;
  double bestT = 0;

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
    if (variance > bestVar) {
      bestVar = variance;
      bestT = t.toDouble();
    }
  }
  return bestT;
}

// ---------------------------------------------------------------------------
// Baris
// ---------------------------------------------------------------------------

/// Kembalikan daftar (top, bottom) tiap baris teks.
///
/// Dua mode dipilih otomatis dari bentuk profil baris:
///
/// - **Citra bersih** (hasil render): celah antarbaris benar-benar nol,
///   sehingga ambang absolut sudah tepat dan paling stabil.
/// - **Foto struk**: tekstur kertas dan bayangan menghasilkan tinta tipis
///   di hampir semua baris. Ambang absolut tidak akan pernah tercapai,
///   jadi dipakai Otsu 1D untuk memisahkan puncak teks dari baseline.
///
/// Pembeda: rasio baris kosong. Citra bersih selalu punya celah putih;
/// foto struk hampir tidak punya.
List<({int top, int bottom})> segmentLines(
  GrayImage ink, {
  double minInkRatio = kLineMinInkRatio,
  int gapMerge = kLineGapMerge,
  int minHeight = kLineMinHeight,
}) {
  final int w = ink.width;
  final int h = ink.height;
  if (w == 0 || h == 0) return const <({int top, int bottom})>[];

  final List<int> profile = List<int>.filled(h, 0);
  bool anyInk = false;
  for (int y = 0; y < h; y++) {
    int count = 0;
    final int rowBase = y * w;
    for (int x = 0; x < w; x++) {
      if (ink.data[rowBase + x] != 0) count++;
    }
    profile[y] = count;
    if (count > 0) anyInk = true;
  }
  if (!anyInk) return const <({int top, int bottom})>[];

  final double absolute = math.max(1.0, w * minInkRatio);

  int emptyRows = 0;
  for (final int v in profile) {
    if (v == 0) emptyRows++;
  }
  final double emptyRatio = emptyRows / h;
  final bool isPhoto = emptyRatio < kLinePhotoEmptyRatio;

  final double threshold = isPhoto
      ? math.max(_otsu1d(profile), absolute)
      : absolute;

  final List<bool> mask = List<bool>.generate(
    h,
    (int y) => profile[y] >= threshold,
    growable: false,
  );

  List<({int start, int end})> runs = _runs(mask);
  runs = _mergeGaps(runs, gapMerge);

  return <({int top, int bottom})>[
    for (final ({int start, int end}) r in runs)
      if (r.end - r.start + 1 >= minHeight)
        (top: r.start, bottom: r.end),
  ];
}

// ---------------------------------------------------------------------------
// Morfologi
// ---------------------------------------------------------------------------

/// Closing kotak k x k untuk menyambung bagian huruf yang terputus.
///
/// Closing = dilasi lalu erosi. Nilai di luar batas citra diperlakukan
/// seperti OpenCV: 0 untuk dilasi (tepi tidak tumbuh) dan 255 untuk
/// erosi (tepi tidak menyusut).
GrayImage closeInk(GrayImage ink, int k) {
  if (k <= 1) {
    return ink.clone();
  }
  final GrayImage dilated = _dilate(ink, k);
  return _erode(dilated, k);
}

GrayImage _dilate(GrayImage src, int k) {
  final int w = src.width;
  final int h = src.height;
  final int r = k ~/ 2;
  final GrayImage out = GrayImage.zeros(w, h);

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      bool hit = false;
      for (int dy = -r; dy <= r && !hit; dy++) {
        final int ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        final int rowBase = ny * w;
        for (int dx = -r; dx <= r; dx++) {
          final int nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          if (src.data[rowBase + nx] != 0) {
            hit = true;
            break;
          }
        }
      }
      out.data[y * w + x] = hit ? 255 : 0;
    }
  }
  return out;
}

GrayImage _erode(GrayImage src, int k) {
  final int w = src.width;
  final int h = src.height;
  final int r = k ~/ 2;
  final GrayImage out = GrayImage.zeros(w, h);

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      bool all = true;
      for (int dy = -r; dy <= r && all; dy++) {
        final int ny = y + dy;
        // Di luar batas dianggap tinta agar tepi tidak menyusut.
        if (ny < 0 || ny >= h) continue;
        final int rowBase = ny * w;
        for (int dx = -r; dx <= r; dx++) {
          final int nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          if (src.data[rowBase + nx] == 0) {
            all = false;
            break;
          }
        }
      }
      out.data[y * w + x] = all ? 255 : 0;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Komponen & penggabungan
// ---------------------------------------------------------------------------

/// Gabungkan komponen yang tumpang tindih horizontal (bagian huruf sama).
List<InkComponent> _mergeOverlapping(
  List<InkComponent> comps, {
  double overlap = 0.5,
}) {
  final List<InkComponent> merged = <InkComponent>[];
  for (final InkComponent c in comps) {
    if (merged.isNotEmpty) {
      final InkComponent p = merged.last;
      final int lo = math.max(p.left, c.left);
      final int hi = math.min(p.right, c.right);
      final int ov = math.max(0, hi - lo + 1);
      final int narrower = math.min(p.width, c.width);
      if (narrower > 0 && ov >= overlap * narrower) {
        p.right = math.max(p.right, c.right);
        p.top = math.min(p.top, c.top);
        p.bottom = math.max(p.bottom, c.bottom);
        p.area += c.area;
        continue;
      }
    }
    merged.add(
      InkComponent(
        left: c.left,
        top: c.top,
        right: c.right,
        bottom: c.bottom,
        area: c.area,
        label: c.label,
      ),
    );
  }
  return merged;
}

/// Pecah bbox lebar pada lembah proyeksi vertikal.
List<({int left, int top, int right, int bottom})> _splitWide(
  List<InkComponent> boxes,
  GrayImage ink,
  double targetWidth,
) {
  final List<({int left, int top, int right, int bottom})> out =
      <({int left, int top, int right, int bottom})>[];

  for (final InkComponent b in boxes) {
    final int boxW = b.width;
    if (boxW <= targetWidth * kCharSplitRatio) {
      out.add((left: b.left, top: b.top, right: b.right, bottom: b.bottom));
      continue;
    }

    // Proyeksi vertikal di dalam bbox.
    final List<int> proj = List<int>.filled(boxW, 0);
    for (int y = b.top; y <= b.bottom; y++) {
      final int rowBase = y * ink.width;
      for (int x = 0; x < boxW; x++) {
        if (ink.data[rowBase + b.left + x] != 0) proj[x]++;
      }
    }

    int parts = math.max(2, (boxW / targetWidth).round());
    parts = math.min(parts, 6);

    final List<int> cuts = <int>[];
    for (int k = 1; k < parts; k++) {
      final int ideal = (boxW * k / parts).round();
      final int margin = math.max(2, (targetWidth * 0.35).round());
      final int lo = math.max(1, ideal - margin);
      final int hi = math.min(boxW - 1, ideal + margin);
      if (hi <= lo) {
        cuts.add(ideal);
        continue;
      }
      int bestIdx = lo;
      int bestVal = proj[lo];
      for (int i = lo + 1; i <= hi; i++) {
        if (proj[i] < bestVal) {
          bestVal = proj[i];
          bestIdx = i;
        }
      }
      cuts.add(bestIdx);
    }

    final List<int> bounds = <int>[0, ...cuts, boxW];
    for (int i = 0; i < bounds.length - 1; i++) {
      final int a = bounds[i];
      final int bb = bounds[i + 1];
      if (bb - a >= kCharMinWidth) {
        out.add((
          left: b.left + a,
          top: b.top,
          right: b.left + bb - 1,
          bottom: b.bottom,
        ));
      }
    }
  }
  return out;
}

double _median(List<int> values) {
  if (values.isEmpty) return 8.0;
  final List<int> sorted = List<int>.of(values)..sort();
  final int n = sorted.length;
  if (n.isOdd) return sorted[n ~/ 2].toDouble();
  return (sorted[n ~/ 2 - 1] + sorted[n ~/ 2]) / 2.0;
}

// ---------------------------------------------------------------------------
// Karakter
// ---------------------------------------------------------------------------

/// Kembalikan bbox (left, top, right, bottom) tiap karakter pada satu baris.
///
/// Closing adaptif: stroke tipis (font thermal) sering terpecah menjadi
/// beberapa komponen setelah binarisasi. Bila jumlah komponen jauh
/// melebihi perkiraan jumlah huruf, closing 2x2 diterapkan untuk
/// menyambungnya kembali. Pada citra yang sudah utuh, closing tidak
/// dipakai agar huruf tidak saling menempel.
List<({int left, int top, int right, int bottom})> segmentCharacters(
  GrayImage ink,
  ({int top, int bottom}) lineBox, {
  int minWidth = kCharMinWidth,
  int minInk = kCharMinInk,
  int closeKernel = kCharCloseKernel,
}) {
  final int h = ink.height;
  final int w = ink.width;

  final int top = math.max(0, lineBox.top);
  final int bottom = math.min(h - 1, lineBox.bottom);
  if (bottom <= top) return const <({int left, int top, int right, int bottom})>[];

  GrayImage strip = ink.crop(0, top, w - 1, bottom);
  if (!_hasInk(strip)) {
    return const <({int left, int top, int right, int bottom})>[];
  }

  List<({int left, int top, int right, int bottom})> candidates(
    GrayImage stripImg,
  ) {
    final List<InkComponent> comps =
        connectedComponents(stripImg, minArea: math.max(minInk, 8));
    if (comps.isEmpty) {
      return const <({int left, int top, int right, int bottom})>[];
    }
    final List<InkComponent> merged = _mergeOverlapping(comps);
    final double target = math.max(
      _median(<int>[for (final InkComponent c in merged) c.width]),
      3.0,
    );
    return _splitWide(merged, stripImg, target);
  }

  List<({int left, int top, int right, int bottom})> boxes =
      candidates(strip);

  // Perkiraan jumlah huruf dari lebar tinta dibagi lebar karakter tipikal.
  int minCol = w;
  int maxCol = -1;
  for (int x = 0; x < strip.width; x++) {
    bool hasInk = false;
    for (int y = 0; y < strip.height; y++) {
      if (strip.data[y * strip.width + x] != 0) {
        hasInk = true;
        break;
      }
    }
    if (hasInk) {
      if (x < minCol) minCol = x;
      if (x > maxCol) maxCol = x;
    }
  }

  final int lineHeight = bottom - top + 1;
  if (maxCol >= minCol && lineHeight > 0) {
    final int span = maxCol - minCol + 1;
    final double expected =
        math.max(1, span / math.max(lineHeight * 0.55, 4.0));
    if (boxes.length > expected * kCharOversplitRatio && closeKernel > 1) {
      final GrayImage closed = closeInk(strip, closeKernel);
      final List<({int left, int top, int right, int bottom})> alt =
          candidates(closed);
      if (alt.isNotEmpty && alt.length < boxes.length) {
        strip = closed;
        boxes = alt;
      }
    }
  }

  final List<({int left, int top, int right, int bottom})> result =
      <({int left, int top, int right, int bottom})>[];

  for (final ({int left, int top, int right, int bottom}) b in boxes) {
    if (b.right - b.left + 1 < minWidth) continue;

    // Batas vertikal rapat mengikuti tinta sebenarnya.
    int firstRow = -1;
    int lastRow = -1;
    int inkCount = 0;
    for (int y = b.top; y <= b.bottom; y++) {
      final int rowBase = y * strip.width;
      for (int x = b.left; x <= b.right; x++) {
        if (strip.data[rowBase + x] != 0) {
          inkCount++;
          if (firstRow < 0) firstRow = y;
          lastRow = y;
        }
      }
    }
    if (firstRow < 0 || inkCount < minInk) continue;

    result.add((
      left: b.left,
      top: top + firstRow,
      right: b.right,
      bottom: top + lastRow,
    ));
  }

  result.sort((
    ({int left, int top, int right, int bottom}) a,
    ({int left, int top, int right, int bottom}) b,
  ) =>
      a.left.compareTo(b.left));

  return result;
}

bool _hasInk(GrayImage img) {
  for (final int v in img.data) {
    if (v != 0) return true;
  }
  return false;
}
