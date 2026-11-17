import 'dart:math' as math;

import 'char_normalizer.dart' show pythonRound;
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
  runs = <({int start, int end})>[
    for (final ({int start, int end}) r in runs)
      if (r.end - r.start + 1 >= minHeight) r,
  ];

  // Pecah ulang blok tinggi yang menyatu akibat noise tepi kertas.
  final int tallLimit = math.max(80, (0.15 * h).floor());
  if (!isPhoto && runs.any((r) => (r.end - r.start + 1) > tallLimit)) {
    final List<({int start, int end})> refined = <({int start, int end})>[];
    for (final ({int start, int end}) r in runs) {
      if ((r.end - r.start + 1) <= tallLimit) {
        refined.add(r);
        continue;
      }
      final List<int> subProf = profile.sublist(r.start, r.end + 1);
      final double subT = math.max(_otsu1d(subProf) * 0.65, absolute);
      final List<bool> subMask = List<bool>.generate(
        subProf.length,
        (int i) => subProf[i] >= subT,
        growable: false,
      );
      List<({int start, int end})> subRuns = _mergeGaps(_runs(subMask), gapMerge);
      subRuns = <({int start, int end})>[
        for (final ({int start, int end}) sr in subRuns)
          if (sr.end - sr.start + 1 >= minHeight)
            (start: r.start + sr.start, end: r.start + sr.end),
      ];
      if (subRuns.isNotEmpty) {
        refined.addAll(subRuns);
      } else {
        refined.add(r);
      }
    }
    runs = refined;
  }

  return <({int top, int bottom})>[
    for (final ({int start, int end}) r in runs)
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

    int parts = math.max(2, pythonRound(boxW / targetWidth));
    parts = math.min(parts, 10);

    final List<int> cuts = <int>[];
    for (int k = 1; k < parts; k++) {
      final int ideal = pythonRound(boxW * k / parts);
      final int margin = math.max(2, pythonRound((boxW / parts) * 0.22));
      final int prevCut = cuts.isEmpty ? 1 : cuts.last + kCharMinWidth;
      final int lo = math.max(prevCut, ideal - margin);
      final int hi = math.min(boxW - 1, ideal + margin);
      if (hi <= lo) {
        if (ideal >= prevCut && ideal < boxW) {
          cuts.add(ideal);
        }
        continue;
      }
      int bestIdx = lo;
      int bestCost = proj[lo] * 10 + (lo - ideal).abs() * 3;
      for (int i = lo + 1; i <= hi; i++) {
        final int cost = proj[i] * 10 + (i - ideal).abs() * 3;
        if (cost < bestCost) {
          bestCost = cost;
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

  final int lineHeight = bottom - top + 1;

  List<({int left, int top, int right, int bottom})> candidates(
    GrayImage stripImg,
  ) {
    final List<InkComponent> comps =
        connectedComponents(stripImg, minArea: math.max(minInk, 8));
    if (comps.isEmpty) {
      return const <({int left, int top, int right, int bottom})>[];
    }
    final List<InkComponent> merged = _mergeOverlapping(comps);
    final double rawMedW =
        _median(<int>[for (final InkComponent c in merged) c.width]);
    final double rawMedH =
        _median(<int>[for (final InkComponent c in merged) c.height]);
    final bool isTextLine = lineHeight >= 14 &&
        rawMedH >= math.max(10.0, lineHeight * 0.55) &&
        (rawMedW / math.max(rawMedH, 1.0)) <= 1.8;
    final double medW = isTextLine
        ? math.min(rawMedW, math.max(lineHeight * 0.68, 3.0))
        : rawMedW;
    final double target = math.max(
      math.max(medW, lineHeight * 0.48),
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

  if (maxCol >= minCol && lineHeight > 0) {
    final int span = maxCol - minCol + 1;
    final double expected =
        math.max(1, span / math.max(lineHeight * 0.55, 4.0));
    final int effK =
        (closeKernel > 1 && lineHeight >= 40) ? math.max(closeKernel, 3) : closeKernel;
    if (boxes.length > expected * kCharOversplitRatio && effK > 1) {
      final GrayImage closed = closeInk(strip, effK);
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

  return _mergeFragments(result, lineHeight);
}

/// Gabungkan pecahan glyph akibat cetakan thermal yang pucat.
///
/// Pada foto struk asli, stroke yang hilang membuat satu huruf terpecah
/// menjadi dua komponen bersebelahan, mis. `M` menjadi dua batang tegak
/// dan `0` menjadi dua lengkung. Tanpa penggabungan, satu huruf
/// menghasilkan dua karakter palsu dan CER melonjak.
///
/// Dua aturan, keduanya aman karena tidak mungkin terjadi pada dua huruf
/// berbeda:
///
/// 1. Kotak yang **tumpang tindih** horizontal (celah <= 0) pasti satu
///    glyph, sebab dua huruf tidak bisa menempati kolom piksel yang sama.
/// 2. Pecahan **sempit** (lebar < 45% median) yang menempel (celah <= 8 px)
///    digabung, asalkan hasil gabungannya masih selebar satu huruf
///    (<= 1.25x median). Gabungan dua huruf asli lebarnya ~2x median.
List<({int left, int top, int right, int bottom})> _mergeFragments(
  List<({int left, int top, int right, int bottom})> boxes,
  int lineHeight,
) {
  if (boxes.length < 2) return boxes;

  final double medianW =
      _median(<int>[for (final ({int left, int top, int right, int bottom}) b in boxes) b.right - b.left + 1]);
  final double refW = math.max(medianW, lineHeight * 0.45);
  final double narrow = math.max(2.0, refW * kCharFragmentNarrowRatio);
  final double maxMergedW = refW * kCharFragmentMaxWidthRatio;

  final List<({int left, int top, int right, int bottom})> merged =
      <({int left, int top, int right, int bottom})>[boxes.first];

  for (int i = 1; i < boxes.length; i++) {
    final ({int left, int top, int right, int bottom}) box = boxes[i];
    final ({int left, int top, int right, int bottom}) prev = merged.last;
    final int gap = box.left - prev.right;
    final int wBox = box.right - box.left + 1;
    final int wPrev = prev.right - prev.left + 1;
    final int mergedW = math.max(prev.right, box.right) - prev.left + 1;

    final bool shouldMerge = gap <= 0 ||
        (gap <= kCharFragmentMaxGap &&
            mergedW <= maxMergedW &&
            (wBox < narrow || wPrev < narrow));

    if (shouldMerge) {
      merged[merged.length - 1] = (
        left: prev.left,
        top: math.min(prev.top, box.top),
        right: math.max(prev.right, box.right),
        bottom: math.max(prev.bottom, box.bottom),
      );
    } else {
      merged.add(box);
    }
  }
  return merged;
}

bool _hasInk(GrayImage img) {
  for (final int v in img.data) {
    if (v != 0) return true;
  }
  return false;
}
