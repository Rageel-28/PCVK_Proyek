/// Segmentasi glyph (karakter) di dalam satu kata.
///
/// Port dari `ML/ocr/chars.py`: connected components → gabung komponen
/// bertumpuk → pecah komponen lebar secara rekursif.
library;

import 'dart:math' as math;

import 'gray_image.dart';
import 'params.dart';
import 'preprocess.dart' show GlyphComponent, connectedComponents;

/// Kotak satu glyph di koordinat citra penuh.
class Glyph {
  const Glyph(this.left, this.top, this.right, this.bottom);

  final int left;
  final int top;
  final int right;
  final int bottom;

  int get width => right - left;
  int get height => bottom - top;

  @override
  String toString() => 'Glyph($left,$top,$right,$bottom)';
}

double _median(List<int> values) {
  if (values.isEmpty) return 0.0;
  final List<int> sorted = List<int>.from(values)..sort();
  final int mid = sorted.length ~/ 2;
  if (sorted.length.isOdd) return sorted[mid].toDouble();
  return (sorted[mid - 1] + sorted[mid]) / 2.0;
}

/// Connected components 8-konektivitas pada pita biner.
List<Glyph> components(BinaryImage band) {
  return connectedComponents(band)
      .map((GlyphComponent c) => Glyph(c.left, c.top, c.right, c.bottom))
      .toList();
}

/// Rasio tumpang-tindih vertikal terhadap tinggi yang lebih pendek.
double _verticalOverlap(Glyph a, Glyph b) {
  final int top = math.max(a.top, b.top);
  final int bottom = math.min(a.bottom, b.bottom);
  final int overlap = math.max(0, bottom - top);
  final int shortest = math.min(a.height, b.height);
  return shortest == 0 ? 0.0 : overlap / shortest;
}

Glyph _union(Glyph a, Glyph b) => Glyph(
      math.min(a.left, b.left),
      math.min(a.top, b.top),
      math.max(a.right, b.right),
      math.max(a.bottom, b.bottom),
    );

/// Gabung komponen yang sebenarnya satu glyph.
List<Glyph> mergeMarks(
  List<Glyph> glyphs,
  int gapLimit,
  double medianWidth, {
  double medianHeight = 0.0,
}) {
  if (glyphs.isEmpty) return <Glyph>[];
  final List<Glyph> ordered = List<Glyph>.from(glyphs)
    ..sort((Glyph a, Glyph b) => a.left.compareTo(b.left));
  final double limit = medianWidth > 0 ? medianWidth * 1.15 : double.infinity;

  final List<Glyph> merged = <Glyph>[ordered[0]];
  for (int i = 1; i < ordered.length; i++) {
    final Glyph glyph = ordered[i];
    final Glyph prev = merged.last;
    final int gap = glyph.left - prev.right;
    final int unionWidth =
        math.max(prev.right, glyph.right) - math.min(prev.left, glyph.left);

    // Komponen yang seluruhnya berada di dalam rentang-x tetangganya pasti
    // bagian dari huruf yang sama (lubang `0` yang terpisah, titik `i`).
    final bool contained = glyph.left >= prev.left && glyph.right <= prev.right;
    final bool touching = gap <= 0 &&
        _verticalOverlap(prev, glyph) > 0.25 &&
        (contained || unionWidth <= limit);

    if (touching || _stackedMark(prev, glyph, medianHeight)) {
      merged[merged.length - 1] = _union(prev, glyph);
    } else {
      merged.add(glyph);
    }
  }
  return merged;
}

/// True bila satu komponen menumpuk tepat di atas/bawah komponen lain.
bool _stackedMark(Glyph a, Glyph b, double medianHeight) {
  if (medianHeight <= 0) return false;

  final int left = math.max(a.left, b.left);
  final int right = math.min(a.right, b.right);
  final int overlap = right - left;
  if (overlap <= 0) return false;

  final int smaller = math.min(a.width, b.width);
  if (smaller <= 0 || overlap / smaller < 0.6) return false;

  final int verticalGap =
      math.max(b.top - a.bottom, a.top - b.bottom);
  if (verticalGap > math.max(1.0, 0.12 * medianHeight)) return false;

  final int unionHeight =
      math.max(a.bottom, b.bottom) - math.min(a.top, b.top);
  final int tallest = math.max(a.height, b.height);
  return unionHeight <= 1.6 * tallest + verticalGap;
}

/// Pecah glyph lebar di lembah profil vertikalnya.
List<Glyph> splitWideGlyph(
  BinaryImage band,
  Glyph glyph, {
  double referenceWidth = 0.0,
  double valleyRatio = 0.25,
}) {
  if (glyph.width < 6 || glyph.height < 4) return <Glyph>[glyph];

  final int gw = glyph.width;
  final int gh = glyph.height;
  if (gw <= 0 || gh <= 0) return <Glyph>[glyph];

  final List<double> profile = List<double>.filled(gw, 0);
  for (int y = glyph.top; y < glyph.bottom; y++) {
    final int row = y * band.width;
    for (int x = glyph.left; x < glyph.right; x++) {
      if (band.data[row + x] != 0) profile[x - glyph.left] += 1;
    }
  }
  if (gw < 6) return <Glyph>[glyph];

  final int margin = gw ~/ 5 < 1 ? 1 : gw ~/ 5;
  final int innerStart = margin;
  final int innerEnd = gw - margin;
  if (innerEnd <= innerStart) return <Glyph>[glyph];

  int cut = innerStart;
  double lowest = profile[innerStart];
  for (int i = innerStart + 1; i < innerEnd; i++) {
    if (profile[i] < lowest) {
      lowest = profile[i];
      cut = i;
    }
  }

  double maxProfile = 0;
  for (final double v in profile) {
    if (v > maxProfile) maxProfile = v;
  }

  final bool hasValley = lowest <= valleyRatio * maxProfile &&
      cut >= 0.15 * gw &&
      (gw - cut) >= 0.15 * gw;

  if (!hasValley) return _splitEvenly(glyph, referenceWidth);

  final Glyph leftPart =
      Glyph(glyph.left, glyph.top, glyph.left + cut, glyph.bottom);
  final Glyph rightPart =
      Glyph(glyph.left + cut, glyph.top, glyph.right, glyph.bottom);

  final List<Glyph> parts = <Glyph>[];
  final Glyph? leftTight = _tighten(band, leftPart);
  if (leftTight != null) parts.add(leftTight);
  final Glyph? rightTight = _tighten(band, rightPart);
  if (rightTight != null) parts.add(rightTight);
  return parts.length == 2 ? parts : <Glyph>[glyph];
}

/// Bagi glyph lebar menjadi bagian selebar acuan.
List<Glyph> _splitEvenly(Glyph glyph, double referenceWidth) {
  if (referenceWidth <= 0) return <Glyph>[glyph];

  final double ratio = glyph.width / referenceWidth;
  final int count = ratio.round();
  if (count < 2) return <Glyph>[glyph];
  if ((ratio - count).abs() > 0.3) return <Glyph>[glyph];

  final double step = glyph.width / count;
  final List<Glyph> parts = <Glyph>[];
  for (int i = 0; i < count; i++) {
    final int left = glyph.left + (i * step).round();
    final int right = glyph.left + ((i + 1) * step).round();
    if (right - left < 2) return <Glyph>[glyph];
    parts.add(Glyph(left, glyph.top, right, glyph.bottom));
  }
  return parts;
}

/// Perketat kotak ke bbox tinta sebenarnya di dalamnya.
Glyph? _tighten(BinaryImage band, Glyph glyph) {
  int left = glyph.right;
  int right = glyph.left;
  int top = glyph.bottom;
  int bottom = glyph.top;
  for (int y = glyph.top; y < glyph.bottom; y++) {
    final int row = y * band.width;
    for (int x = glyph.left; x < glyph.right; x++) {
      if (band.data[row + x] != 0) {
        if (x < left) left = x;
        if (x + 1 > right) right = x + 1;
        if (y < top) top = y;
        if (y + 1 > bottom) bottom = y + 1;
      }
    }
  }
  if (right <= left || bottom <= top) return null;
  return Glyph(left, top, right, bottom);
}

/// Pecah glyph lebar berulang sampai semua bagian selebar glyph biasa.
List<Glyph> _splitToWidth(
  BinaryImage band,
  Glyph glyph,
  double referenceWidth,
  double splitRatio, {
  int depth = 0,
}) {
  if (referenceWidth <= 0 || depth >= 6) return <Glyph>[glyph];
  if (glyph.width <= referenceWidth * splitRatio) return <Glyph>[glyph];

  // Komponen yang jauh lebih lebar pasti berisi beberapa huruf, jadi
  // lembahnya boleh dangkal. Yang hanya sedikit lebar bisa satu huruf lebar.
  final double multiple = glyph.width / referenceWidth;
  final double valleyRatio = multiple >= 2.5 ? 0.55 : 0.25;

  final List<Glyph> parts =
      splitWideGlyph(band, glyph, referenceWidth: referenceWidth, valleyRatio: valleyRatio);
  if (parts.length <= 1) return <Glyph>[glyph];

  final List<Glyph> out = <Glyph>[];
  for (final Glyph part in parts) {
    out.addAll(_splitToWidth(
      band,
      part,
      referenceWidth,
      splitRatio,
      depth: depth + 1,
    ));
  }
  return out;
}

/// Segmentasi glyph dalam satu pita kata/baris.
List<Glyph> segmentGlyphs(
  BinaryImage band, {
  int bandTop = 0,
  int bandLeft = 0,
  double? splitRatio,
  double referenceWidth = 0.0,
  double referenceHeight = 0.0,
}) {
  final double ratio = splitRatio ?? P.glyphSplitRatio;
  if (band.width == 0 || band.height == 0 || !band.hasInk) return <Glyph>[];

  List<Glyph> comps = components(band);
  if (comps.isEmpty) return <Glyph>[];

  final List<int> heights = comps.map((Glyph g) => g.height).toList();
  final List<int> widths = comps.map((Glyph g) => g.width).toList();
  final double medianHeight = _median(heights);
  final double medianWidth = _median(widths);

  // Tinggi acuan sebaiknya dari seluruh baris, bukan kata ini saja: pada
  // kata yang hurufnya terpecah dua (mis. ``2`` jadi dua goresan), median
  // tinggi kata itu setengah tinggi huruf sebenarnya dan titik-titik pada
  // ``i``/``:`` tidak lagi tergabung.
  final double heightRef =
      referenceHeight > 0 ? referenceHeight : medianHeight;
  final int gapLimit = (heightRef * 0.12).round() < 1
      ? 1
      : (heightRef * 0.12).round();
  comps = mergeMarks(comps, gapLimit, medianWidth, medianHeight: heightRef);

  double reference = referenceWidth;
  if (reference <= 0) {
    reference = comps.length >= 3 ? medianWidth : 0.6 * medianHeight;
  }

  final List<Glyph> glyphs = <Glyph>[];
  for (final Glyph glyph in comps) {
    glyphs.addAll(_splitToWidth(band, glyph, reference, ratio));
  }

  glyphs.sort((Glyph a, Glyph b) => a.left.compareTo(b.left));
  return glyphs
      .map((Glyph g) =>
          Glyph(g.left + bandLeft, g.top + bandTop, g.right + bandLeft, g.bottom + bandTop))
      .toList();
}
