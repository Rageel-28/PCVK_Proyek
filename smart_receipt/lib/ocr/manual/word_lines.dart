import 'dart:math' as math;

import 'manual_config.dart';
import 'manual_preprocess.dart';

/// Satu karakter hasil prediksi beserta kotaknya.
///
/// Lebar/tinggi (bukan `right`/`bottom`) dipakai agar perhitungan celah
/// antar karakter sama persis dengan versi Python, yang memakai
/// `left + width` untuk tepi kanan.
typedef ManualChar = ({
  String char,
  double confidence,
  int top,
  int left,
  int height,
  int width,
});

/// Satu kelompok kata pada sebuah baris.
typedef ManualWord = ({
  String text,
  int top,
  int left,
  int right,
  int bottom,
});

int _gapBetween(ManualChar previous, ManualChar current) =>
    current.left - (previous.left + previous.width);

/// Ambang celah untuk batas kata: 2x median celah pada baris itu.
///
/// Celah di dalam kata dan celah antar kata saling tumpang tindih,
/// sehingga ambang tetap (mis. "55% lebar huruf") tidak bisa memisahkan
/// keduanya. Median celah baris jauh lebih stabil.
double manualSpaceThreshold(List<ManualChar> chars) {
  if (chars.length < 2) return 2.0;
  final List<int> gaps = <int>[
    for (int i = 1; i < chars.length; i++) _gapBetween(chars[i - 1], chars[i]),
  ];
  return math.max(2.0, medianOf(gaps) * kManualSpaceGapRatio);
}

/// Rekonstruksi teks satu baris dengan menyisipkan spasi di celah lebar.
String manualJoinWithSpaces(List<ManualChar> chars) {
  if (chars.isEmpty) return '';
  if (chars.length == 1) return chars.first.char;

  final double threshold = manualSpaceThreshold(chars);
  final StringBuffer buffer = StringBuffer()..write(chars.first.char);
  for (int i = 1; i < chars.length; i++) {
    if (_gapBetween(chars[i - 1], chars[i]) > threshold) {
      buffer.write(' ');
    }
    buffer.write(chars[i].char);
  }
  return buffer.toString();
}

/// Pecah satu baris menjadi kelompok kata beserta kotaknya.
///
/// `receipt_parser` mengelompokkan `OcrLine` menurut `centerY` lalu
/// memakai `left`/`right` untuk memisahkan kolom nama barang dan harga.
/// Karena itu tiap kelompok kata harus dikirim terpisah; bila seluruh
/// baris dikirim sebagai satu `OcrLine`, kolom harga hilang.
List<ManualWord> manualSplitWords(List<ManualChar> chars) {
  if (chars.isEmpty) return const <ManualWord>[];

  final double threshold = manualSpaceThreshold(chars);
  final List<ManualWord> words = <ManualWord>[];

  final StringBuffer current = StringBuffer();
  int startLeft = chars.first.left;
  int endRight = chars.first.left + chars.first.width;
  int top = chars.first.top;
  int bottom = chars.first.top + chars.first.height;

  void flush() {
    final String text = current.toString().trim();
    if (text.isNotEmpty) {
      words.add((
        text: text,
        top: top,
        left: startLeft,
        right: endRight,
        bottom: bottom,
      ));
    }
    current.clear();
  }

  current.write(chars.first.char);
  for (int i = 1; i < chars.length; i++) {
    final ManualChar c = chars[i];
    if (_gapBetween(chars[i - 1], c) > threshold) {
      flush();
      startLeft = c.left;
      top = c.top;
      bottom = c.top + c.height;
    }
    current.write(c.char);
    endRight = c.left + c.width;
    if (c.top < top) top = c.top;
    if (c.top + c.height > bottom) bottom = c.top + c.height;
  }
  flush();

  return words;
}
