import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;

import 'classical_config.dart';

/// Koreksi kata hasil OCR memakai leksikon produk.
///
/// Prinsipnya sederhana namun sangat efektif: hasil OCR yang salah
/// biasanya masih berjarak edit kecil dari kata yang benar
/// ("IND0M1E" -> "INDOMIE"). Dengan kamus nama produk + kata umum struk,
/// kesalahan karakter bisa dipulihkan menjadi kata yang benar.
///
/// Levenshtein diimplementasikan manual dengan dynamic programming agar
/// tidak bergantung pada library apa pun.

/// Jarak edit dengan pemotongan dini bila melebihi [maxDist].
int levenshtein(String a, String b, {int? maxDist}) {
  if (a == b) return 0;
  final int la = a.length;
  final int lb = b.length;
  if (la == 0) return lb;
  if (lb == 0) return la;
  if (maxDist != null && (la - lb).abs() > maxDist) {
    return maxDist + 1;
  }

  List<int> prev = List<int>.generate(lb + 1, (int i) => i);
  List<int> cur = List<int>.filled(lb + 1, 0);

  for (int i = 1; i <= la; i++) {
    cur[0] = i;
    int rowMin = cur[0];
    final int ca = a.codeUnitAt(i - 1);
    for (int j = 1; j <= lb; j++) {
      final int cost = ca == b.codeUnitAt(j - 1) ? 0 : 1;
      final int deletion = prev[j] + 1;
      final int insertion = cur[j - 1] + 1;
      final int substitution = prev[j - 1] + cost;
      int best = deletion < insertion ? deletion : insertion;
      if (substitution < best) best = substitution;
      cur[j] = best;
      if (best < rowMin) rowMin = best;
    }
    if (maxDist != null && rowMin > maxDist) {
      return maxDist + 1;
    }
    final List<int> swap = prev;
    prev = cur;
    cur = swap;
  }
  return prev[lb];
}

/// Bersihkan kata kamus: huruf besar, hanya karakter yang relevan.
String _clean(String word) {
  final String upper = word.toUpperCase();
  final StringBuffer sb = StringBuffer();
  for (int i = 0; i < upper.length; i++) {
    final String ch = upper[i];
    final int code = ch.codeUnitAt(0);
    final bool isDigit = code >= 48 && code <= 57;
    final bool isUpper = code >= 65 && code <= 90;
    final bool isSymbol = '.&/+-'.contains(ch);
    if (isDigit || isUpper || isSymbol) {
      sb.write(ch);
    }
  }
  return sb.toString();
}

/// Bersihkan token OCR dengan mempertahankan koma/@ yang mungkin salah baca.
String _cleanToken(String word) {
  final String upper = word.toUpperCase();
  final StringBuffer sb = StringBuffer();
  for (int i = 0; i < upper.length; i++) {
    final String ch = upper[i];
    final int code = ch.codeUnitAt(0);
    final bool isDigit = code >= 48 && code <= 57;
    final bool isUpper = code >= 65 && code <= 90;
    final bool isSymbol = '.,@&/-'.contains(ch);
    if (isDigit || isUpper || isSymbol) {
      sb.write(ch);
    }
  }
  return sb.toString();
}

/// Pasangan karakter yang sering tertukar secara visual pada cetakan thermal.
const Set<String> _kConfusionPairs = <String>{
  '0O', 'O0', 'CO', 'OC', 'OQ', 'QO', 'DO', 'OD', '.O', 'O.', ',O', 'O,', 'GO', 'OG',
  '1I', 'I1', 'IL', 'LI', 'IT', 'TI', 'LT', 'TL', 'IJ', 'JI', 'EL', 'LE',
  'MN', 'NM', 'AM', 'MA', 'MW', 'WM', 'HM', 'MH', 'MR', 'RM', 'IM', 'MI', '.M', 'M.', ',M', 'M,',
  'AN', 'NA', 'HN', 'NH', 'NU', 'UN',
  '2Z', 'Z2', 'GZ', 'ZG', 'AZ', 'ZA', 'EZ', 'ZE', '&Z', 'Z&', '7Z', 'Z7',
  '@E', 'E@', '3E', 'E3', 'EF', 'FE', 'BE', 'EB',
  '8B', 'B8', 'BH', 'HB', 'BR', 'RB',
  'AD', 'DA', 'DS', 'SD', '0D', 'D0',
  'UV', 'VU', 'AU', 'UA', '4U', 'U4', ',U', 'U,',
  ',Y', 'Y,', 'VY', 'YV', 'TY', 'YT',
  '5S', 'S5', '6G', 'G6', '7T', 'T7',
  '4A', 'A4', 'FP', 'PF', 'KR', 'RK',
};

int _subCost(String a, String b) {
  if (a == b) return 0;
  return _kConfusionPairs.contains('$a$b') ? 9 : 20;
}

/// Jarak edit berbobot kesamaan visual OCR (skala x20, identik dengan Python).
int _weightedLevenshtein(String a, String b, int maxCost) {
  final int la = a.length;
  final int lb = b.length;
  if ((la - lb).abs() > 1) return maxCost + 1;
  if (la <= 6 && la != lb) return maxCost + 1;

  List<int> prev = List<int>.generate(lb + 1, (int i) => i * 23);
  List<int> cur = List<int>.filled(lb + 1, 0);

  for (int i = 1; i <= la; i++) {
    cur[0] = i * 23;
    int rowMin = cur[0];
    final String ca = a[i - 1];
    for (int j = 1; j <= lb; j++) {
      final int cost = _subCost(ca, b[j - 1]);
      final int del = prev[j] + 23;
      final int ins = cur[j - 1] + 23;
      final int sub = prev[j - 1] + cost;
      int best = del < ins ? del : ins;
      if (sub < best) best = sub;
      cur[j] = best;
      if (best < rowMin) rowMin = best;
    }
    if (rowMin > maxCost) return maxCost + 1;
    final List<int> swap = prev;
    prev = cur;
    cur = swap;
  }
  return prev[lb];
}

/// Kamus kata untuk koreksi OCR.
class LexiconCorrector {
  LexiconCorrector();

  final Set<String> _words = <String>{};
  final Map<int, List<String>> _byLength = <int, List<String>>{};
  bool _loaded = false;

  bool get isLoaded => _loaded;
  int get wordCount => _words.length;

  Future<void> load({String assetPath = kLexiconAsset}) async {
    if (_loaded) return;
    final String raw = await rootBundle.loadString(assetPath);
    loadFromJsonString(raw);
  }

  void loadFromJsonString(String rawJson) {
    final Map<String, dynamic> json = jsonDecode(rawJson) as Map<String, dynamic>;
    final Object? words = json['words'];
    if (words is! List) {
      throw const FormatException('Leksikon harus punya array "words".');
    }
    addAll(words.map((Object? e) => e.toString()));
  }

  void addAll(Iterable<String> words) {
    for (final String w in words) {
      final String clean = _clean(w);
      if (clean.length >= 2) {
        _words.add(clean);
      }
    }
    _rebuild();
    _loaded = true;
  }

  void _rebuild() {
    _byLength.clear();
    for (final String w in _words) {
      _byLength.putIfAbsent(w.length, () => <String>[]).add(w);
    }
    for (final List<String> bucket in _byLength.values) {
      bucket.sort();
    }
  }

  /// Kandidat dengan panjang berbeda maksimal [maxDist].
  ///
  /// Bila jarak kecil (<= 1), hanya kata berhuruf awal sama yang diuji.
  /// Untuk jarak lebih besar, semua kata pada panjang tersebut ikut
  /// karena huruf pertama pun bisa salah baca.
  Iterable<String> _candidates(String word, int maxLenDiff) sync* {
    if (word.isEmpty) return;
    final int minLen = (word.length - maxLenDiff).clamp(kLexiconMinLength, 1 << 30);
    final int maxLen = word.length + maxLenDiff;
    for (int len = minLen; len <= maxLen; len++) {
      final List<String>? bucket = _byLength[len];
      if (bucket != null) {
        yield* bucket;
      }
    }
  }

  /// Kembalikan (kata terbaik, jarak, apakah dikoreksi).
  ///
  /// Koreksi konservatif memakai biaya edit berbobot kesamaan OCR:
  /// token berangka, terlalu pendek (< [kLexiconMinLength]), atau mengandung
  /// < 2 huruf dibiarkan; kata <= 6 huruf tidak boleh berubah panjangnya;
  /// pemenang harus unggul jelas dari kandidat runner-up.
  ({String word, int distance, bool changed}) correct(
    String word, {
    int? maxDist,
  }) {
    final String w = _cleanToken(word);
    if (w.length < kLexiconMinLength) {
      return (word: word, distance: 0, changed: false);
    }
    if (RegExp('[0-9]').hasMatch(w)) {
      return (word: word, distance: 0, changed: false);
    }
    int alphaCount = 0;
    for (int i = 0; i < w.length; i++) {
      final int code = w.codeUnitAt(i);
      if (code >= 65 && code <= 90) alphaCount++;
    }
    if (alphaCount < 2) {
      return (word: word, distance: 0, changed: false);
    }

    if (_words.contains(w)) {
      return (word: w, distance: 0, changed: false);
    }

    final int limit = maxDist != null
        ? maxDist * 20
        : (w.length <= 5 ? 13 : (w.length <= 7 ? 24 : (w.length <= 9 ? 29 : 33)));
    final int maxLenDiff = w.length <= 6 ? 0 : 1;

    int bestCost = limit + 1;
    int secondCost = limit + 1;
    String? bestWord;

    for (final String cand in _candidates(w, maxLenDiff)) {
      if (cand == w) continue;
      final int cost = _weightedLevenshtein(w, cand, limit);
      if (cost <= limit) {
        if (cost < bestCost) {
          secondCost = bestCost;
          bestCost = cost;
          bestWord = cand;
        } else if (cost < secondCost) {
          secondCost = cost;
        }
      }
    }

    if (bestWord != null && (secondCost > limit || (secondCost - bestCost) >= 6)) {
      return (word: bestWord, distance: (bestCost + 19) ~/ 20, changed: true);
    }
    return (word: word, distance: 0, changed: false);
  }

  /// Koreksi per kata. Kembalikan (teks, jumlah kata terkoreksi).
  ({String text, int fixes}) correctText(String text, {int? maxDist}) {
    final List<String> out = <String>[];
    int fixes = 0;
    for (final String token in text.split(RegExp(r'\s+'))) {
      if (token.isEmpty) continue;
      final ({String word, int distance, bool changed}) r =
          correct(token, maxDist: maxDist);
      if (r.changed) fixes++;
      out.add(r.word);
    }
    return (text: out.join(' '), fixes: fixes);
  }
}
