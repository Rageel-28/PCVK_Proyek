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

/// Bersihkan kata: huruf besar, hanya karakter yang relevan.
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
  Iterable<String> _candidates(String word, int maxDist) sync* {
    if (word.isEmpty) return;
    final String first = word[0];
    for (int len = (word.length - maxDist).clamp(1, 1 << 30);
        len <= word.length + maxDist;
        len++) {
      final List<String>? bucket = _byLength[len];
      if (bucket == null) continue;
      if (maxDist <= 1) {
        for (final String w in bucket) {
          if (w[0] == first) yield w;
        }
      } else {
        yield* bucket;
      }
    }
  }

  /// Kembalikan (kata terbaik, jarak, apakah dikoreksi).
  ({String word, int distance, bool changed}) correct(
    String word, {
    int? maxDist,
  }) {
    final String w = _clean(word);
    if (w.length < 2) {
      return (word: word, distance: 0, changed: false);
    }

    final int limit = maxDist ?? (w.length ~/ 3).clamp(1, 3);

    if (_words.contains(w)) {
      return (word: w, distance: 0, changed: false);
    }

    String best = w;
    int bestDist = limit + 1;

    for (final String cand in _candidates(w, limit)) {
      if (cand == w) continue;
      final int d = levenshtein(w, cand, maxDist: limit);
      if (d < bestDist) {
        best = cand;
        bestDist = d;
        if (d == 1) break;
      }
    }

    if (bestDist <= limit) {
      return (word: best, distance: bestDist, changed: true);
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
