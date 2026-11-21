/// Leksikon kata dengan Levenshtein berbobot biaya konfusi.
///
/// Port dari `ML/ocr/lexicon.py`.
library;

import 'dart:convert';

/// Jarak edit dengan biaya substitusi yang bisa dikustomisasi.
double weightedLevenshtein(
  String a,
  String b, {
  Map<String, double>? costs,
}) {
  final Map<String, double> table = costs ?? const <String, double>{};
  if (a == b) return 0.0;
  if (a.isEmpty) return b.length.toDouble();
  if (b.isEmpty) return a.length.toDouble();

  final int la = a.length;
  final int lb = b.length;
  List<double> previous = List<double>.generate(lb + 1, (int j) => j.toDouble());
  List<double> current = List<double>.filled(lb + 1, 0);

  for (int i = 1; i <= la; i++) {
    current[0] = i.toDouble();
    final String ca = a[i - 1];
    for (int j = 1; j <= lb; j++) {
      final String cb = b[j - 1];
      final double cost = ca == cb
          ? 0.0
          : (table['$ca$cb'] ?? table['$cb$ca'] ?? 1.0);
      final double delete = previous[j] + 1.0;
      final double insert = current[j - 1] + 1.0;
      final double substitute = previous[j - 1] + cost;
      double best = delete;
      if (insert < best) best = insert;
      if (substitute < best) best = substitute;
      current[j] = best;
    }
    final List<double> swap = previous;
    previous = current;
    current = swap;
  }
  return previous[lb];
}

class Lexicon {
  Lexicon(List<String> words, {Map<String, double>? costs})
      : words = (words.where((String w) => w.isNotEmpty).toSet().toList()
          ..sort()),
        costs = costs ?? const <String, double>{} {
    for (final String word in this.words) {
      _byLength.putIfAbsent(word.length, () => <String>[]).add(word);
    }
    _set = this.words.toSet();
  }

  final List<String> words;
  final Map<String, double> costs;
  final Map<int, List<String>> _byLength = <int, List<String>>{};
  late final Set<String> _set;

  int get size => words.length;

  bool contains(String word) => _set.contains(word);

  /// Kata terdekat dalam jarak [maxDistance].
  ///
  /// [predicate] menyaring kandidat (mis. hanya kata dengan 5 digit).
  String? closest(
    String word,
    double maxDistance, {
    bool Function(String candidate)? predicate,
    double Function(String candidate)? distanceOf,
  }) {
    if (word.isEmpty) return null;

    String? bestWord;
    double bestDistance = double.infinity;
    final int span = maxDistance.toInt() + 1;
    for (int length = word.length - span;
        length <= word.length + span;
        length++) {
      final List<String>? candidates = _byLength[length];
      if (candidates == null) continue;
      for (final String candidate in candidates) {
        if (predicate != null && !predicate(candidate)) continue;
        final double distance = distanceOf != null
            ? distanceOf(candidate)
            : weightedLevenshtein(word, candidate, costs: costs);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestWord = candidate;
          if (distance == 0.0) return bestWord;
        }
      }
    }
    if (bestDistance > maxDistance) return null;
    return bestWord;
  }

  static Lexicon fromJsonString(String source) {
    final Map<String, dynamic> payload =
        jsonDecode(source) as Map<String, dynamic>;
    final List<String> words = (payload['words'] as List<dynamic>)
        .map((dynamic w) => w as String)
        .toList();
    final Map<String, double> costs = <String, double>{};
    final List<dynamic> rawCosts =
        (payload['costs'] as List<dynamic>?) ?? <dynamic>[];
    for (final dynamic entry in rawCosts) {
      final List<dynamic> pair = entry as List<dynamic>;
      costs['${pair[0]}${pair[1]}'] = (pair[2] as num).toDouble();
    }
    return Lexicon(words, costs: costs);
  }
}
