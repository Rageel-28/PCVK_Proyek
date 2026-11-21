/// Model bahasa karakter (n-gram dengan interpolasi).
///
/// Port dari `ML/ocr/lm.py`. Bobot dibaca dari `lm.json`.
library;

import 'dart:convert';

const String kBOS = '\u0002';
const String kEOS = '\u0003';

class CharLM {
  CharLM(
    this._ngrams,
    this._unigram, {
    required this.order,
    required this.floor,
  });

  final int order;
  final Map<String, Map<String, double>> _ngrams;
  final Map<String, double> _unigram;
  final double floor;

  static CharLM fromJsonString(String source) {
    final Map<String, dynamic> payload =
        jsonDecode(source) as Map<String, dynamic>;

    final Map<String, Map<String, double>> ngrams =
        <String, Map<String, double>>{};
    final Map<String, dynamic> rawNgrams =
        payload['ngrams'] as Map<String, dynamic>;
    rawNgrams.forEach((String context, dynamic table) {
      final Map<String, double> converted = <String, double>{};
      (table as Map<String, dynamic>).forEach((String ch, dynamic value) {
        converted[ch] = (value as num).toDouble();
      });
      ngrams[context] = converted;
    });

    final Map<String, double> unigram = <String, double>{};
    (payload['unigram'] as Map<String, dynamic>).forEach(
      (String ch, dynamic value) {
        unigram[ch] = (value as num).toDouble();
      },
    );

    return CharLM(
      ngrams,
      unigram,
      order: payload['order'] as int,
      floor: (payload['floor'] as num).toDouble(),
    );
  }

  /// Log-probabilitas [char] setelah [context] (hanya konteks akhir).
  double nextCharLogprobs(String context, String char) {
    for (int n = order; n > 1; n--) {
      final int length = n - 1;
      String key;
      if (context.length < length) {
        key = kBOS * (length - context.length) + context;
      } else {
        key = context.substring(context.length - length);
      }
      final Map<String, double>? table = _ngrams[key];
      if (table != null && table.containsKey(char)) {
        return table[char]!;
      }
    }
    return _unigram[char] ?? floor;
  }

  /// Total log-probabilitas teks.
  double logprob(String text) {
    String context = kBOS * (order - 1);
    double total = 0;
    for (final String ch in (text + kEOS).split('')) {
      total += nextCharLogprobs(context, ch);
      context = (context + ch);
      if (context.length > order - 1) {
        context = context.substring(context.length - (order - 1));
      }
    }
    return total;
  }
}
