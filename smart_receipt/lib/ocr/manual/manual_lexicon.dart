import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;

import 'manual_config.dart';

/// Satu token sebelum dan sesudah koreksi.
class ManualCorrection {
  const ManualCorrection(this.original, this.corrected, this.rule);

  final String original;
  final String corrected;
  final String rule;

  bool get changed => original != corrected;
}

// Huruf yang sering salah dibaca sebagai digit, dan sebaliknya.
// Disalin persis dari `TO_DIGIT` / `TO_LETTER` di lexical_correct.py.
const Map<String, String> _toDigit = <String, String>{
  'O': '0', 'o': '0', 'D': '0', 'Q': '0',
  'I': '1', 'l': '1', 'i': '1', '|': '1', '!': '1', ']': '1', '[': '1',
  'Z': '2', 'z': '2',
  'E': '3',
  'A': '4',
  'S': '5', 's': '5',
  'G': '6', 'b': '6',
  'T': '7',
  'B': '8',
  'g': '9', 'q': '9',
};

const Map<String, String> _toLetter = <String, String>{
  '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A',
  '5': 'S', '6': 'G', '7': 'T', '8': 'B',
};

bool _isDigitCode(int c) => c >= 48 && c <= 57;
bool _isUpper(int c) => c >= 65 && c <= 90;
bool _isLower(int c) => c >= 97 && c <= 122;
bool _isAlnumChar(String ch) {
  final int c = ch.codeUnitAt(0);
  return _isDigitCode(c) || _isUpper(c) || _isLower(c);
}

bool _isDigitChar(String ch) => _isDigitCode(ch.codeUnitAt(0));

List<String> _chars(String s) => s.split('');

double _digitRatio(String token) {
  final List<String> alnum = <String>[
    for (final String c in _chars(token))
      if (_isAlnumChar(c)) c,
  ];
  if (alnum.isEmpty) return 0.0;
  int digits = 0;
  for (final String c in alnum) {
    if (_isDigitChar(c)) digits++;
  }
  return digits / alnum.length;
}

final RegExp _moneyPattern = RegExp(r'^\d{1,3}(?:[.,]\d{3})+$');

/// Token berpemisah yang, bila huruf mirip digit diperbaiki, membentuk angka.
bool _looksLikeMoney(String token) {
  if (!token.contains('.') && !token.contains(',')) return false;
  final StringBuffer fixed = StringBuffer();
  for (final String c in _chars(token)) {
    fixed.write(_toDigit[c] ?? c);
  }
  if (!_moneyPattern.hasMatch(fixed.toString())) return false;
  // Harus mengandung minimal satu digit asli, supaya `O.O` tidak dianggap uang.
  for (final String c in _chars(token)) {
    if (_isDigitChar(c)) return true;
  }
  return false;
}

/// Perbaiki satu token berdasarkan komposisinya.
///
/// [threshold] adalah proporsi digit yang menentukan token dianggap angka.
/// Token campuran seimbang (mis. `N0`, rasio 0,5) sengaja dibiarkan:
/// menebak di situ lebih sering merusak daripada memperbaiki.
ManualCorrection correctToken(String token, {double threshold = 0.6}) {
  if (token.isEmpty) return ManualCorrection(token, token, '');

  // 1. Bentuk uang lebih dulu: pola pemisah ribuan adalah sinyal kuat.
  if (_looksLikeMoney(token)) {
    final String out = _chars(token).map((String c) => _toDigit[c] ?? c).join();
    if (out != token) return ManualCorrection(token, out, 'uang');
    return ManualCorrection(token, token, '');
  }

  final double ratio = _digitRatio(token);
  if (ratio >= threshold) {
    final String out = _chars(token).map((String c) => _toDigit[c] ?? c).join();
    if (out != token) return ManualCorrection(token, out, 'angka');
    return ManualCorrection(token, token, '');
  }

  if (ratio <= 1.0 - threshold) {
    final String out =
        _chars(token).map((String c) => _toLetter[c] ?? c).join();
    if (out != token) return ManualCorrection(token, out, 'kata');
    return ManualCorrection(token, token, '');
  }

  return ManualCorrection(token, token, 'ambigu');
}

/// Selesaikan token campuran huruf+digit memakai kamus.
///
/// Token seperti `N0` tidak bisa diputuskan dari komposisinya (rasio digit
/// tepat 0,5). Kedua tafsiran diuji terhadap kamus, dan yang cocok dipilih.
ManualCorrection resolveAmbiguous(String token, List<String> vocabulary) {
  if (vocabulary.isEmpty) return ManualCorrection(token, token, 'ambigu');

  final String asDigit =
      _chars(token).map((String c) => _toDigit[c] ?? c).join();
  final String asLetter =
      _chars(token).map((String c) => _toLetter[c] ?? c).join();

  final bool letterKnown = vocabulary.contains(asLetter);
  final bool digitKnown = vocabulary.contains(asDigit);
  if (letterKnown && !digitKnown) {
    return ManualCorrection(token, asLetter, 'ambigu->kata');
  }
  if (digitKnown && !letterKnown) {
    return ManualCorrection(token, asDigit, 'ambigu->angka');
  }
  return ManualCorrection(token, token, 'ambigu');
}

/// Jarak Levenshtein klasik dengan dynamic programming.
int editDistance(String a, String b) {
  if (a.isEmpty) return b.length;
  if (b.isEmpty) return a.length;

  List<int> prev = List<int>.generate(b.length + 1, (int i) => i);
  List<int> cur = List<int>.filled(b.length + 1, 0);

  for (int i = 1; i <= a.length; i++) {
    cur[0] = i;
    final int ca = a.codeUnitAt(i - 1);
    for (int j = 1; j <= b.length; j++) {
      final int cost = ca == b.codeUnitAt(j - 1) ? 0 : 1;
      final int deletion = prev[j] + 1;
      final int insertion = cur[j - 1] + 1;
      final int substitution = prev[j - 1] + cost;
      int best = deletion < insertion ? deletion : insertion;
      if (substitution < best) best = substitution;
      cur[j] = best;
    }
    final List<int> swap = prev;
    prev = cur;
    cur = swap;
  }
  return prev[b.length];
}

/// Ganti token dengan kata kamus terdekat bila cukup mirip.
///
/// [maxDistance] sengaja kecil (1). Jarak lebih besar cenderung
/// "memperbaiki" token yang sebenarnya benar menjadi kata lain yang
/// kebetulan mirip — versi tanpa batas ini menurunkan akurasi 14-22 poin.
///
/// Token dipecah pada titik/koma dulu, supaya `ALFAMIDIKU.C8M` bisa
/// dicocokkan per bagian.
///
/// [vocabulary] harus terurut. Bila ada beberapa kandidat berjarak sama,
/// yang pertama dalam urutan itulah yang dipakai; pengurutan membuat hasil
/// deterministik (versi Python mengiterasi `set`, yang urutannya arbitrer
/// dan bisa memberi kata berbeda pada proses berbeda).
ManualCorrection correctWithDictionary(
  String token,
  List<String> vocabulary, {
  int maxDistance = 1,
}) {
  if (token.isEmpty || vocabulary.isEmpty) {
    return ManualCorrection(token, token, '');
  }

  // setara re.split(r"([.,])", token): pemisah ikut menjadi elemen.
  final List<String> parts = <String>[];
  final StringBuffer buffer = StringBuffer();
  for (final String c in _chars(token)) {
    if (c == '.' || c == ',') {
      parts.add(buffer.toString());
      buffer.clear();
      parts.add(c);
    } else {
      buffer.write(c);
    }
  }
  parts.add(buffer.toString());

  bool changed = false;
  final List<String> out = <String>[];
  for (final String part in parts) {
    if (part.isEmpty || part == '.' || part == ',') {
      out.add(part);
      continue;
    }
    if (part.length < 3 || vocabulary.contains(part)) {
      out.add(part);
      continue;
    }

    String? best;
    int bestDistance = maxDistance + 1;
    for (final String word in vocabulary) {
      if ((word.length - part.length).abs() > maxDistance) continue;
      final int d = editDistance(part, word);
      if (d < bestDistance) {
        bestDistance = d;
        best = word;
        if (d == 1) break;
      }
    }

    if (best != null && bestDistance <= maxDistance) {
      out.add(best);
      changed = true;
    } else {
      out.add(part);
    }
  }

  if (changed) {
    return ManualCorrection(token, out.join(), 'kamus');
  }
  return ManualCorrection(token, token, '');
}

/// Koreksi satu baris: uang/angka/kata per token, lalu kamus.
///
/// Kamus dijalankan terakhir karena ia yang paling berisiko mengubah
/// token yang sebenarnya sudah benar.
({String text, List<ManualCorrection> changes}) correctManualLine(
  String text,
  List<String> vocabulary,
) {
  final List<ManualCorrection> changes = <ManualCorrection>[];
  final List<String> outTokens = <String>[];

  // Python `text.split(" ")` mempertahankan token kosong; join " " juga.
  for (final String raw in text.split(' ')) {
    if (raw.isEmpty) {
      outTokens.add(raw);
      continue;
    }

    String lead = '';
    String trail = '';
    String core = raw;
    while (core.isNotEmpty && !_isAlnumChar(core[0])) {
      lead += core[0];
      core = core.substring(1);
    }
    while (core.isNotEmpty && !_isAlnumChar(core[core.length - 1])) {
      trail = core[core.length - 1] + trail;
      core = core.substring(0, core.length - 1);
    }
    if (core.isEmpty) {
      outTokens.add(raw);
      continue;
    }

    ManualCorrection step = correctToken(core);
    if (step.changed) {
      changes.add(step);
    } else if (step.rule == 'ambigu' && vocabulary.isNotEmpty) {
      final ManualCorrection resolved = resolveAmbiguous(core, vocabulary);
      if (resolved.changed) {
        changes.add(resolved);
        step = resolved;
      }
    }

    String finalWord = step.corrected;
    if (vocabulary.isNotEmpty) {
      final ManualCorrection dictStep =
          correctWithDictionary(finalWord, vocabulary);
      if (dictStep.changed) {
        changes.add(dictStep);
        finalWord = dictStep.corrected;
      }
    }

    outTokens.add(lead + finalWord + trail);
  }

  return (text: outTokens.join(' '), changes: changes);
}

/// Kamus kata untuk koreksi leksikal OCR manual.
class ManualLexicon {
  ManualLexicon();

  final Set<String> _words = <String>{};

  /// Kata terurut. Dipakai untuk iterasi kandidat agar deterministik.
  List<String> _ordered = const <String>[];
  bool _loaded = false;

  bool get isLoaded => _loaded;
  int get wordCount => _words.length;

  /// Kosakata terurut (untuk pengujian dan diagnosis).
  List<String> get orderedVocabulary => List<String>.unmodifiable(_ordered);

  Future<void> load({String assetPath = kManualLexiconAsset}) async {
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
      final String upper = w.toUpperCase();
      if (upper.isEmpty) continue;
      _words.add(upper);
    }
    _ordered = _words.toList()..sort();
    _loaded = true;
  }

  /// Koreksi satu baris teks.
  ({String text, List<ManualCorrection> changes}) correctLine(String text) =>
      correctManualLine(text, _ordered);

  /// Koreksi teks multi-baris. Kembalikan (teks, jumlah token terkoreksi).
  ({String text, int fixes}) correctText(String text) {
    int fixes = 0;
    final List<String> lines = <String>[];
    for (final String line in text.split('\n')) {
      final ({String text, List<ManualCorrection> changes}) result =
          correctManualLine(line, _ordered);
      fixes += result.changes.length;
      lines.add(result.text);
    }
    return (text: lines.join('\n'), fixes: fixes);
  }
}
