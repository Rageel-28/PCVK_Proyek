/// Decoder: beam search + model bahasa + koreksi leksikon.
///
/// Port dari `ML/ocr/decode.py`.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'lexicon.dart';
import 'lm.dart';
import 'params.dart';

const double _eps = 1e-12;

final RegExp _spaceBeforePunct = RegExp(r'\s+([.,:])');
final RegExp _spaceAfterPunctBeforeDigit =
    RegExp(r'(\d)([.,:])\s+(\d{3})(?!\d)');

/// Rapikan spasi di sekitar tanda baca angka (`8 . 700` → `8.700`).
String normalizeSpacing(String text) {
  if (text.isEmpty) return text;
  String out = text.replaceAllMapped(
    _spaceBeforePunct,
    (Match m) => m.group(1)!,
  );
  out = out.replaceAllMapped(
    _spaceAfterPunctBeforeDigit,
    (Match m) => '${m.group(1)}${m.group(2)}${m.group(3)}',
  );
  return out;
}

/// Rentetan garis pemisah pada struk (`------`) sering menempel pada baris
/// teks: `------ LAYANAN KONSUMEN ------`. Garis seperti itu bukan isi struk,
/// jadi dibuang; baris yang seluruhnya garis menjadi kosong.
final RegExp _ruleRun = RegExp(r'[-_=]{4,}');

String stripRuleArtifacts(String text) {
  if (text.isEmpty) return text;
  String cleaned =
      text.replaceAll(_ruleRun, ' ').split(RegExp(r'\s+')).where((String s) => s.isNotEmpty).join(' ');
  if (!cleaned.split('').any((String ch) => RegExp(r'[A-Za-z0-9]').hasMatch(ch))) {
    return '';
  }
  return cleaned;
}

const Set<String> _numberPunct = <String>{'.', ',', ':', '/', '-', '%'};
bool _isDigit(String ch) => ch.codeUnitAt(0) >= 48 && ch.codeUnitAt(0) <= 57;

int _digitCount(String token) =>
    token.split('').where(_isDigit).length;

bool isNumericToken(String token) {
  if (token.isEmpty) return false;
  if (_digitCount(token) == 0) return false;
  return token
      .split('')
      .every((String ch) => _isDigit(ch) || _numberPunct.contains(ch));
}

bool isNumericIsh(String token) {
  if (token.isEmpty) return false;
  final int digits = _digitCount(token);
  if (digits < 2) {
    // Satu digit pun tetap jelas angka bila bentuknya harga berpemisah
    // ribuan, mis. `1.OOO` yang sebenarnya `1.000`.
    return _priceShapeRe.hasMatch(token);
  }
  return digits / token.length >= 0.3;
}

/// Cari rangkaian karakter dengan skor gabungan tertinggi.
String beamDecode(
  List<Float64List> probs,
  CharLM lm, {
  int beamWidth = P.beamWidth,
  double lmWeight = P.lmWeight,
  int? candidateWidth,
  String prefix = '',
}) {
  if (probs.isEmpty) return '';
  final int candidate = candidateWidth ?? math.max(beamWidth, 4);
  final int contextLen = math.max(1, lm.order - 1);
  String startContext = kBOS * contextLen + prefix;
  if (startContext.length > contextLen) {
    startContext = startContext.substring(startContext.length - contextLen);
  }

  List<_Beam> beams = <_Beam>[_Beam(0.0, '', startContext)];

  for (int position = 0; position < probs.length; position++) {
    final Float64List row = probs[position];

    // Indeks kandidat teratas.
    final List<int> order = List<int>.generate(row.length, (int i) => i)
      ..sort((int a, int b) {
        final int cmp = row[b].compareTo(row[a]);
        return cmp != 0 ? cmp : a.compareTo(b);
      });
    List<int> top = order.take(candidate).toList();

    // Kalau classifier sangat yakin, model bahasa tidak boleh membantah.
    if (row[top[0]] >= P.lmTrustThreshold) {
      top = top.sublist(0, 1);
    }

    final List<_Beam> candidates = <_Beam>[];
    for (final _Beam beam in beams) {
      for (final int classIndex in top) {
        final String ch = P.charset[classIndex];
        final double acoustic = math.log(row[classIndex] + _eps);
        final double language = lm.nextCharLogprobs(beam.context, ch);
        final double total = beam.score + acoustic + lmWeight * language;
        String nextContext = beam.context + ch;
        if (nextContext.length > contextLen) {
          nextContext = nextContext.substring(nextContext.length - contextLen);
        }
        candidates.add(_Beam(total, beam.text + ch, nextContext));
      }
    }

    candidates.sort((_Beam a, _Beam b) => b.score.compareTo(a.score));
    beams = candidates.take(beamWidth).toList();
  }
  return beams[0].text;
}

class _Beam {
  _Beam(this.score, this.text, this.context);

  final double score;
  final String text;
  final String context;
}

const Map<String, String> _digitConfusion = <String, String>{
  'O': '0', 'o': '0', 'Q': '0', 'D': '0',
  'I': '1', 'l': '1', 'i': '1', '|': '1',
  'Z': '2', 'z': '2',
  'A': '4',
  'S': '5', 's': '5',
  'G': '6', 'b': '6',
  'T': '7',
  'B': '8',
  'g': '9', 'q': '9',
};

final RegExp _priceRe = RegExp(r'^\d{1,3}(?:\.\d{3})+(?:,\d{2})?$');
final RegExp _plainDigitsRe = RegExp(r'^\d{3,}$');

/// Bentuk harga berpemisah ribuan yang masih memuat `O`/`o` sebagai pengganti
/// `0`, mis. `1.OOO` (`1.000`).
final RegExp _priceShapeRe = RegExp(r'^\d+[.,][0-9Oo]{3}$');

/// Bentuk harga yang lebih longgar: semua karakter boleh huruf yang mirip
/// angka. Penjaga agar `L.OO0` (`1.000`) tidak "diperbaiki" menjadi `L.OOO`
/// oleh aturan kata.
final RegExp _loosePriceRe =
    RegExp(r'^[\dOoIlSsZzBbGgTt]+[.,][\dOoIlSsZzBbGgTt]{3}$');

/// Kata kunci yang menandakan token berikutnya adalah nominal.
const Set<String> _amountContext = <String>{
  'DISKON', 'TOTAL', 'TUNAI', 'KEMBALI', 'HEMAT', 'BAYAR', 'JUAL',
  'HARGA', 'SUBTOTAL', 'PPN', 'PPNBM', 'BELANJA', 'SISA', 'DPP', 'ITEM',
  'POTONGAN', 'CASH', 'DEBIT', 'QRIS', 'KARTU', 'TAGIHAN',
  'PEMBULATAN',
};

/// Huruf yang tersisa di kolom nominal hampir pasti angka salah baca.
/// `G` dipetakan ke `0` (bukan `6` seperti `_digitConfusion`) karena harga
/// bulat berakhiran nol jauh lebih sering: `DISKON 50G` adalah `DISKON 500`.
const Map<String, String> _amountLetter = <String, String>{
  'O': '0', 'o': '0', 'Q': '0', 'D': '0', 'G': '0', 'g': '0',
  'I': '1', 'l': '1', 'L': '1', '|': '1',
  'Z': '2', 'z': '2',
  'S': '5', 's': '5',
  'B': '8',
};

/// Baca `token` sebagai nominal bila bentuknya memungkinkan.
String? _amountValue(String token) {
  final List<String> chars = token.split('');
  if (chars.isNotEmpty && _amountLetter.containsKey(chars.last)) {
    chars[chars.length - 1] = _amountLetter[chars.last]!;
  }
  final StringBuffer buffer = StringBuffer();
  for (final String ch in chars) {
    buffer.write(_digitConfusion[ch] ?? _amountLetter[ch] ?? ch);
  }
  String mapped = buffer.toString();
  mapped = mapped.replaceAll(RegExp(r'^[.,]+|[.,]+$'), '');
  if (_priceRe.hasMatch(mapped) || _plainDigitsRe.hasMatch(mapped)) {
    return mapped;
  }
  return null;
}

String _contextKey(String token) =>
    token.replaceAll(RegExp(r'[.,:\-/%&]'), '').toUpperCase();

/// Kebalikan `_digitConfusion`: dipakai saat token jelas sebuah **kata**.
/// `0` dan `O` bentuknya identik pada font tanpa garis miring, jadi
/// classifier tidak punya dasar memilih. Konteks kata yang memutuskan:
/// `SIDOM0R0`, `K0NSUMEN`, `R0TI`, `PR0MO` semuanya kata.
const Map<String, String> _letterConfusion = <String, String>{
  '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B',
};

/// Ganti angka yang sebenarnya huruf di dalam sebuah kata.
///
/// Hanya berlaku bila token mayoritas huruf dan tidak berbentuk harga atau
/// ukuran kemasan. Tanpa penjaga itu, `100G` akan menjadi `IOOG`.
String normalizeAlphaToken(String token) {
  if (token.isEmpty) return token;
  int letters = 0;
  for (final String ch in token.split('')) {
    if (RegExp(r'[A-Za-z]').hasMatch(ch)) letters++;
  }
  if (letters < 2) return token;
  if (letters / token.length < 0.6) return token;
  if (isPackagingToken(token)) return token;
  // Bentuk harga (`L.OO0`) bukan kata; biarkan aturan angka yang menangani.
  if (_loosePriceRe.hasMatch(token)) return token;
  final StringBuffer buffer = StringBuffer();
  for (final String ch in token.split('')) {
    buffer.write(_letterConfusion[ch] ?? ch);
  }
  return buffer.toString();
}

/// Perbaiki huruf yang sebenarnya angka di dalam token angka.
String normalizeNumericToken(String token) {
  if (!isNumericToken(token) && !isNumericIsh(token)) return token;
  if (isPackagingToken(token)) return token;

  final StringBuffer buffer = StringBuffer();
  for (final String ch in token.split('')) {
    buffer.write(_digitConfusion[ch] ?? ch);
  }
  final String mapped = buffer.toString();
  if (mapped == token) return token;
  if (_priceRe.hasMatch(mapped) || _plainDigitsRe.hasMatch(mapped)) {
    return mapped;
  }
  return token;
}

/// Satuan kemasan yang lazim di struk minimarket.
const List<String> _unitTokens = <String>[
  'G', 'KG', 'GR', 'MG', 'ML', 'L', 'S', 'PCS', 'BTL', 'SCH', 'PCH',
  'PACK', 'BOX', 'BKS', 'RLS', 'TOP', 'X',
];

/// Satuan yang salah baca dua huruf sekaligus pada cetakan thermal.
/// `M`/`N` dan `I`/`L` bentuknya hampir sama, jadi `200ML` menjadi `200NIL`.
const Map<String, String> _unitAliases = <String, String>{
  'NIL': 'ML',
  'N1L': 'ML',
  'M1L': 'ML',
  'N1': 'ML',
  'NI': 'ML',
  'NII': 'ML',
  'MI': 'ML',
  'NILIL': 'ML',
  'NL': 'ML',
  'MIL': 'ML',
  'K6': 'KG',
  'KS': 'KG',
};

final RegExp _packagingRe =
    RegExp(r'^(\d[\dOoIlSsZzBbGgTt]*)([A-Za-z]+)$');

/// Digit yang bentuknya menyerupai satuan satu huruf.
const Map<String, List<String>> _digitToUnitLetter = <String, List<String>>{
  '6': <String>['G'],
  '1': <String>['L'],
  '5': <String>['S'],
};

/// Perbaiki ukuran kemasan seperti `250G` atau `200ML`.
String normalizePackagingToken(
  String token, {
  bool Function(String candidate)? known,
}) {
  final Match? match = _packagingRe.firstMatch(token);
  if (match == null) {
    if (known == null) return token;
    final StringBuffer buffer = StringBuffer();
    for (final String ch in token.split('')) {
      buffer.write(_digitConfusion[ch] ?? ch);
    }
    final String asDigits = buffer.toString();
    if (!_allDigits(asDigits) || asDigits.length < 3) return token;
    final List<String> letters =
        _digitToUnitLetter[asDigits[asDigits.length - 1]] ?? <String>[];
    for (final String letter in letters) {
      final String candidate =
          asDigits.substring(0, asDigits.length - 1) + letter;
      if (known(candidate)) return candidate;
    }
    return token;
  }

  final String digits = match.group(1)!;
  final String suffix = match.group(2)!;
  final StringBuffer buffer = StringBuffer();
  for (final String ch in digits.split('')) {
    buffer.write(_digitConfusion[ch] ?? ch);
  }
  final String fixedDigits = buffer.toString();
  final String upper = suffix.toUpperCase();

  if (_unitTokens.contains(upper)) return fixedDigits + upper;

  double? bestDistance;
  String? bestUnit;
  for (final String unit in _unitTokens) {
    if ((unit.length - upper.length).abs() > 1) continue;
    final double distance = weightedLevenshtein(upper, unit);
    if (distance <= 1.0 &&
        (bestDistance == null || distance < bestDistance)) {
      bestDistance = distance;
      bestUnit = unit;
    }
  }
  if (bestUnit != null) return fixedDigits + bestUnit;

  // Satuan tiga huruf yang salah baca dua huruf sekaligus. Jarak editnya 2,
  // jadi tidak tertangkap aturan di atas; peta tertutup ini menanganinya
  // tanpa membuka pintu bagi kata lain.
  final String? alias = _unitAliases[upper];
  if (alias != null) return fixedDigits + alias;
  return token;
}

bool _allDigits(String value) {
  if (value.isEmpty) return false;
  for (final String ch in value.split('')) {
    if (!_isDigit(ch)) return false;
  }
  return true;
}

/// True bila token berbentuk ukuran kemasan (angka + satuan).
bool isPackagingToken(String token, {bool Function(String candidate)? known}) {
  if (token.isEmpty || token.length < 3) return false;

  final Match? match = _packagingRe.firstMatch(token);
  if (match != null) {
    final String suffix = match.group(2)!.toUpperCase();
    if (_unitTokens.contains(suffix)) return true;
    for (final String unit in _unitTokens) {
      if ((unit.length - suffix.length).abs() <= 1 &&
          weightedLevenshtein(suffix, unit) <= 1.0) {
        return true;
      }
    }
    return false;
  }

  if (known == null) return false;
  final StringBuffer buffer = StringBuffer();
  for (final String ch in token.split('')) {
    buffer.write(_digitConfusion[ch] ?? ch);
  }
  final String asDigits = buffer.toString();
  if (!_allDigits(asDigits) || asDigits.length < 3) return false;
  final List<String> letters =
      _digitToUnitLetter[asDigits[asDigits.length - 1]] ?? <String>[];
  for (final String letter in letters) {
    if (known(asDigits.substring(0, asDigits.length - 1) + letter)) {
      return true;
    }
  }
  return false;
}

/// Perbaiki kata yang salah eja memakai leksikon.
String correctWords(
  String text,
  Lexicon lexicon, {
  double maxDistance = P.lexiconMaxDistance,
  bool useLexicon = true,
}) {
  final List<String> pieces = text.split(' ');
  final List<String> out = <String>[];
  String previous = '';
  for (final String token in pieces) {
    if (token.isEmpty || token.length < 3) {
      out.add(token);
      previous = _contextKey(token);
      continue;
    }
    if (_amountContext.contains(previous)) {
      // Kolom nominal: percayai konteks lebih dari bentuk glyph, karena
      // `500` dan `50G` sama-sama masuk akal tanpa konteks.
      final String? amount = _amountValue(token);
      if (amount != null) {
        out.add(amount);
        previous = amount;
        continue;
      }
    }
    if (isPackagingToken(token, known: lexicon.contains)) {
      out.add(normalizePackagingToken(token, known: lexicon.contains));
      previous = _contextKey(token);
      continue;
    }
    if (isNumericToken(token) || isNumericIsh(token)) {
      out.add(normalizeNumericToken(token));
      previous = _contextKey(token);
      continue;
    }
    if (lexicon.contains(token)) {
      out.add(token);
      previous = _contextKey(token);
      continue;
    }
    // Kata yang memuat angka pengganti huruf (`SIDOM0R0`) diperbaiki sebelum
    // pencarian leksikon, supaya jaraknya ke entri benar nol.
    final String alphaFixed = normalizeAlphaToken(token);
    if (alphaFixed != token) {
      out.add(alphaFixed);
      previous = _contextKey(alphaFixed);
      continue;
    }
    if (!useLexicon) {
      out.add(token);
      previous = _contextKey(token);
      continue;
    }
    final double limit =
        math.max(1.0, math.min(maxDistance, token.length * P.lexiconMaxRatio));
    final String? replacement = lexicon.closest(token, limit);
    out.add(replacement ?? token);
    previous = _contextKey(token);
  }
  return out.join(' ');
}

/// Decode satu kata, kembalikan `(teks, keyakinan)`.
(String, double) decodeLine(
  List<Float64List> probs,
  CharLM lm, {
  Lexicon? lexicon,
  int beamWidth = P.beamWidth,
  double lmWeight = P.lmWeight,
  String prefix = '',
}) {
  if (probs.isEmpty) return ('', 0.0);

  String text = beamDecode(
    probs,
    lm,
    beamWidth: beamWidth,
    lmWeight: lmWeight,
    prefix: prefix,
  );

  double total = 0;
  for (final Float64List row in probs) {
    double best = row[0];
    for (int i = 1; i < row.length; i++) {
      if (row[i] > best) best = row[i];
    }
    total += best;
  }
  final double confidence = total / probs.length;

  if (lexicon != null) {
    text = correctWords(
      text,
      lexicon,
      useLexicon: confidence < P.lexiconTrustThreshold,
    );
  }
  return (text, confidence);
}
