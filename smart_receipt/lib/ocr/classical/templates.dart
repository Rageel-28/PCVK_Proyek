/// Pencocokan baris terhadap templat baris struk yang dikenal.
///
/// Port dari `ML/ocr/templates.py`.
library;

import 'dart:convert';

import 'decode.dart';
import 'lexicon.dart';

/// Rangkaian angka pada baris, hanya dari token yang memang angka.
String numberSignature(String text) {
  final StringBuffer buffer = StringBuffer();
  for (final String token in text.split(' ')) {
    if (token.isEmpty) continue;
    if (!isNumericToken(token) && !isNumericIsh(token)) continue;
    for (final String ch in token.split('')) {
      buffer.write(_digitConfusionForTemplate[ch] ?? ch);
    }
  }
  return buffer.toString();
}

/// Salinan peta konfusi dari decode (tidak diekspor di sana).
const Map<String, String> _digitConfusionForTemplate = <String, String>{
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

/// Ganti baris hasil OCR dengan templat terdekat bila cukup mirip.
String snapToTemplate(
  String text,
  List<String> templates, {
  double maxRatio = 0.2,
  int minLength = 8,
}) {
  final String stripped = text.trim();
  if (stripped.length < minLength || templates.isEmpty) return text;

  final String textDigits = numberSignature(stripped);
  double? bestRatio;
  String? bestTemplate;

  for (final String template in templates) {
    final String templateDigits = numberSignature(template);
    double ratioLimit;
    if (templateDigits.isNotEmpty) {
      // Templat berangka: angka harus sama persis. Penjaga ini melindungi
      // harga, tanggal, dan nomor struk.
      if (templateDigits != textDigits) continue;
      ratioLimit = maxRatio;
    } else {
      // Templat tanpa angka: huruf yang salah baca sering muncul sebagai
      // digit, jadi penjaga angka dilonggarkan.
      ratioLimit = maxRatio > 0.45 ? maxRatio : 0.45;
    }

    final int lengthLimit =
        (template.length * 0.4).toInt() > 2 ? (template.length * 0.4).toInt() : 2;
    if ((template.length - stripped.length).abs() > lengthLimit) continue;

    final double distance = weightedLevenshtein(stripped, template);
    final double ratio = distance / (template.isEmpty ? 1 : template.length);
    if (ratio > ratioLimit) continue;
    if (bestRatio == null || ratio < bestRatio) {
      bestRatio = ratio;
      bestTemplate = template;
    }
  }
  return bestTemplate ?? text;
}

class Templates {
  Templates(List<String> lines) : lines = _clean(lines);

  final List<String> lines;

  static List<String> _clean(List<String> source) {
    final List<String> out = <String>[];
    final Set<String> seen = <String>{};
    for (final String line in source) {
      final String cleaned =
          line.split(RegExp(r'\s+')).where((String s) => s.isNotEmpty).join(' ');
      if (cleaned.isEmpty) continue;
      if (seen.add(cleaned)) out.add(cleaned);
    }
    return out;
  }

  int get length => lines.length;

  String snap(String text, {double maxRatio = 0.2}) =>
      snapToTemplate(text, lines, maxRatio: maxRatio);

  static Templates fromJsonString(String source) {
    final List<dynamic> raw = jsonDecode(source) as List<dynamic>;
    return Templates(raw.map((dynamic e) => e as String).toList());
  }
}
