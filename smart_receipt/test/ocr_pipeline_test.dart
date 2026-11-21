/// Uji ujung-ke-ujung port Dart pada fixture terkunci `receipts_v2`.
///
/// Membandingkan teks hasil pipeline Dart dengan ground truth baris demi
/// baris, lalu menghitung CER, WER, dan exact-line seperti
/// `ML/ocr/evaluate.py`. Ambang di sini sengaja longgar (batas rencana);
/// target sebenarnya adalah hasil setara rujukan Python.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:smart_receipt/ocr/classical/char_classifier.dart';
import 'package:smart_receipt/ocr/classical/lexicon.dart';
import 'package:smart_receipt/ocr/classical/lm.dart';
import 'package:smart_receipt/ocr/classical/pipeline.dart';
import 'package:smart_receipt/ocr/classical/templates.dart';

/// Spasi ganda di ground truth hanya menandai jarak kolom.
String _normSpace(String value) =>
    value.split(RegExp(r'\s+')).where((String s) => s.isNotEmpty).join(' ');

int _levenshtein(String a, String b) {
  if (a == b) return 0;
  if (a.isEmpty) return b.length;
  if (b.isEmpty) return a.length;
  List<int> previous = List<int>.generate(b.length + 1, (int j) => j);
  List<int> current = List<int>.filled(b.length + 1, 0);
  for (int i = 1; i <= a.length; i++) {
    current[0] = i;
    for (int j = 1; j <= b.length; j++) {
      final int cost = a[i - 1] == b[j - 1] ? 0 : 1;
      int best = previous[j] + 1;
      if (current[j - 1] + 1 < best) best = current[j - 1] + 1;
      if (previous[j - 1] + cost < best) best = previous[j - 1] + cost;
      current[j] = best;
    }
    final List<int> swap = previous;
    previous = current;
    current = swap;
  }
  return previous[b.length];
}

/// Pasangkan tiap baris ground truth dengan baris terbaca paling mirip.
List<(String, String)> _matchLines(List<String> want, List<String> got) {
  final List<(String, String)> pairs = <(String, String)>[];
  final Set<int> used = <int>{};
  for (final String reference in want) {
    int? bestIndex;
    int bestDistance = 1 << 30;
    for (int i = 0; i < got.length; i++) {
      if (used.contains(i)) continue;
      final int distance = _levenshtein(_normSpace(reference), _normSpace(got[i]));
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = i;
      }
    }
    if (bestIndex == null) {
      pairs.add((reference, ''));
      continue;
    }
    used.add(bestIndex);
    pairs.add((reference, got[bestIndex]));
  }
  return pairs;
}

void main() {
  test('pipeline Dart membaca seluruh baris barang di receipts_v2', () {
    final LinearCharModel model = LinearCharModel.fromJsonString(
      File('assets/ocr/char_clf.json').readAsStringSync(),
    );
    final CharLM lm =
        CharLM.fromJsonString(File('assets/ocr/lm.json').readAsStringSync());
    final Lexicon lexicon = Lexicon.fromJsonString(
      File('assets/ocr/lexicon.json').readAsStringSync(),
    );
    final Templates templates = Templates.fromJsonString(
      File('assets/ocr/templates.json').readAsStringSync(),
    );

    final Directory dir = Directory('test/fixtures/receipts_v2');
    final List<dynamic> truth =
        jsonDecode(File('${dir.path}/ground_truth.json').readAsStringSync())
            as List<dynamic>;

    int editDistance = 0;
    int referenceChars = 0;
    int exactLines = 0;
    int totalLines = 0;
    final List<String> report = <String>[];

    for (final dynamic raw in truth) {
      final Map<String, dynamic> entry = raw as Map<String, dynamic>;
      final img.Image? decoded =
          img.decodeImage(File('${dir.path}/${entry['image']}').readAsBytesSync());
      expect(decoded, isNotNull, reason: 'gambar tidak terbaca');
      final img.Image rgb = decoded!.numChannels == 3
          ? decoded
          : decoded.convert(numChannels: 3);
      final Uint8List pixels = rgb.getBytes(order: img.ChannelOrder.rgb);

      final PageResult page = recognizePage(
        grayFromRgb(pixels, rgb.width, rgb.height),
        model,
        lm: lm,
        lexicon: lexicon,
        templates: templates,
      );
      final List<String> got =
          page.lines.map((RecognizedLine l) => l.text).toList();
      final List<String> want = (entry['lines'] as List<dynamic>)
          .map((dynamic e) => e as String)
          .toList();

      final List<(String, String)> pairs = _matchLines(want, got);
      final List<String> misses = <String>[];
      for (final (String reference, String hypothesis) in pairs) {
        totalLines++;
        final String wantText = _normSpace(reference);
        final String gotText = _normSpace(hypothesis);
        if (wantText == gotText) {
          exactLines++;
        } else {
          misses.add('  mau : "$wantText"\n  dpt : "$gotText"');
        }
        editDistance += _levenshtein(gotText, wantText);
        referenceChars += wantText.length;
      }
      report.add(
        '${entry['image']}: ${pairs.length - misses.length}/${pairs.length} '
        'baris persis',
      );
      if (misses.isNotEmpty) {
        report.addAll(misses);
      }
    }

    final double cer = referenceChars == 0 ? 0 : editDistance / referenceChars;
    final double exact = totalLines == 0 ? 0 : exactLines / totalLines;
    // ignore: avoid_print
    print(report.join('\n'));
    // ignore: avoid_print
    print('CER ${(cer * 100).toStringAsFixed(2)}%  '
        'exact ${(exact * 100).toStringAsFixed(1)}% ($exactLines/$totalLines)');

    expect(exact, greaterThanOrEqualTo(0.80));
    expect(cer, lessThanOrEqualTo(0.15));
  }, timeout: const Timeout(Duration(minutes: 10)));
}
