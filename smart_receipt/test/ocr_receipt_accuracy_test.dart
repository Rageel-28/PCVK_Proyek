import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:smart_receipt/ocr/classical/char_classifier.dart';
import 'package:smart_receipt/ocr/classical/classical_ocr_engine.dart';
import 'package:smart_receipt/ocr/classical/lexicon_corrector.dart';

/// Ukur akurasi pipeline Dart pada struk Indomaret sintetis.
///
/// Ground truth dibuat oleh generator struk (seed tetap), jadi angka ini
/// deterministik. Tes ini adalah pengaman regresi: bila binarisasi atau
/// segmentasi diubah dan akurasi jatuh jauh, tes ini gagal.
///
/// Metrik: CER per baris acuan terhadap baris OCR terbaik (urutan baris
/// pada keluaran OCR tidak harus sama, yang penting tiap baris inti
/// terbaca).
void main() {
  late ClassicalOcrEngine engine;
  late List<Map<String, dynamic>> gt;

  setUpAll(() async {
    final classifier = CharClassifier()
      ..loadFromJsonString(File('assets/ocr_char_svm.json').readAsStringSync());
    final lexicon = LexiconCorrector()
      ..loadFromJsonString(File('assets/ocr_lexicon.json').readAsStringSync());
    engine = ClassicalOcrEngine(classifier: classifier, lexicon: lexicon);
    await engine.load();

    final raw =
        File('test/fixtures/receipts/ground_truth.json').readAsStringSync();
    gt = (jsonDecode(raw) as List<dynamic>)
        .cast<Map<String, dynamic>>()
        .toList();
  });

  test('akurasi struk Indomaret (metode default)', () async {
    double totalCer = 0;
    int totalLines = 0;
    int exactLines = 0;
    final List<String> report = <String>[];

    for (final e in gt) {
      final String name = e['image'] as String;
      final List<String> refLines =
          (e['lines'] as List<dynamic>).cast<String>().toList();

      final result = await engine.runFile('test/fixtures/receipts/$name');
      final List<String> hypLines =
          result.lines.map((l) => l.text).where((l) => l.trim().isNotEmpty).toList();

      double lineCer = 0;
      for (final r in refLines) {
        final double best = _cer(r, hypLines);
        lineCer += best;
        if (best == 0.0) exactLines++;
      }
      final double avg = refLines.isEmpty ? 1.0 : lineCer / refLines.length;
      totalCer += avg;
      totalLines += refLines.length;
      report.add('$name CER=${(avg * 100).toStringAsFixed(1)}% '
          'conf=${result.meanConfidence.toStringAsFixed(2)}');
    }

    final double meanCer = gt.isEmpty ? 1.0 : totalCer / gt.length;
    // ignore: avoid_print
    print('\n--- Hasil OCR struk Indomaret (Dart) ---');
    for (final r in report) {
      // ignore: avoid_print
      print('  $r');
    }
    // ignore: avoid_print
    print('  RATA-RATA CER=${(meanCer * 100).toStringAsFixed(2)}% '
        'baris tepat=$exactLines/$totalLines');

    // Pengaman regresi: akurasi harus jauh lebih baik dari acak.
    // Baseline (sauvola) ~43%; adaptive target ~26%.
    expect(
      meanCer,
      lessThan(0.35),
      reason: 'CER ${(meanCer * 100).toStringAsFixed(1)}% terlalu tinggi. '
          'Perubahan binarisasi/segmentasi mungkin merusak akurasi.',
    );
  }, timeout: const Timeout(Duration(minutes: 10)));
}

double _cer(String ref, List<String> hypLines) {
  final String rn = _norm(ref);
  double best = 1.0;
  for (final h in hypLines) {
    final double c = _cerPair(rn, _norm(h));
    if (c < best) best = c;
  }
  return best;
}

String _norm(String s) => s.toUpperCase().split(RegExp(r'\s+')).join(' ');

int _levenshtein(String a, String b) {
  if (a == b) return 0;
  if (a.isEmpty) return b.length;
  if (b.isEmpty) return a.length;
  List<int> prev = List<int>.generate(b.length + 1, (i) => i);
  for (int i = 1; i <= a.length; i++) {
    List<int> cur = <int>[i];
    for (int j = 1; j <= b.length; j++) {
      cur.add(<int>[
        prev[j] + 1,
        cur[j - 1] + 1,
        prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1),
      ].reduce((x, y) => x < y ? x : y));
    }
    prev = cur;
  }
  return prev[b.length];
}

double _cerPair(String ref, String hyp) {
  if (ref.isEmpty) return hyp.isEmpty ? 0.0 : 1.0;
  return _levenshtein(ref, hyp) / ref.length;
}
