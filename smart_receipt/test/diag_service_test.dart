import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:smart_receipt/ocr/classical/char_classifier.dart';
import 'package:smart_receipt/ocr/classical/classical_ocr_engine.dart';
import 'package:smart_receipt/ocr/classical/lexicon_corrector.dart';
import 'package:smart_receipt/ocr/image_quality.dart';
import 'package:smart_receipt/ocr/ocr_raw_text.dart';
import 'package:smart_receipt/ocr/ocr_text_corrector.dart';
import 'package:smart_receipt/ocr/receipt_parser.dart';

/// Replikasi flow OcrService.persis: prepare -> recognize -> correct -> parse.
/// Untuk menemukan langkah yang membuat hasil aplikasi berbeda dari engine mentah.
void main() {
  test('replikasi OcrService', () async {
    final classifier = CharClassifier()
      ..loadFromJsonString(File('assets/ocr_char_svm.json').readAsStringSync());
    final lexicon = LexiconCorrector()
      ..loadFromJsonString(File('assets/ocr_lexicon.json').readAsStringSync());
    final engine = ClassicalOcrEngine(classifier: classifier, lexicon: lexicon);
    await engine.load();
    final parser = ReceiptParser();
    final preprocessor = ImagePreprocessor();

    const String src = 'test/fixtures/receipts/indomaret_01.jpg';
    final String out =
        '/tmp/ocr_prepared_test_${DateTime.now().millisecondsSinceEpoch}.jpg';

    final report = await preprocessor.prepare(sourcePath: src, outputPath: out);
    // ignore: avoid_print
    print('prepare: rotated=${report.rotated} lowLight=${report.lowLight} '
        'blurry=${report.blurry} size=${report.width}x${report.height} '
        'brightness=${report.brightness.toStringAsFixed(0)} '
        'blur=${report.blurScore.toStringAsFixed(1)}');

    void reportParse(String label, OcrRawResult raw) {
      final ParsedReceipt p = parser.parse(raw);
      // ignore: avoid_print
      print('=== $label (source=${p.source}) ===');
      // ignore: avoid_print
      print('  total=${p.total} (struk=${p.totalFromReceipt}) '
          'itemsTotal=${p.itemsTotal}');
      for (final ParsedItem it in p.items) {
        // ignore: avoid_print
        print('    "${it.name}" ${it.qty} x ${it.price} = ${it.lineTotal}');
      }
    }

    // 1. Engine pada gambar ASLI (tanpa prepare)
    final OcrRawResult rawOrig = await engine.recognize(src);
    reportParse('ASLI + correct', correctOcrText(rawOrig));

    // 2. Engine pada gambar yang sudah di-prepare (seperti aplikasi)
    final OcrRawResult rawPrep = await engine.recognize(out);
    reportParse('PREPARED + correct', correctOcrText(rawPrep));

    // 3. Teks mentah dari prepared (tanpa correct) untuk diagnosa
    // ignore: avoid_print
    print('--- teks mentah prepared (10 baris pertama) ---');
    final List<String> prepLines =
        rawPrep.lines.map((OcrLine l) => l.text).toList();
    for (int i = 0; i < prepLines.length && i < 12; i++) {
      // ignore: avoid_print
      print('  ${prepLines[i]}');
    }

    try {
      File(out).deleteSync();
    } catch (_) {}
  }, timeout: const Timeout(Duration(minutes: 8)));
}
