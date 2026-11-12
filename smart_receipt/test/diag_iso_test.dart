import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:smart_receipt/ocr/classical/char_classifier.dart';
import 'package:smart_receipt/ocr/classical/classical_ocr_engine.dart';
import 'package:smart_receipt/ocr/classical/lexicon_corrector.dart';
import 'package:smart_receipt/ocr/ocr_raw_text.dart';
import 'package:smart_receipt/ocr/ocr_text_corrector.dart';
import 'package:smart_receipt/ocr/receipt_parser.dart';

/// Isolasi: upscale vs CLAHE — mana yang merusak OCR.
void main() {
  test('isolasi upscale vs CLAHE', () async {
    final classifier = CharClassifier()
      ..loadFromJsonString(File('assets/ocr_char_svm.json').readAsStringSync());
    final lexicon = LexiconCorrector()
      ..loadFromJsonString(File('assets/ocr_lexicon.json').readAsStringSync());
    final engine = ClassicalOcrEngine(classifier: classifier, lexicon: lexicon);
    await engine.load();
    final parser = ReceiptParser();

    const String src = 'test/fixtures/receipts/indomaret_01.jpg';
    final img.Image decoded = img.bakeOrientation(
      img.decodeImage(File(src).readAsBytesSync())!,
    );

    Future<void> run(String label, img.Image image, String path) async {
      final File f = File(path);
      await f.writeAsBytes(img.encodeJpg(image, quality: 92));
      final OcrRawResult raw = await engine.recognize(path);
      final ParsedReceipt p = parser.parse(correctOcrText(raw));
      // ignore: avoid_print
      print('$label: total=${p.total} (struk=${p.totalFromReceipt}) '
          'items=${p.items.length} itemsTotal=${p.itemsTotal}');
      try {
        f.deleteSync();
      } catch (_) {}
    }

    // 1. Asli
    await run('ASLI       ', decoded, '/tmp/t_orig.jpg');

    // 2. Upscale saja (short edge -> 900)
    final int shortEdge = decoded.width < decoded.height ? decoded.width : decoded.height;
    final double factor = 900 / shortEdge;
    final img.Image upscaled = img.copyResize(
      decoded,
      width: (decoded.width * factor).round(),
      height: (decoded.height * factor).round(),
      interpolation: img.Interpolation.cubic,
    );
    await run('UPSCALE    ', upscaled, '/tmp/t_up.jpg');

    // 3. CLAHE saja (pakai ImagePreprocessor internal via prepare tanpa upscale)
    //    Tidak bisa dipanggil langsung (private), jadi simulasi: terapkan
    //    _normalizeLocalContrast lewat prepare pada gambar yang sudah 900+.
    //    Sebagai proxy, bandingkan upscale+CLAHE (prepare) vs upscale saja.
    // ignore: avoid_print
    print('selesai');
  }, timeout: const Timeout(Duration(minutes: 8)));
}
