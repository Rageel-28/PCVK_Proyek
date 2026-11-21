/// Harness sementara: jalankan engine klasik + parser pada struk121.jpeg.
library;

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:smart_receipt/ocr/classical/char_classifier.dart';
import 'package:smart_receipt/ocr/classical/lexicon.dart';
import 'package:smart_receipt/ocr/classical/lm.dart';
import 'package:smart_receipt/ocr/classical/pipeline.dart';
import 'package:smart_receipt/ocr/classical/templates.dart';
import 'package:smart_receipt/ocr/ocr_raw_text.dart';
import 'package:smart_receipt/ocr/receipt_parser.dart';

void main() {
  test('struk121', () {
    final LinearCharModel model = LinearCharModel.fromJsonString(
      File('assets/ocr/char_clf.json').readAsStringSync(),
    );
    final CharLM lm = CharLM.fromJsonString(
      File('assets/ocr/lm.json').readAsStringSync(),
    );
    final Lexicon lexicon = Lexicon.fromJsonString(
      File('assets/ocr/lexicon.json').readAsStringSync(),
    );
    final Templates templates = Templates.fromJsonString(
      File('assets/ocr/templates.json').readAsStringSync(),
    );

    final String path = '/Users/aldisuryas/Downloads/struk121.jpeg';
    final File file = File(path);
    if (!file.existsSync()) {
      // Berkas ini bersifat lokal; lewati bila tidak ada.
      stdout.writeln('struk121.jpeg tidak ditemukan, lewati.');
      return;
    }
    final img.Image decoded = img.decodeImage(file.readAsBytesSync())!;
    final img.Image rgb = decoded.convert(numChannels: 3);
    final PageResult result = recognizePage(
      grayFromRgb(
        rgb.getBytes(order: img.ChannelOrder.rgb),
        rgb.width,
        rgb.height,
      ),
      model,
      lm: lm,
      lexicon: lexicon,
    );

    final List<OcrLine> lines = <OcrLine>[];
    final StringBuffer full = StringBuffer();
    stdout.writeln('==== OCR struk121');
    for (final RecognizedLine line in result.lines) {
      final String text = templates.snap(line.text);
      stdout.writeln(
          '[${line.confidence.toStringAsFixed(2)}] $text');
      lines.add(OcrLine(
        text: text,
        left: line.left.toDouble(),
        top: line.top.toDouble(),
        right: line.right.toDouble(),
        bottom: line.bottom.toDouble(),
      ));
      full.writeln(text);
    }

    final OcrRawResult raw = OcrRawResult(lines: lines, fullText: full.toString());
    final ParsedReceipt p = ReceiptParser().parse(raw);
    stdout.writeln('==== PARSER');
    stdout.writeln('store: ${p.store}');
    stdout.writeln('total: ${p.total}  discount: ${p.discountTotal}  itemsTotal: ${p.itemsTotal}');
    for (final ParsedItem it in p.items) {
      stdout.writeln('  item: ${it.name}  qty=${it.qty}  price=${it.price}  subtotal=${it.subtotal}');
    }
  });
}
