/// Pemeriksaan manual: jalankan pipeline Dart pada foto struk asli.
///
/// Berkas ini sengaja bukan bagian dari suite tetap; dipakai untuk
/// membandingkan hasil port Dart dengan rujukan Python pada struk asli
/// di `../testdata/real`.
library;

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:smart_receipt/ocr/classical/char_classifier.dart';
import 'package:smart_receipt/ocr/classical/lexicon.dart';
import 'package:smart_receipt/ocr/classical/lm.dart';
import 'package:smart_receipt/ocr/classical/pipeline.dart';
import 'package:smart_receipt/ocr/classical/templates.dart';

void main() {
  test('struk asli', () {
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

    for (final String name in <String>[
      'alfamart_01.jpeg',
      'indomaret_01.png',
    ]) {
      final img.Image decoded =
          img.decodeImage(File('../testdata/real/$name').readAsBytesSync())!;
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
      stdout.writeln('==== $name');
      for (final RecognizedLine line in result.lines) {
        final String text = templates.snap(line.text);
        stdout.writeln('[${line.confidence.toStringAsFixed(2)}] $text');
      }
    }
  });
}
