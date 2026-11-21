/// Regresi: latar/tepi gelap tidak boleh menyatukan seluruh baris struk.
///
/// Sauvola menandai latar gelap dan bayangan tepi sebagai tinta. Karena
/// komponennya menyambung dari atas ke bawah, setiap baris citra tetap punya
/// tinta, proyeksi horizontal tidak pernah turun ke nol, dan semua baris
/// menyatu menjadi satu pita raksasa yang lalu dibuang sebagai "graphic".
/// Akibatnya struk yang jelas terbaca menghasilkan nol baris.
///
/// [stripArtifactComponents] membuang komponen tinta yang terlalu besar untuk
/// bisa berupa karakter sebelum deteksi baris dijalankan.
library;

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:smart_receipt/ocr/classical/char_classifier.dart';
import 'package:smart_receipt/ocr/classical/gray_image.dart';
import 'package:smart_receipt/ocr/classical/lexicon.dart';
import 'package:smart_receipt/ocr/classical/lm.dart';
import 'package:smart_receipt/ocr/classical/pipeline.dart';
import 'package:smart_receipt/ocr/classical/preprocess.dart';
import 'package:smart_receipt/ocr/classical/templates.dart';

/// Citra biner sintetis: strip vertikal penuh setinggi citra di tepi kiri
/// (artefak bayangan) plus beberapa batang "teks" yang wajar.
BinaryImage _syntheticPage() {
  const int w = 400;
  const int h = 600;
  final BinaryImage page = BinaryImage.zeros(w, h);

  void fillRect(int left, int top, int right, int bottom) {
    for (int y = top; y < bottom; y++) {
      for (int x = left; x < right; x++) {
        page.data[y * w + x] = 255;
      }
    }
  }

  // Artefak: strip 30 px penuh dari atas ke bawah di tepi kiri.
  fillRect(0, 0, 30, h);
  // Lima batang teks 200x24 px, jarak 80 px.
  for (int i = 0; i < 5; i++) {
    fillRect(60, 60 + i * 80, 260, 84 + i * 80);
  }
  return page;
}

int _inkInColumn(BinaryImage image, int x) {
  int count = 0;
  for (int y = 0; y < image.height; y++) {
    if (image.data[y * image.width + x] != 0) count++;
  }
  return count;
}

void main() {
  test('komponen artefak setinggi citra dibuang, teks tetap', () {
    final BinaryImage page = _syntheticPage();
    expect(_inkInColumn(page, 5), page.height);

    final BinaryImage cleaned = stripArtifactComponents(page);

    // Strip artefak hilang sepenuhnya.
    expect(_inkInColumn(cleaned, 5), 0);
    // Batang teks masih utuh: baris pertama y=60..84, x=60..260.
    expect(cleaned.data[70 * cleaned.width + 100], isNot(0));
    expect(cleaned.data[70 * cleaned.width + 250], isNot(0));
    // Tidak ada piksel teks yang ikut terhapus.
    int ink = 0;
    for (int i = 0; i < cleaned.length; i++) {
      if (cleaned.data[i] != 0) ink++;
    }
    expect(ink, 5 * 200 * 24);
  });

  test('struk dengan latar gelap tetap menghasilkan baris terbaca', () {
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

    // Dua foto yang sebelumnya menghasilkan nol baris karena latar gelap.
    for (final String name in <String>[
      '01_indomaret_20180607.jpg',
      '03_alfamart_permata_gardenia_20190120.jpg',
    ]) {
      final img.Image decoded =
          img.decodeImage(File('../testdata/random/$name').readAsBytesSync())!;
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
        templates: templates,
      );
      expect(
        result.lines.length,
        greaterThanOrEqualTo(10),
        reason: '$name hanya menghasilkan ${result.lines.length} baris',
      );
    }
  });
}
