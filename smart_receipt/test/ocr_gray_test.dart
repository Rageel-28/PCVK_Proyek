/// Uji konversi citra → grayscale.
///
/// Kasus yang dijaga: PNG grayscale didekode menjadi citra **satu kanal**.
/// Kalau kanal RGB-nya tidak dinormalkan lebih dulu, luma yang dihitung dari
/// tiga kanal hanya menyisakan 0.299 x nilai aslinya dan seluruh citra
/// menjadi jauh lebih gelap — bug yang pernah terjadi di port sebelumnya.
library;

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:smart_receipt/ocr/classical/pipeline.dart';

void main() {
  test('grayscale satu kanal tidak menjadi gelap', () {
    final img.Image gray = img.Image(width: 2, height: 1, numChannels: 1);
    gray.setPixelRgb(0, 0, 128, 128, 128);
    gray.setPixelRgb(1, 0, 200, 200, 200);

    final img.Image rgb = gray.numChannels == 3
        ? gray
        : gray.convert(numChannels: 3);
    final Uint8List pixels = rgb.getBytes(order: img.ChannelOrder.rgb);

    final result = grayFromRgb(pixels, rgb.width, rgb.height);
    expect(result.data[0], closeTo(128, 2));
    expect(result.data[1], closeTo(200, 2));
  });
}
