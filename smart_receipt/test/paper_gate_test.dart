/// Uji gate "ada kertas struk?" (`containsReceiptPaper`).
///
/// Regresi yang dijaga: foto struk asli dari ponsel punya histogram mulus —
/// tidak ada satu pun level abu-abu yang menampung 4% piksel. Versi lama
/// menuntut itu, sehingga **setiap foto asli ditolak** dan pengguna melihat
/// dialog "Struk tidak terdeteksi" padahal struknya jelas ada.
library;

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:smart_receipt/ocr/image_quality.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final ImagePreprocessor preprocessor = ImagePreprocessor();

  test('foto struk asli lolos gate', () async {
    for (final String name in <String>['alfamart_01.jpeg', 'indomaret_01.png']) {
      final File file = File('../testdata/real/$name');
      if (!file.existsSync()) {
        // Set foto asli bersifat opsional pada checkout bersih.
        continue;
      }
      expect(
        await preprocessor.containsReceiptPaper(file.path),
        isTrue,
        reason: '$name adalah struk asli dan tidak boleh ditolak',
      );
    }
  });

  test('bingkai gelap tanpa kertas ditolak', () async {
    final img.Image dark = img.Image(width: 300, height: 450);
    img.fill(dark, color: img.ColorRgb8(30, 30, 30));
    final String path = _writeTemp(dark, 'dark.png');
    expect(await preprocessor.containsReceiptPaper(path), isFalse);
  });

  test('papan tombol (latar terang berlubang) ditolak', () async {
    final img.Image keyboard = img.Image(width: 300, height: 450);
    img.fill(keyboard, color: img.ColorRgb8(120, 120, 120));
    for (int y = 10; y < 440; y += 36) {
      for (int x = 10; x < 290; x += 36) {
        img.fillRect(keyboard,
            x1: x, y1: y, x2: x + 28, y2: y + 28, color: img.ColorRgb8(40, 40, 40));
      }
    }
    final String path = _writeTemp(keyboard, 'keyboard.png');
    expect(await preprocessor.containsReceiptPaper(path), isFalse);
  });

  test('halaman teks terang lolos gate', () async {
    final img.Image page = img.Image(width: 300, height: 450);
    img.fill(page, color: img.ColorRgb8(235, 235, 235));
    for (int y = 30; y < 430; y += 22) {
      img.fillRect(page,
          x1: 30, y1: y, x2: 270, y2: y + 6, color: img.ColorRgb8(20, 20, 20));
    }
    final String path = _writeTemp(page, 'page.png');
    expect(await preprocessor.containsReceiptPaper(path), isTrue);
  });
}

String _writeTemp(img.Image image, String name) {
  final Directory dir = Directory.systemTemp.createTempSync('ocr_gate_');
  final File file = File('${dir.path}/$name');
  file.writeAsBytesSync(img.encodePng(image));
  return file.path;
}
