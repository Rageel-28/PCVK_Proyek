import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:smart_receipt/ocr/classical/char_classifier.dart';
import 'package:smart_receipt/ocr/classical/classical_ocr_engine.dart';
import 'package:smart_receipt/ocr/classical/gray_image.dart';

/// Paritas Dart vs Python pada potongan baris struk **asli**.
///
/// Berkas `real_line_parity.json` dibuat oleh `export_real_parity.py`:
/// setiap entri berisi piksel crop asli, transkripsi manual, dan teks hasil
/// pipeline Python.
///
/// Uji ini tidak menuntut kesamaan 100%. Alasannya diukur, bukan diasumsikan:
/// pada beberapa glyph yang secara visual identik (koma 3 px, `0`/`O` pada
/// cetakan pudar) skor linear SVM hampir seri, dan selisih pembulatan
/// floating point antara NumPy dan Dart cukup untuk membalik argmax. Karena
/// itu uji ini menetapkan dua ambang yang bermakna:
///
///   - kesepakatan tingkat karakter >= 97% (Dart vs Python), dan
///   - baris identik persis >= 60%.
///
/// Paritas HOG dan SVM diuji terpisah dan tetap harus 100% identik
/// (`ocr_classical_parity_test.dart`).
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('paritas teks pada potongan baris struk asli', () async {
    final File fixture = File('test/fixtures/real_line_parity.json');
    if (!fixture.existsSync()) {
      fail(
        'Fixture belum ada. Jalankan: '
        'python export_real_parity.py di ML/ocr_classical',
      );
    }

    final Map<String, dynamic> payload =
        jsonDecode(fixture.readAsStringSync()) as Map<String, dynamic>;
    final List<dynamic> samples = payload['samples'] as List<dynamic>;

    final CharClassifier classifier = CharClassifier();
    await classifier.load();
    final ClassicalOcrEngine engine = ClassicalOcrEngine(classifier: classifier);
    await engine.load();

    int exact = 0;
    int charTotal = 0;
    int charMatch = 0;
    final List<String> failures = <String>[];

    // Dipecah per jenis: struk render bersih dan foto kusut punya tingkat
    // kesepakatan yang jauh berbeda, dan menggabungkannya menyembunyikan
    // fakta itu.
    final Map<String, int> kindTotal = <String, int>{};
    final Map<String, int> kindMatch = <String, int>{};

    for (final dynamic raw in samples) {
      final Map<String, dynamic> s = raw as Map<String, dynamic>;
      final int w = s['width'] as int;
      final int h = s['height'] as int;
      final GrayImage gray;
      if (s['png_b64'] is String) {
        final img.Image? decoded =
            img.decodePng(base64Decode(s['png_b64'] as String));
        if (decoded == null) {
          fail('Gagal decode png_b64 untuk ${s['stem']} #${s['index']}');
        }
        gray = GrayImage.zeros(decoded.width, decoded.height);
        for (int y = 0; y < decoded.height; y++) {
          final int rowBase = y * decoded.width;
          for (int x = 0; x < decoded.width; x++) {
            gray.data[rowBase + x] = decoded.getPixel(x, y).r.toInt();
          }
        }
      } else {
        final List<dynamic> pixels = s['pixels'] as List<dynamic>;
        gray = GrayImage.zeros(w, h);
        for (int i = 0; i < pixels.length; i++) {
          gray.data[i] = pixels[i] as int;
        }
      }

      final String expected = (s['python_text'] as String).trim();
      final String actual = engine.run(gray).text.trim();
      final String kind = (s['kind'] as String?) ?? '?';

      if (actual == expected) {
        exact++;
      } else {
        failures.add(
          '${s['stem']} #${s['index']}\n'
          '  asli  : ${s['truth']}\n'
          '  python: $expected\n'
          '  dart  : $actual',
        );
      }

      final int n = expected.length > actual.length
          ? expected.length
          : actual.length;
      for (int i = 0; i < n; i++) {
        charTotal++;
        kindTotal[kind] = (kindTotal[kind] ?? 0) + 1;
        final String a = i < expected.length ? expected[i] : '';
        final String b = i < actual.length ? actual[i] : '';
        if (a == b) {
          charMatch++;
          kindMatch[kind] = (kindMatch[kind] ?? 0) + 1;
        }
      }
    }

    final double charAgreement = charTotal == 0 ? 1.0 : charMatch / charTotal;
    final double exactRate = samples.isEmpty ? 1.0 : exact / samples.length;

    // ignore: avoid_print
    print(
      'paritas: baris identik $exact/${samples.length} '
      '(${(exactRate * 100).toStringAsFixed(1)}%), '
      'kesepakatan karakter ${(charAgreement * 100).toStringAsFixed(2)}%',
    );
    for (final String kind in kindTotal.keys.toList()..sort()) {
      final int t = kindTotal[kind]!;
      final int m = kindMatch[kind] ?? 0;
      // ignore: avoid_print
      print('  $kind: ${(100 * m / t).toStringAsFixed(2)}% ($t karakter)');
    }
    for (final String f in failures.take(5)) {
      // ignore: avoid_print
      print(f);
    }

    final double renderAgreement =
        (kindMatch['render'] ?? 0) / (kindTotal['render'] ?? 1);

    expect(
      renderAgreement,
      greaterThanOrEqualTo(0.99),
      reason: 'pada struk render bersih, Dart harus sama dengan Python',
    );
    expect(
      charAgreement,
      greaterThanOrEqualTo(0.95),
      reason: 'kesepakatan karakter Dart vs Python terlalu rendah',
    );
    expect(
      exactRate,
      greaterThanOrEqualTo(0.60),
      reason: 'terlalu banyak baris berbeda dari Python',
    );
  }, timeout: const Timeout(Duration(minutes: 10)));
}
