import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:smart_receipt/ocr/classical/char_classifier.dart';
import 'package:smart_receipt/ocr/classical/classical_ocr_engine.dart';
import 'package:smart_receipt/ocr/classical/gray_image.dart';
import 'package:smart_receipt/ocr/classical/lexicon_corrector.dart';
import 'package:smart_receipt/ocr/classical/segmenter.dart';

/// Uji end-to-end pipeline OCR klasik pada gambar nyata.
///
/// Gambar uji dibuat oleh `ML/ocr_classical/export_test_images.py`.
void main() {
  late Map<String, dynamic> manifest;
  late CharClassifier classifier;
  late LexiconCorrector lexicon;

  setUpAll(() {
    manifest = jsonDecode(
      File('test/fixtures/line_images.json').readAsStringSync(),
    ) as Map<String, dynamic>;

    classifier = CharClassifier()
      ..loadFromJsonString(File('assets/ocr_char_svm.json').readAsStringSync());

    lexicon = LexiconCorrector()
      ..loadFromJsonString(File('assets/ocr_lexicon.json').readAsStringSync());
  });

  group('leksikon', () {
    test('memuat kamus dari aset', () {
      expect(lexicon.isLoaded, isTrue);
      expect(lexicon.wordCount, greaterThan(1000));
    });

    test('levenshtein menghitung jarak edit dengan benar', () {
      expect(levenshtein('COKLAT', 'COKLAT'), 0);
      expect(levenshtein('COK1AT', 'COKLAT'), 1);
      expect(levenshtein('INDOMIE', 'INDOMI'), 1);
      expect(levenshtein('', 'ABC'), 3);
    });

    test('mengoreksi salah baca digit menjadi huruf', () {
      final ({String word, int distance, bool changed}) r =
          lexicon.correct('COK1AT');
      expect(r.word, 'COKLAT');
      expect(r.changed, isTrue);
    });

    test('kata yang sudah benar tidak diubah', () {
      final ({String word, int distance, bool changed}) r =
          lexicon.correct('INDOMIE');
      expect(r.word, 'INDOMIE');
      expect(r.changed, isFalse);
    });
  });

  group('segmentasi baris', () {
    test('menemukan minimal satu baris pada citra teks', () {
      final img.Image? decoded =
          img.decodePng(File('test/fixtures/line_0.png').readAsBytesSync());
      expect(decoded, isNotNull);

      final GrayImage gray = GrayImage.fromImage(decoded!);
      // Segmentasi diuji pada citra yang sudah dibinerkan.
      final ClassicalOcrEngine engine = ClassicalOcrEngine(
        classifier: classifier,
        lexicon: lexicon,
      );
      final ({dynamic gray, dynamic ink, double angle, double scale}) prep =
          engine.prepare(gray);

      final List<({int top, int bottom})> lines = segmentLines(prep.ink);
      expect(lines.length, greaterThanOrEqualTo(1));
    });
  });

  group('pipeline end-to-end', () {
    test('hasil Dart sama persis dengan hasil Python', () async {
      final ClassicalOcrEngine engine = ClassicalOcrEngine(
        classifier: classifier,
        lexicon: lexicon,
      );

      final List<dynamic> images = manifest['images'] as List<dynamic>;
      int exact = 0;
      final List<String> report = <String>[];

      for (final dynamic raw in images) {
        final Map<String, dynamic> entry = raw as Map<String, dynamic>;
        final String file = entry['file'] as String;
        final String expected = entry['text'] as String;
        final String pythonText = entry['python_text'] as String;

        final img.Image? decoded =
            img.decodePng(File('test/fixtures/$file').readAsBytesSync());
        expect(decoded, isNotNull, reason: 'gagal dekode $file');

        final ClassicalOcrResult result =
            engine.run(GrayImage.fromImage(decoded!));

        final String got = result.text.trim();
        if (got == expected) {
          exact++;
        }
        report.add(
          'acuan="$expected" python="$pythonText" dart="$got" '
          '(conf ${result.meanConfidence.toStringAsFixed(2)})',
        );
      }

      // Fixture dipilih dari varian yang berhasil dikenali Python. Setelah
      // seluruh perbedaan numerik disamakan (pembulatan resize, urutan
      // denoise, konvensi rotasi, penanganan tepi Sauvola), port Dart
      // menghasilkan teks yang identik untuk semua fixture.
      expect(
        exact / images.length,
        equals(1.0),
        reason: report.join('\n'),
      );
    });
  });

  group('kontrak OcrEngine', () {
    test('menghasilkan OcrLine per kelompok kata', () async {
      final ClassicalOcrEngine engine = ClassicalOcrEngine(
        classifier: classifier,
        lexicon: lexicon,
      );

      final result = await engine.recognize('test/fixtures/line_0.png');

      // "INDOMIE GORENG" harus menghasilkan dua OcrLine terpisah, bukan
      // satu baris selebar gambar. Ini penting agar receipt_parser dapat
      // memisahkan kolom nama dan harga.
      expect(result.lines.length, greaterThanOrEqualTo(2));
      expect(result.fullText.trim().isNotEmpty, isTrue);

      // Setiap baris harus punya geometri yang masuk akal.
      for (final line in result.lines) {
        expect(line.right, greaterThan(line.left));
        expect(line.bottom, greaterThan(line.top));
      }
    });
  });
}
