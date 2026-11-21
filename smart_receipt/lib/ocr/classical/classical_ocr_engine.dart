/// Mesin OCR klasik: implementasi [OcrEngine] di atas pipeline pure-ML.
///
/// Memuat model karakter, model bahasa, leksikon, dan templat dari aset,
/// lalu menjalankan pipeline yang sama dengan rujukan Python
/// (`ML/ocr/pipeline.py`).
///
/// Pipeline berjalan di **isolate terpisah**. Tanpa itu, OCR satu struk
/// memblokir UI puluhan detik sampai Android menampilkan dialog
/// "aplikasi tidak merespons". Model di-parse ulang di dalam isolate dari
/// teks JSON yang sudah dimuat; biayanya ratusan milidetik, jauh di bawah
/// waktu OCR itu sendiri.
library;

import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;

import '../ocr_engine.dart';
import '../ocr_raw_text.dart';
import 'char_classifier.dart';
import 'lexicon.dart';
import 'lm.dart';
import 'pipeline.dart';
import 'templates.dart';

/// Nama aset. Semua di bawah `assets/ocr/`.
class ClassicalOcrAssets {
  static const String charModel = 'assets/ocr/char_clf.json';
  static const String languageModel = 'assets/ocr/lm.json';
  static const String lexicon = 'assets/ocr/lexicon.json';
  static const String templates = 'assets/ocr/templates.json';
}

/// Satu baris hasil OCR dalam bentuk yang bisa dikirim antar-isolate.
class _RawLine {
  const _RawLine(this.text, this.left, this.top, this.right, this.bottom);

  final String text;
  final int left;
  final int top;
  final int right;
  final int bottom;

  List<Object> toList() => <Object>[text, left, top, right, bottom];

  static _RawLine fromList(List<Object?> values) => _RawLine(
        values[0]! as String,
        values[1]! as int,
        values[2]! as int,
        values[3]! as int,
        values[4]! as int,
      );
}

/// Jalankan pipeline pada piksel RGB mentah. Fungsi tingkat atas supaya
/// bisa dipakai `Isolate.run`.
List<Object> _recognizeInIsolate(List<Object> job) {
  final Uint8List pixels = job[0] as Uint8List;
  final int width = job[1] as int;
  final int height = job[2] as int;
  final List<String> sources = (job[3] as List<Object?>).cast<String>();

  final PageResult page = recognizePage(
    grayFromRgb(pixels, width, height),
    LinearCharModel.fromJsonString(sources[0]),
    lm: CharLM.fromJsonString(sources[1]),
    lexicon: Lexicon.fromJsonString(sources[2]),
    templates: Templates.fromJsonString(sources[3]),
  );

  return <Object>[
    page.scale,
    page.readable,
    page.lines
        .map((RecognizedLine line) => _RawLine(
              line.text,
              line.left,
              line.top,
              line.right,
              line.bottom,
            ).toList())
        .toList(),
  ];
}

class ClassicalOcrEngine implements OcrEngine {
  ClassicalOcrEngine._({required this.sources});

  /// Teks JSON mentah keempat model, urutannya tetap.
  final List<String> sources;

  static Future<ClassicalOcrEngine> load() async {
    try {
      final List<String> sources = await Future.wait(<Future<String>>[
        rootBundle.loadString(ClassicalOcrAssets.charModel),
        rootBundle.loadString(ClassicalOcrAssets.languageModel),
        rootBundle.loadString(ClassicalOcrAssets.lexicon),
        rootBundle.loadString(ClassicalOcrAssets.templates),
      ]);
      // Parse sekali di isolate utama untuk memastikan aset valid; kalau
      // rusak, kesalahannya muncul saat memuat, bukan saat memindai.
      LinearCharModel.fromJsonString(sources[0]);
      CharLM.fromJsonString(sources[1]);
      Lexicon.fromJsonString(sources[2]);
      Templates.fromJsonString(sources[3]);
      return ClassicalOcrEngine._(sources: sources);
    } catch (error) {
      throw OcrEngineException('Gagal memuat model OCR: $error');
    }
  }

  @override
  Future<OcrRawResult> recognize(String imagePath) async {
    final Uint8List bytes;
    try {
      bytes = await File(imagePath).readAsBytes();
    } catch (error) {
      throw OcrEngineException('Gambar tidak dapat dibaca: $error');
    }

    final img.Image? decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw OcrEngineException('Format gambar tidak dikenali.');
    }
    final img.Image oriented = img.bakeOrientation(decoded);
    final img.Image rgb = oriented.numChannels == 3
        ? oriented
        : oriented.convert(numChannels: 3);
    final Uint8List pixels = rgb.getBytes(order: img.ChannelOrder.rgb);
    final int width = rgb.width;
    final int height = rgb.height;

    final List<Object> result;
    // Salin ke variabel lokal: closure isolate tidak boleh menangkap `this`
    // (objek mesin tidak bisa dikirim antar-isolate).
    final List<String> modelSources = sources;
    try {
      result = await Isolate.run(
        () => _recognizeInIsolate(<Object>[
          pixels,
          width,
          height,
          modelSources,
        ]),
      );
    } catch (error) {
      throw OcrEngineException('OCR gagal dijalankan: $error');
    }

    final List<Object?> rawLines = result[2] as List<Object?>;
    final List<OcrLine> lines = rawLines
        .map((Object? raw) => _RawLine.fromList(raw as List<Object?>))
        .map((_RawLine line) => OcrLine(
              text: line.text,
              left: line.left.toDouble(),
              top: line.top.toDouble(),
              right: line.right.toDouble(),
              bottom: line.bottom.toDouble(),
            ))
        .toList();

    final String fullText =
        lines.map((OcrLine line) => line.text).join('\n');
    return OcrRawResult(lines: lines, fullText: fullText);
  }

  /// Pipeline klasik melakukan praprosesnya sendiri, jadi ia meminta citra
  /// asli: rotasi/unsharp/JPEG ulang dari [ImagePreprocessor] akan
  /// menghilangkan goresan tipis yang justru jadi andalannya.
  @override
  bool get needsRawImage => true;

  @override
  Future<void> dispose() async {}
}
