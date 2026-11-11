import 'dart:developer' as developer;
import 'dart:io';
import 'dart:math' as math;

import 'package:path_provider/path_provider.dart';

import 'classical/classical_ocr_engine.dart';
import 'image_quality.dart';
import 'ocr_engine.dart';
import 'ocr_raw_text.dart';
import 'ocr_text_corrector.dart';
import 'receipt_parser.dart';

/// Everything produced by scanning one receipt.
class OcrScanResult {
  const OcrScanResult({
    required this.sourcePath,
    required this.quality,
    required this.raw,
    required this.parsed,
  });

  /// Original captured/gallery image.
  final String sourcePath;

  /// Preprocessing report + path of the image that was actually OCR'd.
  final ImageQualityReport quality;

  final OcrRawResult raw;
  final ParsedReceipt parsed;

  bool get hasRecognizedText => !raw.isEmpty;
}

/// Raised for user-facing OCR failures (empty result, unreadable image, ...).
class OcrFailure implements Exception {
  OcrFailure(this.message, {this.hint});

  final String message;
  final String? hint;

  @override
  String toString() => message;
}

/// Pipeline stages, surfaced so the UI can show real progress.
enum OcrStage { preprocessing, recognizing, parsing, done }

/// Orkestrasi pipeline lengkap:
/// capture -> pra-proses (rotasi/blur/kurang cahaya) -> OCR klasik -> parse.
///
/// Tahap OCR dijalankan oleh [ClassicalOcrEngine]: pengolahan citra klasik
/// dan machine learning klasik (HOG + Linear SVM) yang dibangun sendiri,
/// tanpa library OCR siap pakai dan tanpa deep learning.
class OcrService {
  OcrService({OcrEngine? engine, ImagePreprocessor? preprocessor})
      : _engine = engine ?? ClassicalOcrEngine(),
        _preprocessor = preprocessor ?? ImagePreprocessor();

  final OcrEngine _engine;
  final ImagePreprocessor _preprocessor;
  final ReceiptParser _parser = ReceiptParser();

  Future<OcrScanResult> process(
    String sourcePath, {
    void Function(OcrStage stage)? onStage,
  }) async {
    final Directory dir = await getTemporaryDirectory();
    final String preparedPath =
        '${dir.path}/ocr_prepared_${DateTime.now().millisecondsSinceEpoch}.jpg';

    onStage?.call(OcrStage.preprocessing);
    final ImageQualityReport quality;
    try {
      quality = await _preprocessor.prepare(
        sourcePath: sourcePath,
        outputPath: preparedPath,
      );
    } catch (e) {
      throw OcrFailure(
        'Gambar tidak dapat diproses.',
        hint: 'Pastikan file gambar valid lalu coba lagi.',
      );
    }

    onStage?.call(OcrStage.recognizing);
    OcrRawResult raw;
    ParsedReceipt parsed;
    try {
      final OcrRawResult rawFromEngine = await _engine.recognize(quality.preparedPath);
      raw = correctOcrText(rawFromEngine);
      parsed = _parser.parse(raw);
    } on OcrEngineException catch (e) {
      throw OcrFailure('OCR gagal dijalankan.', hint: e.message);
    }

    // Gallery photos in particular often fade unevenly or arrive at a different
    // scale than an in-app capture, which can cost the recognizer most of the
    // digits. When the first pass looks weak, try up to two alternatives and
    // keep whichever reading the parser trusts more.
    if (_isWeak(parsed)) {
      parsed = await _retryOcr(
        quality: quality,
        dir: dir,
        raw: raw,
        parsed: parsed,
        onRaw: (OcrRawResult value) => raw = value,
      );
    }

    if (raw.isEmpty) {
      bool paperFound = true;
      try {
        paperFound = await _preprocessor.containsReceiptPaper(quality.preparedPath);
      } catch (_) {
        paperFound = true;
      }
      if (!paperFound) {
        throw OcrFailure(
          'Struk tidak terdeteksi pada foto.',
          hint: 'Pastikan struk terlihat utuh dan mengisi sebagian besar '
              'bingkai, lalu hindari latar gelap seperti keyboard.',
        );
      }
      throw OcrFailure(
        'Tidak ada teks yang terbaca pada struk.',
        hint: 'Coba foto ulang dengan pencahayaan lebih terang dan '
            'posisikan struk memenuhi bingkai.',
      );
    }

    // Last resort for gallery imports that are stored sideways: the engine does not
    // auto-rotate, so a receipt lying on its side yields a couple of stray
    // lines. Try a quarter turn and keep the better reading.
    if (raw.lineCount < 5) {
      try {
        final String rotatedPath = '${dir.path}/ocr_rotated_'
            '${DateTime.now().millisecondsSinceEpoch}.jpg';
        await _preprocessor.prepareRotated(
          preparedPath: quality.preparedPath,
          outputPath: rotatedPath,
          degrees: 90,
        );
        final OcrRawResult rotatedRaw =
            correctOcrText(await _engine.recognize(rotatedPath));
        final ParsedReceipt rotatedParsed = _parser.parse(rotatedRaw);
        if (ReceiptParser.scoreOf(rotatedParsed) >
            ReceiptParser.scoreOf(parsed)) {
          raw = rotatedRaw;
          parsed = rotatedParsed;
        }
      } catch (_) {
        // Keep the original result.
      }
    }

    onStage?.call(OcrStage.parsing);
    onStage?.call(OcrStage.done);

    developer.log(
      'OCR selesai: ${raw.lineCount} baris, '
      'store="${parsed.store}", items=${parsed.items.length}, '
      'total=${parsed.total}, sumber=${parsed.source}, '
      'brightness=${quality.brightness.toStringAsFixed(0)}, '
      'blur=${quality.blurScore.toStringAsFixed(1)}, '
      'rotated=${quality.rotationDegrees}°',
      name: 'SmartReceipt.OCR',
    );

    return OcrScanResult(
      sourcePath: sourcePath,
      quality: quality,
      raw: raw,
      parsed: parsed,
    );
  }

  Future<void> dispose() => _engine.dispose();

  /// True when the first pass clearly lost the receipt structure.
  bool _isWeak(ParsedReceipt parsed) {
    if (parsed.items.isEmpty) {
      return true;
    }
    // A receipt without a printed total, or one whose items do not reconcile
    // with it, is a good candidate for a second look.
    if (!parsed.totalFromReceipt) {
      return true;
    }
    final int gap = (parsed.total + parsed.discountTotal - parsed.itemsTotal).abs();
    return gap > 100;
  }

  /// Retries OCR on better-prepared variants of the image and keeps the best
  /// reading. Each attempt is best-effort: a failure leaves the previous result
  /// in place rather than failing the scan.
  Future<ParsedReceipt> _retryOcr({
    required ImageQualityReport quality,
    required Directory dir,
    required OcrRawResult raw,
    required ParsedReceipt parsed,
    required void Function(OcrRawResult) onRaw,
  }) async {
    final int stamp = DateTime.now().millisecondsSinceEpoch;
    ParsedReceipt best = parsed;
    OcrRawResult bestRaw = raw;

    Future<void> attempt(String path) async {
      final OcrRawResult retryRaw =
          correctOcrText(await _engine.recognize(path));
      final ParsedReceipt retryParsed = _parser.parse(retryRaw);
      if (ReceiptParser.scoreOf(retryParsed) > ReceiptParser.scoreOf(best)) {
        best = retryParsed;
        bestRaw = retryRaw;
      }
    }

    // 1. Perspective correction — crop the receipt out of the surrounding frame
    //    and correct any projective distortion. This runs first so subsequent
    //    strategies operate on the tightest possible content region.
    try {
      final String perspPath = '${dir.path}/ocr_persp_$stamp.jpg';
      final String result = await _preprocessor.preparePerspectiveCorrected(
        preparedPath: quality.preparedPath,
        outputPath: perspPath,
      );
      if (result == perspPath) {
        await attempt(perspPath);
      }
    } catch (_) {
      // Ignore and continue.
    }

    // 2. Crop to the region that actually contains content. This is what fixes
    //    a gallery photo where the receipt occupies a corner of the frame: the
    //    crop is scaled up, so the characters get more pixels than a re-read of
    //    the full frame could ever give them.
    //
    //    Paper bounds are preferred over ink bounds: on a photo taken on a desk
    //    the keyboard is darker and larger than the print, so an ink-based box
    //    swallows it. Ink bounds remain the fallback for tightly framed shots
    //    where the paper fills the frame. OCR text bounds are the last resort,
    //    since the first pass is exactly what failed.
    ImageRegion? region;
    try {
      region = await _preprocessor.detectPaperBounds(quality.preparedPath);
    } catch (_) {
      region = null;
    }
    if (region == null) {
      try {
        region = await _preprocessor.detectInkBounds(quality.preparedPath);
      } catch (_) {
        region = null;
      }
    }
    region ??= _textBounds(raw);

    if (region != null) {
      final double imageArea = quality.width * quality.height.toDouble();
      final double regionArea = region.width * region.height;
      if (imageArea > 0 && regionArea / imageArea < 0.9) {
        try {
          final String cropPath = '${dir.path}/ocr_crop_$stamp.jpg';
          await _preprocessor.prepareCrop(
            preparedPath: quality.preparedPath,
            outputPath: cropPath,
            left: region.left,
            top: region.top,
            right: region.right,
            bottom: region.bottom,
          );
          await attempt(cropPath);
        } catch (_) {
          // Ignore and continue with the other strategies.
        }
      }
    }

    // 3. Local binarisation, for faded thermal print and uneven lighting.
    if (_isWeak(best)) {
      try {
        final String binaryPath = '${dir.path}/ocr_binary_$stamp.jpg';
        await _preprocessor.prepareBinarized(
          preparedPath: quality.preparedPath,
          outputPath: binaryPath,
        );
        await attempt(binaryPath);
      } catch (_) {
        // Ignore.
      }
    }

    onRaw(bestRaw);
    return best;
  }

  /// Bounding box of the recognized text. Used only as a fallback when pixel
  /// based ink detection cannot produce a region.
  ImageRegion? _textBounds(OcrRawResult raw) {
    if (raw.lines.isEmpty) {
      return null;
    }
    double left = double.infinity;
    double top = double.infinity;
    double right = 0;
    double bottom = 0;
    for (final OcrLine line in raw.lines) {
      left = math.min(left, line.left);
      top = math.min(top, line.top);
      right = math.max(right, line.right);
      bottom = math.max(bottom, line.bottom);
    }
    if (right <= left || bottom <= top) {
      return null;
    }
    return ImageRegion(left, top, right, bottom);
  }
}
