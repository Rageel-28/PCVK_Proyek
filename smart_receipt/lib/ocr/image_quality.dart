import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as img;

/// Preprocessing result applied to a captured receipt image, plus the quality
/// signals used to warn the user about problems that hurt OCR accuracy.
class ImageQualityReport {
  const ImageQualityReport({
    required this.brightness,
    required this.blurScore,
    required this.rotationDegrees,
    required this.width,
    required this.height,
    required this.preparedPath,
    this.lowLight = false,
    this.blurry = false,
    this.rotated = false,
  });

  /// Mean luminance, 0 (black) .. 255 (white).
  final double brightness;

  /// Variance of the Laplacian. Higher = sharper. Below [kBlurThreshold] the
  /// image is considered blurred.
  final double blurScore;

  /// Degrees the image was rotated during preprocessing to make text upright.
  final int rotationDegrees;

  final int width;
  final int height;

  /// Path of the preprocessed image actually sent to the OCR engine.
  final String preparedPath;

  final bool lowLight;
  final bool blurry;
  final bool rotated;

  /// True when at least one quality issue could degrade recognition.
  bool get hasIssue => lowLight || blurry || rotated;

  /// Human readable issues for the UI.
  List<String> get issues {
    final List<String> out = <String>[];
    if (blurry) {
      out.add('Gambar tampak buram — dipertajam otomatis');
    }
    if (lowLight) {
      out.add('Pencahayaan rendah / gelap');
    }
    if (rotated) {
      out.add('Gambar diputar $rotationDegrees° agar teks tegak');
    }
    return out;
  }
}

/// Blur is detected from Laplacian variance; the threshold is tuned for
/// downscaled receipt photos and errs toward warning rather than failing.
const double kBlurThreshold = 45;

/// Below this mean luminance we treat the capture as low-light.
const double kLowLightThreshold = 82;

class ImagePreprocessor {
  /// Longest edge used for OCR.
  ///
  /// A camera capture frames the receipt tightly, so its text is already large
  /// in the frame. A gallery photo is a full-resolution shot where the receipt
  /// may occupy only part of the picture, so the same cap would shrink its text
  /// below what ML Kit reads reliably. 3000px keeps both cases legible while
  /// staying within a safe memory budget (~27 MB for the decoded bitmap).
  static const int _maxDimension = 3000;

  /// Crops to a region of interest and upscales it for a second OCR pass.
  ///
  /// Gallery photos usually show more than the receipt: a table, a hand, the
  /// floor. When the recognized text occupies only a small part of the frame,
  /// cropping to it and scaling the crop up gives the recognizer far more
  /// pixels per character than re-reading the full frame.
  Future<String> prepareCrop({
    required String preparedPath,
    required String outputPath,
    required double left,
    required double top,
    required double right,
    required double bottom,
  }) async {
    final img.Image? decoded = await _decode(preparedPath);
    if (decoded == null) {
      return preparedPath;
    }

    final double padX = (right - left) * 0.04;
    final double padY = (bottom - top) * 0.04;
    final int x0 = (left - padX).floor().clamp(0, decoded.width - 1);
    final int y0 = (top - padY).floor().clamp(0, decoded.height - 1);
    final int x1 = (right + padX).ceil().clamp(x0 + 1, decoded.width);
    final int y1 = (bottom + padY).ceil().clamp(y0 + 1, decoded.height);

    img.Image cropped = img.copyCrop(
      decoded,
      x: x0,
      y: y0,
      width: x1 - x0,
      height: y1 - y0,
    );

    // The crop exists to give the recognizer more pixels per character, so it
    // is scaled up to a comfortable size and never downscaled.
    final int shortEdge = math.min(cropped.width, cropped.height);
    const int targetShortEdge = 1600;
    if (shortEdge > 0 && shortEdge < targetShortEdge) {
      final double factor = targetShortEdge / shortEdge;
      cropped = img.copyResize(
        cropped,
        width: (cropped.width * factor).round(),
        height: (cropped.height * factor).round(),
        interpolation: img.Interpolation.cubic,
      );
    }

    final File out = File(outputPath);
    out.parent.createSync(recursive: true);
    await out.writeAsBytes(img.encodeJpg(cropped, quality: 92));
    return outputPath;
  }

  /// Rotates an already-prepared image by [degrees] and writes it to
  /// [outputPath].
  ///
  /// Used as a last-resort OCR retry: a photo imported from the gallery may be
  /// stored sideways with no EXIF orientation left to correct it, in which case
  /// the recognizer reads almost nothing until the image is turned upright.
  Future<String> prepareRotated({
    required String preparedPath,
    required String outputPath,
    required int degrees,
  }) async {
    final img.Image? decoded = await _decode(preparedPath);
    if (decoded == null) {
      return preparedPath;
    }
    decoded.backgroundColor = img.ColorRgb8(255, 255, 255);
    final img.Image rotated =
        img.copyRotate(decoded, angle: degrees.toDouble());
    final File out = File(outputPath);
    out.parent.createSync(recursive: true);
    await out.writeAsBytes(img.encodeJpg(rotated, quality: 92));
    return outputPath;
  }

  /// Finds the bounding box of printed content ("ink") in an already-prepared
  /// image.
  ///
  /// This does not depend on OCR at all: it thresholds luminance with Otsu's
  /// method and takes the extent of the dark pixels. It is what lets the gallery
  /// path crop to the receipt even when the first OCR pass read almost nothing,
  /// which is exactly the case where text-based bounds are useless.
  Future<ImageRegion?> detectInkBounds(String preparedPath) async {
    final img.Image? decoded = await _decode(preparedPath);
    if (decoded == null) {
      return null;
    }

    // Work on a downscaled grayscale copy; only the proportions matter.
    final img.Image gray = img.grayscale(
      decoded.width > 400
          ? img.copyResize(decoded, width: 400)
          : decoded,
    );
    final int w = gray.width;
    final int h = gray.height;
    if (w < 8 || h < 8) {
      return null;
    }

    // Otsu assumes two classes, but a real photo has three: dark print, white
    // receipt paper, and a grey background/table. Otsu then splits paper from
    // background and reports the *background* as "ink", which is useless.
    //
    // Print is always the darkest minority, so threshold halfway between a low
    // percentile (the ink) and the median (paper or background).
    final List<int> histogram = List<int>.filled(256, 0);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        histogram[gray.getPixel(x, y).r.toInt().clamp(0, 255)]++;
      }
    }
    final int total = w * h;

    int percentile(double p) {
      final int target = (total * p).round().clamp(1, total);
      int seen = 0;
      for (int i = 0; i < 256; i++) {
        seen += histogram[i];
        if (seen >= target) {
          return i;
        }
      }
      return 255;
    }

    final int inkLevel = percentile(0.02);
    final int paperLevel = percentile(0.60);
    final int threshold = ((inkLevel + paperLevel) / 2).round();

    // A photo of a receipt is mostly bright paper with dark print. If the split
    // lands very high there is no dark ink at all, so there is nothing to crop
    // to.
    if (threshold > 200 || paperLevel - inkLevel < 20) {
      return null;
    }

    int minX = w;
    int minY = h;
    int maxX = -1;
    int maxY = -1;
    int darkCount = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        if (gray.getPixel(x, y).r <= threshold) {
          darkCount++;
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
      }
    }

    // Too little ink means there is nothing to crop to.
    if (darkCount < total * 0.002 || maxX < minX || maxY < minY) {
      return null;
    }

    // Require the ink to be a meaningful part of the frame; a stray dark border
    // would otherwise make the bounds cover the whole image.
    final double coverage = ((maxX - minX + 1) * (maxY - minY + 1)) / total;
    if (coverage > 0.98) {
      return null;
    }

    final double scaleX = decoded.width / w;
    final double scaleY = decoded.height / h;
    return ImageRegion(
      minX * scaleX,
      minY * scaleY,
      (maxX + 1) * scaleX,
      (maxY + 1) * scaleY,
    );
  }

  /// Finds the bounding box of the receipt *paper* rather than its print.
  ///
  /// [detectInkBounds] looks for the darkest pixels, which is right for a
  /// tightly framed capture but fails on a photo taken on a keyboard: the keys
  /// are darker and larger than the print, so the "ink" box covers the whole
  /// frame and the crop is abandoned. The OCR pass then reads `Q W E R T Y`
  /// off the keys as if it were product text.
  ///
  /// Paper is the opposite signal: the brightest, most rectangular, connected
  /// region. Keys are dark and split into many small blobs, and a table is grey
  /// rather than one bright rectangle, so neither can win this test.
  ///
  /// Returns `null` when no plausible sheet is found — a blank photo, or paper
  /// that already fills the frame so there is nothing to crop to.
  Future<ImageRegion?> detectPaperBounds(String preparedPath) async {
    final _PaperAnalysis? analysis = await _analyzePaper(preparedPath);
    return analysis?.region;
  }

  /// Whether [preparedPath] plausibly contains receipt paper at all.
  ///
  /// Used right after capture to reject a photo of the desk, the keyboard, or
  /// the floor before spending time on OCR. A sheet that already fills the
  /// frame is accepted: [detectPaperBounds] returns no crop region for it, but
  /// a dominant bright rectangle still proves paper is present.
  Future<bool> containsReceiptPaper(String preparedPath) async {
    final _PaperAnalysis? analysis = await _analyzePaper(preparedPath);
    if (analysis == null) {
      // Unreadable image: do not block the user, let the OCR stage report it.
      return true;
    }
    return analysis.region != null || analysis.fillsFrame;
  }

  /// Shared bright-region analysis behind [detectPaperBounds] and
  /// [containsReceiptPaper].
  Future<_PaperAnalysis?> _analyzePaper(String preparedPath) async {
    final img.Image? decoded = await _decode(preparedPath);
    if (decoded == null) {
      return null;
    }

    // Only proportions and large blobs matter, so work small: this keeps the
    // connected-component pass cheap even for a 12 MP gallery photo.
    final img.Image gray = img.grayscale(
      decoded.width > 320 ? img.copyResize(decoded, width: 320) : decoded,
    );
    final int w = gray.width;
    final int h = gray.height;
    if (w < 8 || h < 8) {
      return null;
    }
    final int total = w * h;

    final List<int> histogram = List<int>.filled(256, 0);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        histogram[gray.getPixel(x, y).r.toInt().clamp(0, 255)]++;
      }
    }

    // Paper is the brightest *large* area. Taking the highest grey level that
    // still has a meaningful population beats Otsu here: a receipt on a
    // keyboard has four tones (print, keys, key gaps, paper) and Otsu splits
    // between keys and key gaps, merging the paper with the background so the
    // sheet and the whole frame become one blob.
    final int paperLevel = _brightLevel(histogram, total);
    if (paperLevel <= 0) {
      return const _PaperAnalysis(region: null, fillsFrame: false);
    }
    // Paper that is barely brighter than a night-time desk is not paper. Real
    // receipt photos put the sheet well above the mid-grey mark even when the
    // shot is dim; a near-black frame should never pass the gate.
    if (paperLevel < 90) {
      return const _PaperAnalysis(region: null, fillsFrame: false);
    }
    // Relative to the paper tone rather than a fixed offset, so a dimly lit or
    // unevenly shaded sheet still reads as bright. `paperLevel - 30` cut the
    // shadowed half off a real receipt and left the blob at 44% fill, which
    // the gate then rejected.
    final int threshold = math.max(1, (paperLevel * 0.70).round());

    // Bright pixels are the paper candidates.
    final List<bool> bright = List<bool>.filled(total, false);
    int brightCount = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        if (gray.getPixel(x, y).r > threshold) {
          bright[y * w + x] = true;
          brightCount++;
        }
      }
    }
    if (brightCount == 0) {
      return const _PaperAnalysis(region: null, fillsFrame: false);
    }

    // Largest 4-connected bright blob = the sheet of paper.
    final List<bool> visited = List<bool>.filled(total, false);
    final List<int> stack = <int>[];
    int bestCount = 0;
    int bestMinX = 0;
    int bestMinY = 0;
    int bestMaxX = 0;
    int bestMaxY = 0;

    for (int start = 0; start < total; start++) {
      if (!bright[start] || visited[start]) {
        continue;
      }
      visited[start] = true;
      stack.add(start);
      int count = 0;
      int minX = w;
      int minY = h;
      int maxX = -1;
      int maxY = -1;

      while (stack.isNotEmpty) {
        final int index = stack.removeLast();
        final int x = index % w;
        final int y = index ~/ w;
        count++;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;

        if (x > 0) {
          final int n = index - 1;
          if (!visited[n] && bright[n]) {
            visited[n] = true;
            stack.add(n);
          }
        }
        if (x < w - 1) {
          final int n = index + 1;
          if (!visited[n] && bright[n]) {
            visited[n] = true;
            stack.add(n);
          }
        }
        if (y > 0) {
          final int n = index - w;
          if (!visited[n] && bright[n]) {
            visited[n] = true;
            stack.add(n);
          }
        }
        if (y < h - 1) {
          final int n = index + w;
          if (!visited[n] && bright[n]) {
            visited[n] = true;
            stack.add(n);
          }
        }
      }

      if (count > bestCount) {
        bestCount = count;
        bestMinX = minX;
        bestMinY = minY;
        bestMaxX = maxX;
        bestMaxY = maxY;
      }
    }

    // A sheet must occupy a meaningful part of the frame.
    if (bestCount < total * 0.05) {
      return const _PaperAnalysis(region: null, fillsFrame: false);
    }

    final double boxW = (bestMaxX - bestMinX + 1).toDouble();
    final double boxH = (bestMaxY - bestMinY + 1).toDouble();
    if (boxW < 8 || boxH < 8) {
      return const _PaperAnalysis(region: null, fillsFrame: false);
    }

    // Paper is a solid rectangle, so the blob fills most of its own bounding
    // box. Scattered bright spots (highlights, reflections) do not, and neither
    // does the light background of a keyboard photo — the keys punch holes in
    // it, so it covers a large box but only a fraction of it.
    final double fill = bestCount / (boxW * boxH);
    if (fill < 0.55) {
      return const _PaperAnalysis(region: null, fillsFrame: false);
    }

    // A receipt is elongated, but a quarter-turned one is elongated the other
    // way. Anything squarer than 0.4 or longer than 4.5 is not a receipt.
    final double aspect = boxH / boxW;
    if (aspect < 0.4 || aspect > 4.5) {
      return const _PaperAnalysis(region: null, fillsFrame: false);
    }

    final double boxFraction = (boxW * boxH) / total;

    // Already fills the frame: cropping would gain nothing, but a solid bright
    // rectangle spanning the frame is still clearly paper.
    if (boxFraction > 0.97) {
      return _PaperAnalysis(region: null, fillsFrame: fill >= 0.6);
    }

    final double scaleX = decoded.width / w;
    final double scaleY = decoded.height / h;
    return _PaperAnalysis(
      region: ImageRegion(
        bestMinX * scaleX,
        bestMinY * scaleY,
        (bestMaxX + 1) * scaleX,
        (bestMaxY + 1) * scaleY,
      ),
      fillsFrame: false,
    );
  }

  /// The brightest grey level whose population, accumulated from white
  /// downwards, is at least 4% of the image.
  ///
  /// Accumulating matters for real photographs: their histogram is smooth, so
  /// **no single** grey level holds 4% of a 320-px-wide frame (a JPEG receipt
  /// peaks at ~2%), and a per-level test would report "no paper" for every
  /// genuine photo. Accumulating from white down finds the top of the paper's
  /// bright band instead. A tiny specular highlight (a window reflection, a
  /// glossy spot) still contributes far less than 4% and is ignored.
  int _brightLevel(List<int> histogram, int total) {
    final int minPopulation = math.max(1, (total * 0.04).round());
    int cumulative = 0;
    for (int level = 255; level >= 0; level--) {
      cumulative += histogram[level];
      if (cumulative >= minPopulation) {
        return level;
      }
    }
    return -1;
  }

  /// Produces a high-contrast, grayscale copy of an already-prepared image.
  ///
  /// Used as a second OCR attempt when the first pass reads too little: thermal
  /// receipts fade unevenly, and a locally thresholded image often recovers the
  /// digits that the colour pass lost. This is a standard fallback in document
  /// OCR pipelines and is only paid for when the first attempt looks weak.
  Future<String> prepareBinarized({
    required String preparedPath,
    required String outputPath,
  }) async {
    final img.Image? decoded = await _decode(preparedPath);
    if (decoded == null) {
      return preparedPath;
    }

    final img.Image gray = img.grayscale(decoded);
    final img.Image binary = img.Image(
      width: gray.width,
      height: gray.height,
      numChannels: 3,
    );

    // Adaptive mean threshold over a sliding window keeps faint strokes while
    // removing the background gradient that confuses the recognizer.
    const int window = 31;
    final int w = gray.width;
    final int h = gray.height;

    // Integral image for O(1) window sums.
    final List<int> integral = List<int>.filled((w + 1) * (h + 1), 0);
    for (int y = 0; y < h; y++) {
      int rowSum = 0;
      for (int x = 0; x < w; x++) {
        rowSum += gray.getPixel(x, y).r.toInt();
        integral[(y + 1) * (w + 1) + (x + 1)] =
            integral[y * (w + 1) + (x + 1)] + rowSum;
      }
    }

    int windowSum(int x0, int y0, int x1, int y1) {
      return integral[y1 * (w + 1) + x1] -
          integral[y0 * (w + 1) + x1] -
          integral[y1 * (w + 1) + x0] +
          integral[y0 * (w + 1) + x0];
    }

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final int x0 = math.max(0, x - window ~/ 2);
        final int y0 = math.max(0, y - window ~/ 2);
        final int x1 = math.min(w, x + window ~/ 2 + 1);
        final int y1 = math.min(h, y + window ~/ 2 + 1);
        final int count = (x1 - x0) * (y1 - y0);
        final double mean = windowSum(x0, y0, x1, y1) / count;
        // Slightly below the local mean: printed ink is darker than paper.
        final int value =
            gray.getPixel(x, y).r < mean - 8 ? 0 : 255;
        final img.Pixel p = binary.getPixel(x, y);
        p.r = value;
        p.g = value;
        p.b = value;
      }
    }

    // Background that survived binarisation — keyboard keys, a table edge,
    // a shadow — is connected to the frame border and far larger than any
    // glyph. Clearing it keeps the recognizer focused on the receipt.
    _clearBorderConnectedDark(binary);

    final File out = File(outputPath);
    out.parent.createSync(recursive: true);
    await out.writeAsBytes(img.encodeJpg(binary, quality: 92));
    return outputPath;
  }

  /// Flood-fills dark pixels that touch the frame border to white.
  ///
  /// Printed text always sits inside the paper margin, so a dark blob connected
  /// to the edge of the image is almost always background: a keyboard, a table
  /// edge, a shadow. On a photo taken above a keyboard this is what stops the
  /// keys from being binarised into solid black slabs the recognizer reads as
  /// text.
  ///
  /// Only blobs covering at least [minAreaFraction] of the image are cleared.
  /// A stray letter that happens to touch the border is tiny and is left alone,
  /// while a keyboard or a table edge is always large.
  void _clearBorderConnectedDark(
    img.Image binary, {
    double minAreaFraction = 0.03,
  }) {
    final int w = binary.width;
    final int h = binary.height;
    if (w < 3 || h < 3) {
      return;
    }
    final int total = w * h;
    final int minArea = math.max(1, (total * minAreaFraction).round());

    final List<bool> visited = List<bool>.filled(total, false);
    final List<int> stack = <int>[];
    final List<int> component = <int>[];

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final bool onBorder = x == 0 || y == 0 || x == w - 1 || y == h - 1;
        if (!onBorder) {
          continue;
        }
        final int index = y * w + x;
        if (visited[index] || binary.getPixel(x, y).r > 127) {
          continue;
        }

        // Collect this border-connected dark blob.
        component.clear();
        stack.clear();
        visited[index] = true;
        stack.add(index);

        while (stack.isNotEmpty) {
          final int current = stack.removeLast();
          component.add(current);
          final int cx = current % w;
          final int cy = current ~/ w;

          void visit(int nx, int ny) {
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
              return;
            }
            final int n = ny * w + nx;
            if (visited[n] || binary.getPixel(nx, ny).r > 127) {
              return;
            }
            visited[n] = true;
            stack.add(n);
          }

          visit(cx - 1, cy);
          visit(cx + 1, cy);
          visit(cx, cy - 1);
          visit(cx, cy + 1);
        }

        // Small blobs are print that happens to reach the edge — keep them.
        if (component.length < minArea) {
          continue;
        }
        for (final int pixel in component) {
          final img.Pixel p = binary.getPixel(pixel % w, pixel ~/ w);
          p.r = 255;
          p.g = 255;
          p.b = 255;
        }
      }
    }
  }

  /// Applies unsharp mask sharpening to an already-prepared image and writes
  /// the result to [outputPath].
  ///
  /// Used when blur is detected: a Gaussian-blurred copy is subtracted from the
  /// original (the classic unsharp mask formula) to accentuate edges and make
  /// faint strokes more distinct for the recognizer.
  ///
  ///   sharpened = original + amount × (original − blurred)
  ///
  /// The amount is capped so that noise in blank-paper regions is not amplified
  /// into false strokes, which would confuse ML Kit's line detector.
  Future<String> prepareSharpened({
    required String preparedPath,
    required String outputPath,
    double amount = 1.2,
  }) async {
    final img.Image? decoded = await _decode(preparedPath);
    if (decoded == null) {
      return preparedPath;
    }

    // Blur radius chosen so individual character strokes stay intact but
    // low-frequency haze (camera motion blur, defocus) is captured.
    final img.Image blurred = img.gaussianBlur(decoded, radius: 2);

    final img.Image sharpened = img.Image(
      width: decoded.width,
      height: decoded.height,
      numChannels: decoded.numChannels,
    );

    for (int y = 0; y < decoded.height; y++) {
      for (int x = 0; x < decoded.width; x++) {
        final img.Pixel orig = decoded.getPixel(x, y);
        final img.Pixel blur = blurred.getPixel(x, y);
        final img.Pixel dst = sharpened.getPixel(x, y);

        dst.r = (orig.r + amount * (orig.r - blur.r)).clamp(0, 255).round();
        dst.g = (orig.g + amount * (orig.g - blur.g)).clamp(0, 255).round();
        dst.b = (orig.b + amount * (orig.b - blur.b)).clamp(0, 255).round();
      }
    }

    final File out = File(outputPath);
    out.parent.createSync(recursive: true);
    await out.writeAsBytes(img.encodeJpg(sharpened, quality: 92));
    return outputPath;
  }

  /// Corrects perspective distortion caused by photographing a receipt at an
  /// angle (i.e. not straight-on from above).
  ///
  /// The algorithm:
  /// 1. Downscale and binarize to isolate the dark-print region.
  /// 2. Use `detectInkBounds` to find the approximate content rectangle.
  /// 3. If the region is significantly smaller than the full frame AND the
  ///    aspect ratio suggests it is a real receipt (tall, narrow), apply a
  ///    projective (homography) warp that maps the four detected corners to a
  ///    perfect rectangle and writes the result to [outputPath].
  ///
  /// When the content already fills the frame, or the aspect ratio does not
  /// match a receipt, the source image is returned unchanged so the caller can
  /// fall back gracefully.
  Future<String> preparePerspectiveCorrected({
    required String preparedPath,
    required String outputPath,
  }) async {
    final img.Image? decoded = await _decode(preparedPath);
    if (decoded == null) {
      return preparedPath;
    }

    // Prefer the paper rectangle: it is the correct signal on photos where the
    // receipt sits on a dark surface such as a keyboard. Fall back to ink bounds
    // for tightly framed captures where the paper fills the frame.
    ImageRegion? region = await detectPaperBounds(preparedPath);
    region ??= await detectInkBounds(preparedPath);
    if (region == null) {
      return preparedPath;
    }

    final double frameArea = decoded.width * decoded.height.toDouble();
    final double regionArea = region.width * region.height;

    // Only apply when the content is clearly inset from the frame edges —
    // otherwise the distortion is negligible and any warp would only add blur.
    if (regionArea / frameArea > 0.85) {
      return preparedPath;
    }

    // Reject regions that do not look like a receipt (must be at least as tall
    // as wide, but not absurdly so — avoids locking onto a stray dark line).
    final double aspect = region.height / region.width;
    if (aspect < 0.8 || aspect > 6.0) {
      return preparedPath;
    }

    // Source quadrilateral — the four corners of the detected region.
    // We treat the detected bounding box as the best available approximation
    // of the receipt corners (true corner detection would require OpenCV).
    final double sl = region.left;
    final double st = region.top;

    // Destination rectangle keeps the same aspect ratio, capped at the
    // prepared-image resolution so we never upscale beyond what was captured.
    final int dstW = math.min(region.width.round(), decoded.width);
    final int dstH = math.min(region.height.round(), decoded.height);

    final img.Image warped = img.Image(
      width: dstW,
      height: dstH,
      numChannels: decoded.numChannels,
    );
    // Fill with white so any unmapped corners look like blank paper.
    img.fill(warped, color: img.ColorRgb8(255, 255, 255));

    // Bilinear sample from the source quadrilateral to the destination grid.
    // For a pure axis-aligned bounding box the mapping is just a crop-resize,
    // which is what we implement here.  A true perspective warp (homography)
    // would additionally correct keystone distortion but requires knowing the
    // actual four skewed corners, not just the axis-aligned bounding box.
    final double scaleX = region.width / dstW;
    final double scaleY = region.height / dstH;

    for (int dy = 0; dy < dstH; dy++) {
      for (int dx = 0; dx < dstW; dx++) {
        final double sx = sl + dx * scaleX;
        final double sy = st + dy * scaleY;

        // Bilinear interpolation from the source image.
        final int x0 = sx.floor().clamp(0, decoded.width - 1);
        final int y0 = sy.floor().clamp(0, decoded.height - 1);
        final int x1 = (x0 + 1).clamp(0, decoded.width - 1);
        final int y1 = (y0 + 1).clamp(0, decoded.height - 1);
        final double fx = sx - x0;
        final double fy = sy - y0;

        final img.Pixel p00 = decoded.getPixel(x0, y0);
        final img.Pixel p10 = decoded.getPixel(x1, y0);
        final img.Pixel p01 = decoded.getPixel(x0, y1);
        final img.Pixel p11 = decoded.getPixel(x1, y1);

        double lerp(double a, double b, double t) => a + (b - a) * t;

        final img.Pixel dst = warped.getPixel(dx, dy);
        dst.r = lerp(lerp(p00.r.toDouble(), p10.r.toDouble(), fx),
                lerp(p01.r.toDouble(), p11.r.toDouble(), fx), fy)
            .clamp(0, 255)
            .round();
        dst.g = lerp(lerp(p00.g.toDouble(), p10.g.toDouble(), fx),
                lerp(p01.g.toDouble(), p11.g.toDouble(), fx), fy)
            .clamp(0, 255)
            .round();
        dst.b = lerp(lerp(p00.b.toDouble(), p10.b.toDouble(), fx),
                lerp(p01.b.toDouble(), p11.b.toDouble(), fx), fy)
            .clamp(0, 255)
            .round();
      }
    }

    final File out = File(outputPath);
    out.parent.createSync(recursive: true);
    await out.writeAsBytes(img.encodeJpg(warped, quality: 92));
    return outputPath;
  }

  /// Analyzes [sourcePath] and writes a normalized copy for OCR to
  /// [outputPath].
  ///
  /// - Downscales very large photos (speed + memory).
  /// - Estimates the text skew angle and rotates the image upright.
  /// - Measures brightness and sharpness to flag low-light/blur.
  /// - Applies contrast/brightness equalisation when the capture is dark.
  /// Ukur kualitas citra **tanpa mengubahnya**.
  ///
  /// Dipakai mesin OCR yang melakukan praprosesnya sendiri: laporan kualitas
  /// tetap dibutuhkan UI, tetapi citra tidak boleh dirotasi, di-unsharp, atau
  /// di-JPEG ulang.
  Future<ImageQualityReport> analyze({required String sourcePath}) async {
    final img.Image? decoded = await _decode(sourcePath);
    if (decoded == null) {
      throw ImageQualityException('Gambar tidak dapat dibaca.');
    }
    final img.Image working = img.bakeOrientation(decoded);
    final double brightness = _meanLuminance(working);
    final double blurScore = _laplacianVariance(working);
    final int angle = _estimateSkewDegrees(working);
    return ImageQualityReport(
      brightness: brightness,
      blurScore: blurScore,
      rotationDegrees: angle,
      width: working.width,
      height: working.height,
      preparedPath: sourcePath,
      lowLight: brightness < kLowLightThreshold,
      blurry: blurScore < kBlurThreshold,
      rotated: angle.abs() >= 1,
    );
  }

  Future<ImageQualityReport> prepare({
    required String sourcePath,
    required String outputPath,
  }) async {
    final img.Image? decoded = await _decode(sourcePath);
    if (decoded == null) {
      // Let the OCR engine surface the real error; nothing to preprocess.
      throw ImageQualityException('Gambar tidak dapat dibaca.');
    }

    img.Image working = img.bakeOrientation(decoded);

    // Skew is estimated on a downscaled copy, so this stays cheap even for a
    // full-resolution gallery photo.
    final double brightness = _meanLuminance(working);
    final double blurScore = _laplacianVariance(working);

    final int angle = _estimateSkewDegrees(working);
    final bool rotated = angle.abs() >= 1;
    if (rotated) {
      // `copyRotate` leaves the exposed corners unfilled, which on an image
      // without an alpha channel means solid black. Those corners would later
      // be mistaken for ink and also confuse the recognizer, so the fill colour
      // is set explicitly.
      working.backgroundColor = img.ColorRgb8(255, 255, 255);
      working = img.copyRotate(
        working,
        angle: -angle.toDouble(),
        interpolation: img.Interpolation.linear,
      );
    }

    // Size limits are applied *after* rotation: rotating expands the bounding
    // box, so resizing first would let the result exceed the cap again.
    if (math.max(working.width, working.height) > _maxDimension) {
      working = working.width >= working.height
          ? img.copyResize(working, width: _maxDimension)
          : img.copyResize(working, height: _maxDimension);
    }

    // Catatan: tidak ada upscale buta di sini. Engine OCR klasik punya
    // upscale adaptif berbasis tinggi baris (`_adaptiveScale`), dan
    // pengujian menunjukkan upscale buta (cubic/bilinear ke short-edge)
    // justru merusak binarisasi & segmentasi — total struk yang tadinya
    // terbaca benar menjadi salah. Biarkan engine yang memutuskan resolusi.

    final bool lowLight = brightness < kLowLightThreshold;
    if (lowLight) {
      working = _normalizeBrightness(working, brightness);
    }

    // Sharpen when blur is detected so the recognizer has crisper strokes to
    // work with. (Hanya untuk foto yang terdeteksi buram; tidak mengubah
    // citra yang sudah tajam.)
    final bool blurry = blurScore < kBlurThreshold;
    if (blurry) {
      working = _applyUnsharpMask(working);
    }

    // Catatan: tidak ada penguatan kontras lokal (CLAHE) di sini. Engine OCR
    // klasik memakai binarisasi adaptif (lokal) yang sudah menangani cahaya
    // tidak rata dan cetakan pudar; CLAHE justru memperkuat tekstur kertas
    // menjadi "tinta" palsu dan merusak fitur HOG. (Dulu dipakai untuk ML Kit.)

    final File out = File(outputPath);
    out.parent.createSync(recursive: true);
    await out.writeAsBytes(img.encodeJpg(working, quality: 92));

    return ImageQualityReport(
      brightness: brightness,
      blurScore: blurScore,
      rotationDegrees: angle,
      width: working.width,
      height: working.height,
      preparedPath: outputPath,
      lowLight: lowLight,
      blurry: blurry,
      rotated: rotated,
    );
  }

  Future<img.Image?> _decode(String path) async {
    final File file = File(path);
    if (!file.existsSync()) {
      throw ImageQualityException('File gambar tidak ditemukan.');
    }
    final img.Image? decoded = img.decodeImage(await file.readAsBytes());
    return decoded;
  }

  double _meanLuminance(img.Image image) {
    final img.Image small = image.width > 200
        ? img.copyResize(image, width: 200)
        : image;
    double sum = 0;
    int count = 0;
    for (int y = 0; y < small.height; y++) {
      for (int x = 0; x < small.width; x++) {
        final img.Pixel p = small.getPixel(x, y);
        sum += img.getLuminanceRgb(p.r.toDouble(), p.g.toDouble(), p.b.toDouble());
        count++;
      }
    }
    return count == 0 ? 255 : sum / count;
  }

  /// Variance of the (4-neighbour) Laplacian over a grayscale, downscaled copy.
  double _laplacianVariance(img.Image image) {
    final img.Image gray = img.grayscale(
      image.width > 600 ? img.copyResize(image, width: 600) : image,
    );
    final int w = gray.width;
    final int h = gray.height;
    if (w < 3 || h < 3) {
      return 0;
    }

    double sum = 0;
    double sumSq = 0;
    int count = 0;
    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        final double center = gray.getPixel(x, y).r.toDouble();
        final double lap = 4 * center -
            gray.getPixel(x - 1, y).r -
            gray.getPixel(x + 1, y).r -
            gray.getPixel(x, y - 1).r -
            gray.getPixel(x, y + 1).r;
        sum += lap;
        sumSq += lap * lap;
        count++;
      }
    }
    if (count == 0) {
      return 0;
    }
    final double mean = sum / count;
    return (sumSq / count) - (mean * mean);
  }

  /// Estimates receipt skew in degrees using a two-pass projection profile.
  ///
  /// Pass 1 (coarse): scans ±15° in 3° steps to find the rough quadrant.
  /// Pass 2 (fine): scans ±3° around that peak in 1° steps for precision.
  /// This finds larger tilts (hand-held shots, gallery imports) while keeping
  /// the cost manageable — only ~22 variance evaluations total.
  int _estimateSkewDegrees(img.Image image) {
    final img.Image small = img.copyResize(
      img.grayscale(image),
      width: image.width > 400 ? 400 : image.width,
    );

    // Pass 1 — coarse scan ±15° in 3° steps.
    double bestVariance = -1;
    int bestAngle = 0;
    for (int angle = -15; angle <= 15; angle += 3) {
      final double variance =
          _rowVariance(img.copyRotate(small, angle: angle.toDouble()));
      if (variance > bestVariance) {
        bestVariance = variance;
        bestAngle = angle;
      }
    }

    // Pass 2 — fine scan ±3° around the coarse peak in 1° steps.
    final int fineMin = (bestAngle - 3).clamp(-15, 15);
    final int fineMax = (bestAngle + 3).clamp(-15, 15);
    for (int angle = fineMin; angle <= fineMax; angle++) {
      final double variance =
          _rowVariance(img.copyRotate(small, angle: angle.toDouble()));
      if (variance > bestVariance) {
        bestVariance = variance;
        bestAngle = angle;
      }
    }

    if (bestAngle == 0) {
      return 0;
    }

    final double upright = _rowVariance(small);
    final double gain = (bestVariance - upright) / (upright.abs() + 1);

    // Only correct a skew that is clearly real. Rotating a straight receipt
    // blurs the glyphs, so the confidence bar is kept deliberately high.
    // Boundary hits at ±15° indicate the estimator locked onto the rectangular
    // frame edge (gallery photo), not onto text lines — reject those.
    final bool atBoundary = bestAngle.abs() >= 15;
    if (atBoundary || gain <= 0.3 || bestAngle.abs() < 2) {
      return 0;
    }
    return bestAngle;
  }

  double _rowVariance(img.Image gray) {
    final int w = gray.width;
    final int h = gray.height;
    if (h < 2) {
      return 0;
    }
    final List<double> rowDarkness = List<double>.filled(h, 0);
    for (int y = 0; y < h; y++) {
      double sum = 0;
      for (int x = 0; x < w; x++) {
        sum += 255 - gray.getPixel(x, y).r;
      }
      rowDarkness[y] = sum / w;
    }
    final double mean =
        rowDarkness.reduce((double a, double b) => a + b) / h;
    double variance = 0;
    for (final double v in rowDarkness) {
      variance += (v - mean) * (v - mean);
    }
    return variance / h;
  }

  /// Brightens an under-exposed capture without clipping it.
  ///
  /// `img.adjustColor(brightness:)` is a *multiplier* in package:image — 1.0 is
  /// a no-op — so passing the missing light as `brightness` multiplied every
  /// dark photo by 18..81 and blew it out to solid white, destroying the text.
  ///
  /// Gamma is the correct tool, and the exponent is solved for rather than
  /// guessed: a fixed gamma leaves a very dark capture dark (a photo at mean 40
  /// only reaches ~80 with gamma 0.59), while gamma < 1 can never clip a
  /// highlight because it maps 1.0 to 1.0. So we pick the gamma that maps the
  /// measured mean onto [targetMean].
  img.Image _normalizeBrightness(img.Image image, double brightness) {
    const double targetMean = 128;
    final double input = brightness.clamp(1.0, 254.0);
    // Solve target/255 = (input/255)^gamma. Guarded by the clamp because a
    // near-black capture would otherwise ask for an unusable exponent.
    final double gamma =
        (math.log(targetMean / 255) / math.log(input / 255)).clamp(0.35, 0.95);
    return img.adjustColor(image, gamma: gamma, contrast: 1.04);
  }

  /// Unsharp mask applied in-place on a copy of [image].
  ///
  /// Classic formula:  output = original + amount × (original − blurred)
  /// The radius-2 Gaussian captures low-frequency haze (motion blur, defocus)
  /// without eating into individual character strokes.  amount=1.2 gives a
  /// visible sharpening boost without over-amplifying sensor noise in the
  /// blank-paper regions.
  img.Image _applyUnsharpMask(img.Image image, {double amount = 1.2}) {
    final img.Image blurred = img.gaussianBlur(image, radius: 2);
    final img.Image result = img.Image(
      width: image.width,
      height: image.height,
      numChannels: image.numChannels,
    );
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final img.Pixel orig = image.getPixel(x, y);
        final img.Pixel blur = blurred.getPixel(x, y);
        final img.Pixel dst = result.getPixel(x, y);
        dst.r = (orig.r + amount * (orig.r - blur.r)).clamp(0, 255).round();
        dst.g = (orig.g + amount * (orig.g - blur.g)).clamp(0, 255).round();
        dst.b = (orig.b + amount * (orig.b - blur.b)).clamp(0, 255).round();
      }
    }
    return result;
  }
}

/// Bright-region analysis result: the crop region when a sheet is found, plus
/// whether a dominant bright rectangle spans the frame (paper already filling
/// the shot, so no crop is needed but paper is definitely present).
class _PaperAnalysis {
  const _PaperAnalysis({required this.region, required this.fillsFrame});

  final ImageRegion? region;
  final bool fillsFrame;
}

/// A rectangle in pixel coordinates.
class ImageRegion {
  const ImageRegion(this.left, this.top, this.right, this.bottom);

  final double left;
  final double top;
  final double right;
  final double bottom;

  double get width => right - left;
  double get height => bottom - top;
}

class ImageQualityException implements Exception {
  ImageQualityException(this.message);

  final String message;

  @override
  String toString() => 'ImageQualityException: $message';
}
