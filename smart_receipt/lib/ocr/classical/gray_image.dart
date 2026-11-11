import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// Citra grayscale 8-bit dengan buffer datar.
///
/// Pipeline OCR klasik melakukan banyak operasi per piksel (gradien,
/// jendela Sauvola, proyeksi). Mengakses `img.Image.getPixel` per piksel
/// sangat lambat karena melibatkan beberapa pemanggilan method. Buffer
/// datar `Uint8List` membuat semua operasi hanya berupa indeks array.
class GrayImage {
  GrayImage(this.width, this.height, this.data)
      : assert(data.length == width * height, 'panjang data harus w*h');

  GrayImage.zeros(int width, int height, [int value = 0])
      : width = width,
        height = height,
        data = Uint8List(width * height)..fillRange(0, width * height, value);

  final int width;
  final int height;

  /// Satu byte per piksel, baris demi baris (row-major).
  final Uint8List data;

  int get length => data.length;

  int at(int x, int y) => data[y * width + x];

  void set(int x, int y, int value) {
    data[y * width + x] = value;
  }

  /// Konversi dari `img.Image` memakai luma Rec.601.
  ///
  /// Penting: PNG grayscale didekode menjadi citra **satu kanal**, di mana
  /// `pixel.g` dan `pixel.b` bernilai 0. Menghitung luma dari ketiga kanal
  /// pada citra seperti itu hanya menyisakan 0.299 x nilai, sehingga
  /// seluruh citra menjadi jauh lebih gelap. Karena itu citra dinormalkan
  /// dulu ke tiga kanal lewat `img.grayscale`, lalu nilai merah dibaca.
  factory GrayImage.fromImage(img.Image source) {
    final img.Image graySource =
        source.numChannels == 1 ? source : img.grayscale(source);

    final int w = graySource.width;
    final int h = graySource.height;
    final GrayImage out = GrayImage.zeros(w, h);

    for (int y = 0; y < h; y++) {
      final int rowBase = y * w;
      for (int x = 0; x < w; x++) {
        final img.Pixel p = graySource.getPixel(x, y);
        out.data[rowBase + x] = p.r.toInt().clamp(0, 255);
      }
    }
    return out;
  }

  img.Image toImage() {
    final img.Image out = img.Image(width: width, height: height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int v = data[y * width + x];
        out.setPixelRgb(x, y, v, v, v);
      }
    }
    return out;
  }

  GrayImage clone() => GrayImage(width, height, Uint8List.fromList(data));

  /// Potong area persegi panjang. Koordinat dijepit ke batas citra.
  GrayImage crop(int left, int top, int right, int bottom) {
    final int l = left.clamp(0, width - 1);
    final int t = top.clamp(0, height - 1);
    final int r = right.clamp(l, width - 1);
    final int b = bottom.clamp(t, height - 1);
    final int cw = r - l + 1;
    final int ch = b - t + 1;
    final GrayImage out = GrayImage.zeros(cw, ch);
    for (int y = 0; y < ch; y++) {
      final int srcStart = (t + y) * width + l;
      out.data.setRange(y * cw, y * cw + cw, data, srcStart);
    }
    return out;
  }

  /// Ubah ukuran dengan interpolasi bilinear.
  GrayImage resized(int newWidth, int newHeight) {
    if (newWidth == width && newHeight == height) {
      return clone();
    }
    final GrayImage out = GrayImage.zeros(newWidth, newHeight);
    if (newWidth == 0 || newHeight == 0) {
      return out;
    }

    final double sx = width / newWidth;
    final double sy = height / newHeight;

    for (int y = 0; y < newHeight; y++) {
      final double fy = (y + 0.5) * sy - 0.5;
      final int y0 = fy.floor().clamp(0, height - 1);
      final int y1 = (y0 + 1).clamp(0, height - 1);
      final double wy = (fy - y0).clamp(0.0, 1.0);

      for (int x = 0; x < newWidth; x++) {
        final double fx = (x + 0.5) * sx - 0.5;
        final int x0 = fx.floor().clamp(0, width - 1);
        final int x1 = (x0 + 1).clamp(0, width - 1);
        final double wx = (fx - x0).clamp(0.0, 1.0);

        final double v00 = data[y0 * width + x0].toDouble();
        final double v01 = data[y0 * width + x1].toDouble();
        final double v10 = data[y1 * width + x0].toDouble();
        final double v11 = data[y1 * width + x1].toDouble();

        final double top = v00 + (v01 - v00) * wx;
        final double bottom = v10 + (v11 - v10) * wx;
        out.data[y * newWidth + x] =
            (top + (bottom - top) * wy).round().clamp(0, 255);
      }
    }
    return out;
  }

  /// Putar citra tanpa memperbesar kanvas.
  ///
  /// Konvensi mengikuti `cv2.warpAffine` dengan matriks dari
  /// `cv2.getRotationMatrix2D`: sudut positif = berlawanan arah jarum jam,
  /// dan piksel tujuan mengambil sampel dari piksel sumber terdekat
  /// (INTER_NEAREST). Arah pemetaan ini penting agar estimasi sudut dan
  /// penerapannya konsisten dengan versi Python.
  GrayImage rotated(double degrees, {int borderValue = 0}) {
    if (degrees.abs() < 1e-9) {
      return clone();
    }
    final double rad = degrees * math.pi / 180.0;
    final double cosA = math.cos(rad);
    final double sinA = math.sin(rad);

    final double cx = (width - 1) / 2.0;
    final double cy = (height - 1) / 2.0;

    final GrayImage out = GrayImage.zeros(width, height, borderValue);

    for (int y = 0; y < height; y++) {
      final double dy = y - cy;
      for (int x = 0; x < width; x++) {
        final double dx = x - cx;
        // Kebalikan rotasi OpenCV: src = M^-1 * dst.
        final double srcX = cosA * dx - sinA * dy + cx;
        final double srcY = sinA * dx + cosA * dy + cy;
        final int sx = srcX.round();
        final int sy = srcY.round();
        if (sx < 0 || sy < 0 || sx >= width || sy >= height) {
          continue;
        }
        out.data[y * width + x] = data[sy * width + sx];
      }
    }
    return out;
  }

  /// Ubah ukuran dengan interpolasi bicubic, meniru `cv2.INTER_CUBIC`.
  ///
  /// OpenCV memakai kernel Catmull-Rom dengan a = -0.75 dan perataan
  /// piksel di tengah (`src = (dst + 0.5) * scale - 0.5`). Binarisasi dan
  /// segmentasi sangat sensitif terhadap perbedaan sub-piksel, jadi
  /// interpolasi bilinear tidak cukup untuk mencapai paritas.
  GrayImage resizedBicubic(int newWidth, int newHeight) {
    if (newWidth == width && newHeight == height) {
      return clone();
    }
    final GrayImage out = GrayImage.zeros(newWidth, newHeight);
    if (newWidth == 0 || newHeight == 0) {
      return out;
    }

    final double sx = width / newWidth;
    final double sy = height / newHeight;

    final List<double> wx = List<double>.filled(4, 0);
    final List<int> ix = List<int>.filled(4, 0);
    final List<double> wy = List<double>.filled(4, 0);
    final List<int> iy = List<int>.filled(4, 0);

    for (int y = 0; y < newHeight; y++) {
      final double fy = (y + 0.5) * sy - 0.5;
      final int y0 = fy.floor();
      for (int k = 0; k < 4; k++) {
        iy[k] = (y0 - 1 + k).clamp(0, height - 1);
        wy[k] = _cubicWeight(fy - (y0 - 1 + k));
      }

      for (int x = 0; x < newWidth; x++) {
        final double fx = (x + 0.5) * sx - 0.5;
        final int x0 = fx.floor();
        for (int k = 0; k < 4; k++) {
          ix[k] = (x0 - 1 + k).clamp(0, width - 1);
          wx[k] = _cubicWeight(fx - (x0 - 1 + k));
        }

        double sum = 0;
        for (int j = 0; j < 4; j++) {
          final int rowBase = iy[j] * width;
          double rowSum = 0;
          for (int i = 0; i < 4; i++) {
            rowSum += data[rowBase + ix[i]] * wx[i];
          }
          sum += rowSum * wy[j];
        }
        out.data[y * newWidth + x] = sum.round().clamp(0, 255);
      }
    }
    return out;
  }

  /// Kernel bicubic Catmull-Rom dengan a = -0.75 (nilai OpenCV).
  static double _cubicWeight(double t) {
    const double a = -0.75;
    final double x = t.abs();
    if (x <= 1.0) {
      return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0) {
      return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    }
    return 0.0;
  }

  /// Hitung histogram 256 bin.
  List<int> histogram() {
    final List<int> hist = List<int>.filled(256, 0);
    for (final int v in data) {
      hist[v]++;
    }
    return hist;
  }
}
