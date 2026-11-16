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

  /// Ubah ukuran dengan interpolasi bicubic, meniru `cv2.INTER_CUBIC` bit-per-bit.
  ///
  /// OpenCV memakai kernel Catmull-Rom (a = -0.75), perataan piksel di tengah
  /// (`src = (dst + 0.5) * scale - 0.5`), batas `BORDER_REFLECT_101`, dan
  /// **aritmetika fixed-point 11 bit** untuk koefisiennya. Langkah terakhir
  /// itu yang menentukan: pembulatan `cvRound(coef * 2048)` menggeser nilai
  /// hasil sampai satu satuan intensitas dibanding perhitungan floating
  /// point biasa, dan pergeseran sekecil itu cukup membalik prediksi tanda
  /// baca setinggi 3 px. Karena itu implementasi ini meniru fixed-point-nya.
  GrayImage resizedBicubic(int newWidth, int newHeight) {
    if (newWidth == width && newHeight == height) {
      return clone();
    }
    final GrayImage out = GrayImage.zeros(newWidth, newHeight);
    if (newWidth == 0 || newHeight == 0) {
      return out;
    }

    const int coefScale = 1 << 11;
    const int coefShift = 22;
    const int roundShift = 1 << (coefShift - 1);

    final double sxScale = width / newWidth;
    final double syScale = height / newHeight;

    // Cache koefisien horizontal: tiap kolom tujuan punya 4 tap tetap.
    final List<int> xIdx = List<int>.filled(newWidth * 4, 0);
    final List<int> xCoef = List<int>.filled(newWidth * 4, 0);
    for (int x = 0; x < newWidth; x++) {
      final double fx = (x + 0.5) * sxScale - 0.5;
      int base = fx.floor();
      final double frac = fx - base;
      if (base < 0) base = 0;
      if (base > width - 2) base = math.max(0, width - 2);
      for (int k = 0; k < 4; k++) {
        final int srcIdx = base - 1 + k;
        xIdx[x * 4 + k] = _reflect101(srcIdx, width);
        final double w = _cubicWeight(frac - (k - 1));
        xCoef[x * 4 + k] = (w * coefScale).round();
      }
    }

    final List<int> row = List<int>.filled(newWidth, 0);
    for (int y = 0; y < newHeight; y++) {
      final double fy = (y + 0.5) * syScale - 0.5;
      int baseY = fy.floor();
      final double fracY = fy - baseY;
      if (baseY < 0) baseY = 0;
      if (baseY > height - 2) baseY = math.max(0, height - 2);

      final List<int> yIdx = List<int>.filled(4, 0);
      final List<int> yCoef = List<int>.filled(4, 0);
      for (int k = 0; k < 4; k++) {
        yIdx[k] = _reflect101(baseY - 1 + k, height);
        yCoef[k] = (_cubicWeight(fracY - (k - 1)) * coefScale).round();
      }

      for (int x = 0; x < newWidth; x++) {
        final int xi = x * 4;
        final int c0 = xCoef[xi];
        final int c1 = xCoef[xi + 1];
        final int c2 = xCoef[xi + 2];
        final int c3 = xCoef[xi + 3];
        final int i0 = xIdx[xi];
        final int i1 = xIdx[xi + 1];
        final int i2 = xIdx[xi + 2];
        final int i3 = xIdx[xi + 3];

        int sum = 0;
        for (int j = 0; j < 4; j++) {
          final int rowBase = yIdx[j] * width;
          final int hsum = data[rowBase + i0] * c0 +
              data[rowBase + i1] * c1 +
              data[rowBase + i2] * c2 +
              data[rowBase + i3] * c3;
          sum += hsum * yCoef[j];
        }
        final int value = (sum + roundShift) >> coefShift;
        row[x] = value < 0 ? 0 : (value > 255 ? 255 : value);
      }

      final int rowBase = y * newWidth;
      for (int x = 0; x < newWidth; x++) {
        out.data[rowBase + x] = row[x];
      }
    }
    return out;
  }

  /// Indeks batas `BORDER_REFLECT_101`, sama seperti `cv2.resize`.
  ///
  /// OpenCV mencerminkan tanpa mengulang piksel tepi: -1 -> 1, n -> n-2.
  /// Memakai `clamp` (replicate) membuat baris/kolom tepi hasil resize
  /// berbeda satu piksel dari Python, dan perbedaan sekecil itu cukup
  /// membalik prediksi tanda baca.
  static int _reflect101(int i, int n) {
    if (n <= 1) return 0;
    int idx = i;
    while (idx < 0 || idx >= n) {
      if (idx < 0) {
        idx = -idx;
      } else {
        idx = 2 * (n - 1) - idx;
      }
    }
    return idx;
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
