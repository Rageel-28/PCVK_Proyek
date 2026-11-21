/// Citra grayscale dan biner sederhana untuk pipeline OCR klasik.
///
/// Konvensi di seluruh pipeline: pada citra biner, **255 = tinta** dan
/// **0 = kertas**. Data disimpan baris demi baris (row-major) di [Uint8List]
/// agar mudah diport dari numpy dan hemat memori.
library;

import 'dart:typed_data';

class GrayImage {
  GrayImage(this.width, this.height, this.data)
      : assert(data.length == width * height, 'panjang data tidak cocok');

  GrayImage.zeros(this.width, this.height)
      : data = Uint8List(width * height);

  final int width;
  final int height;
  final Uint8List data;

  int get length => data.length;

  int at(int x, int y) => data[y * width + x];

  void set(int x, int y, int value) => data[y * width + x] = value;

  GrayImage clone() => GrayImage(width, height, Uint8List.fromList(data));
}

class BinaryImage {
  BinaryImage(this.width, this.height, this.data)
      : assert(data.length == width * height, 'panjang data tidak cocok');

  BinaryImage.zeros(this.width, this.height)
      : data = Uint8List(width * height);

  final int width;
  final int height;
  final Uint8List data;

  int get length => data.length;

  bool isInk(int x, int y) => data[y * width + x] != 0;

  void setInk(int x, int y, bool ink) =>
      data[y * width + x] = ink ? 255 : 0;

  bool get hasInk {
    for (int i = 0; i < data.length; i++) {
      if (data[i] != 0) return true;
    }
    return false;
  }

  /// Salin sub-citra ke citra biner baru.
  BinaryImage crop(int left, int top, int right, int bottom) {
    final int w = right - left;
    final int h = bottom - top;
    final BinaryImage out = BinaryImage.zeros(w, h);
    for (int y = 0; y < h; y++) {
      final int src = (top + y) * width + left;
      out.data.setRange(y * w, y * w + w, data, src);
    }
    return out;
  }
}
