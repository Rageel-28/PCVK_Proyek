import 'classical_config.dart';
import 'gray_image.dart';

/// Pembulatan seperti Python `round()` (setengah ke nilai genap).
///
/// Dart `round()` membulatkan setengah menjauhi nol, Python membulatkan
/// ke nilai genap. Perbedaan ini bisa menggeser ukuran hasil resize satu
/// piksel sehingga fitur HOG ikut berbeda.
int _pyRound(double value) {
  final double floorV = value.floorToDouble();
  final double diff = value - floorV;
  if (diff < 0.5) return floorV.toInt();
  if (diff > 0.5) return (floorV + 1).toInt();
  // Tepat 0.5: pilih nilai genap.
  return floorV.toInt().isEven ? floorV.toInt() : (floorV + 1).toInt();
}

/// Potong satu karakter lalu normalkan ke kanvas 32x32.
///
/// Rasio aspek dijaga: sisi terpanjang diskalakan ke `32 - 4` (menyisakan
/// 2 px margin tiap sisi), lalu ditempel di tengah kanvas putih.
///
/// Fungsi ini HARUS identik dengan `normalize_char` di
/// `ML/ocr_classical/render_synthetic.py`. Perbedaan sedikit saja membuat
/// akurasi jatuh karena distribusi fitur saat latih dan saat uji berbeda.
GrayImage normalizeChar(
  GrayImage gray,
  ({int left, int top, int right, int bottom}) box,
) {
  final int h = gray.height;
  final int w = gray.width;

  final int left = box.left.clamp(0, w - 1);
  final int top = box.top.clamp(0, h - 1);
  final int right = box.right.clamp(left, w - 1);
  final int bottom = box.bottom.clamp(top, h - 1);

  final int cw = right - left + 1;
  final int ch = bottom - top + 1;

  final GrayImage out = GrayImage.zeros(kCharSize, kCharSize, 255);
  if (cw <= 0 || ch <= 0) {
    return out;
  }

  final GrayImage crop = gray.crop(left, top, right, bottom);

  final double scale = (kCharSize - 4) / (ch > cw ? ch : cw);
  final int nh = _pyRound(ch * scale).clamp(1, kCharSize);
  final int nw = _pyRound(cw * scale).clamp(1, kCharSize);

  final GrayImage scaled = crop.resized(nw, nh);

  final int ox = (kCharSize - nw) ~/ 2;
  final int oy = (kCharSize - nh) ~/ 2;

  for (int y = 0; y < nh; y++) {
    final int dy = oy + y;
    if (dy < 0 || dy >= kCharSize) continue;
    for (int x = 0; x < nw; x++) {
      final int dx = ox + x;
      if (dx < 0 || dx >= kCharSize) continue;
      out.data[dy * kCharSize + dx] = scaled.data[y * nw + x];
    }
  }

  return out;
}

/// Pembulatan Python dipakai oleh perhitungan skala.
///
/// Dibuat publik agar dapat diuji terpisah bila paritas perlu diperiksa.
int pythonRound(double value) => _pyRound(value);
