import 'gray_image.dart';

/// Zoning / Grid Features: kepadatan tinta per zona.
///
/// Kanvas karakter dibagi menjadi `grid × grid` sel; tiap sel menyumbang
/// satu nilai 0..1, yaitu rata-rata intensitas piksel tinta di sel itu.
///
/// Beda peran dengan HOG:
///   - HOG    → ARAH tepi (bentuk goresan); posisi tinta dibuang
///   - Zoning → SEBARAN tinta; arah diabaikan, tetapi zona yang padat dan
///              yang kosong ikut terekam
///
/// Contoh: `L` dan `7` dapat punya histogram gradien mirip, tetapi zonanya
/// berbeda jelas — `L` padat di kolom kiri dan baris bawah, `7` padat di
/// baris atas dan kolom kanan.
///
/// Fungsi ini HARUS identik dengan `zoning()` di `ML/ocr_classical/features.py`.
/// Bila berbeda, vektor yang masuk SVM saat latih dan saat uji tidak lagi
/// satu distribusi, dan akurasi aplikasi menyimpang dari laporan.
List<double> zoning(GrayImage image, {required int grid, int charSize = 32}) {
  if (image.width != charSize || image.height != charSize) {
    throw ArgumentError(
      'Dimensi harus ${charSize}x$charSize, dapat '
      '${image.width}x${image.height}',
    );
  }

  final List<double> out = List<double>.filled(grid * grid, 0.0);
  final int h = image.height;
  final int w = image.width;

  for (int r = 0; r < grid; r++) {
    final int y0 = r * h ~/ grid;
    final int y1 = (r + 1) * h ~/ grid;
    for (int c = 0; c < grid; c++) {
      final int x0 = c * w ~/ grid;
      final int x1 = (c + 1) * w ~/ grid;

      double sum = 0;
      int count = 0;
      for (int y = y0; y < y1; y++) {
        final int base = y * w;
        for (int x = x0; x < x1; x++) {
          // Inversi: tinta menjadi terang, sama seperti konvensi HOG.
          sum += 255.0 - image.data[base + x];
          count++;
        }
      }
      out[r * grid + c] = count == 0 ? 0.0 : (sum / count) / 255.0;
    }
  }

  return out;
}
