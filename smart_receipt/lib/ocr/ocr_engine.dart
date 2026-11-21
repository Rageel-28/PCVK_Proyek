import 'ocr_raw_text.dart';

/// Antarmuka mesin OCR.
///
/// Sengaja dijaga sebagai antarmuka agar bagian lain aplikasi tidak pernah
/// bergantung langsung pada implementasi tertentu. Mesin OCR dipasang
/// lewat parameter constructor `OcrService`.
abstract class OcrEngine {
  /// Kenali teks pada [imagePath] (berkas lokal yang dapat dibaca).
  ///
  /// Mengembalikan teks mentah beserta geometri per baris agar pemanggil
  /// dapat mempertimbangkan tata letak (struk sangat terstruktur, urutan
  /// baris penting).
  Future<OcrRawResult> recognize(String imagePath);

  /// Apakah mesin ini ingin menerima citra **asli**, tanpa praproses aplikasi.
  ///
  /// Pipeline klasik sudah melakukan deskew, binarisasi adaptif, dan upscale
  /// sendiri. Memberinya gambar yang sudah diputar, di-unsharp, lalu di-JPEG
  /// ulang justru merusak goresan tipis cetakan thermal — hasilnya penuh
  /// derau seperti `220M-....` dan `ROT1`. Mesin yang mengembalikan `true`
  /// akan diberi berkas sumber apa adanya.
  bool get needsRawImage => false;

  /// Lepaskan sumber daya yang dipegang mesin.
  Future<void> dispose();
}

/// Dilempar ketika mesin OCR sendiri gagal (bukan "teks tidak ditemukan").
class OcrEngineException implements Exception {
  OcrEngineException(this.message);

  final String message;

  @override
  String toString() => 'OcrEngineException: $message';
}
