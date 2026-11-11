import 'ocr_raw_text.dart';

/// Antarmuka mesin OCR.
///
/// Sengaja dijaga sebagai antarmuka agar bagian lain aplikasi tidak pernah
/// bergantung langsung pada implementasi tertentu. Implementasi yang
/// dipakai adalah [ClassicalOcrEngine] di `ocr/classical/`, yang
/// menjalankan seluruh pipeline pengolahan citra klasik di perangkat.
abstract class OcrEngine {
  /// Kenali teks pada [imagePath] (berkas lokal yang dapat dibaca).
  ///
  /// Mengembalikan teks mentah beserta geometri per baris agar pemanggil
  /// dapat mempertimbangkan tata letak (struk sangat terstruktur, urutan
  /// baris penting).
  Future<OcrRawResult> recognize(String imagePath);

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
