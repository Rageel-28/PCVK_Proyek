# 02 — Arsitektur Aplikasi

Dokumen ini menjelaskan struktur aplikasi Flutter `smart_receipt/`, alur
memindai struk, pengelolaan state, dan penyimpanan data.

## Teknologi & Dependensi

| Paket | Peran |
|---|---|
| `sqflite` + `sqflite_common_ffi` | Basis data lokal SQLite (uji di desktop memakai FFI) |
| `image_picker` | Ambil foto dari kamera/galeri |
| `image` | Dekode & manipulasi citra (rotasi, crop, binarisasi, deteksi kualitas) |
| `provider` | State management (`ChangeNotifier`) |
| `fl_chart` | Grafik pengeluaran |
| `intl` | Format tanggal & mata uang |
| `uuid` | ID transaksi |
| `shared_preferences` | Preferensi (mis. tema) |

> **Tidak ada paket OCR atau deep learning.** Seluruh pengenalan teks
> ditulis sendiri di `lib/ocr/classical/` (lihat dokumen 03).

## Struktur `lib/`

```
lib/
├── main.dart                 Entry point: tema, MultiProvider, route utama
├── ocr/                      Lapisan OCR
│   ├── ocr_engine.dart       Antarmuka OcrEngine (kontrak mesin OCR)
│   ├── ocr_raw_text.dart     OcrRawResult / OcrLine (teks + geometri)
│   ├── ocr_service.dart      Orkestrasi: pra-proses → OCR → parse
│   ├── image_quality.dart    Deteksi blur/cahaya/rotasi + variasi gambar
│   ├── ocr_text_corrector.dart  Koreksi karakter pasca-OCR
│   ├── receipt_parser.dart   Teks mentah → ParsedReceipt
│   ├── receipt_classifier.dart  Fallback klasifikasi berbasis kata kunci
│   ├── tfidf_classifier.dart  Eksekutor model TF-IDF + Naive Bayes
│   ├── ocr_abbreviations.dart   Pemetaan singkatan
│   └── classical/            Pipeline OCR klasik (lihat dokumen 03)
├── screens/                  13 berkas layar (lihat dokumen 08)
├── providers/                ReceiptProvider — state global transaksi
├── models/                   Model data + taksonomi kategori (categories.dart)
├── services/                 Analitik, prediksi, normalisasi, evaluasi
├── helpers/                  database_helper.dart (schema SQLite)
├── data/                     receipt_adapter.dart (parser ↔ model)
├── navigation/               app_route.dart, app_nav.dart (transisi)
├── theme/                    Palet warna + controller terang/gelap
└── widgets/                  Komponen UI berulang
```

## Layar & Alur Pengguna

```
SplashScreen
   └─> HomeDashboard  (ringkasan bulanan, grafik, CTA "Pindai Struk")
         └─> ScanReceiptScreen        (kamera / galeri)
               └─> OcrProcessingScreen (memuat engine, menjalankan pipeline)
                     └─> CategoryClassificationScreen   (item + kategori)
                           └─> ReviewTransactionScreen  (konfirmasi/koreksi)
                                 └─> TransactionSavedScreen
```

Layar pendukung: `TransactionHistoryScreen`, `TransactionDetailScreen`,
`SpendingPredictionScreen` (regresi linear atas riwayat bulanan),
`AnalysisScreen`, `SettingsScreen` (tema), `EmptyStateScreen`.

## Antarmuka Mesin OCR

Kontrak mesin OCR didefinisikan di `lib/ocr/ocr_engine.dart`:

```dart
abstract class OcrEngine {
  Future<OcrRawResult> recognize(String imagePath);
  bool get needsRawImage;          // default: false
  Future<void> dispose();
}
```

- `OcrRawResult` berisi daftar `OcrLine` (teks + bounding box) dan teks
  penuh. Geometri per baris **dibutuhkan** oleh parser untuk
  merekonstruksi tata letak tabel struk.
- `needsRawImage` memisahkan mesin yang melakukan praprosesnya sendiri
  (`ClassicalOcrEngine` mengembalikan `true`) dari mesin yang mengharapkan
  citra sudah disiapkan aplikasi. Ini penting: praproses aplikasi
  (rotasi, unsharp-mask, re-encode JPEG) justru merusak goresan tipis
  cetakan thermal yang menjadi bahan utama pipeline klasik.
- `OcrFailure` dan `OcrEngineException` membedakan kegagalan
  user-facing (gambar tidak terbaca) dari kegagalan mesin.

Mesin dipasang via constructor: `OcrService(engine: ...)`. Saat ini
`ClassicalOcrEngine.load()` dipanggil di `screens/ocr_processing.dart`
saat layar pemrosesan dibuka; instansnya di-cache di layar tersebut dan
dipakai ulang untuk percobaan berikutnya.

## Alur Satu Pemindaian (`OcrService.process`)

Didefinisikan di `lib/ocr/ocr_service.dart`, dengan callback
`onStage` (`preprocessing → recognizing → parsing → done`) yang
ditampilkan sebagai checklist progres di UI:

1. **Pra-proses & analisis kualitas** (`ImagePreprocessor`):
   - Mesin dengan `needsRawImage = true`: hanya **analisis**
     (kecerahan rata-rata, varians Laplacian untuk blur, rotasi)
     tanpa mengubah berkas sumber.
   - Mesin lain: gambar disiapkan (skala maksimum 3000 px, rotasi,
     penajaman) ke berkas sementara.
2. **Pengenalan teks** — `engine.recognize(...)` menjalankan pipeline OCR
   (di isolate), lalu `correctOcrText(...)` memperbaiki karakter yang
   tertukar (lihat dokumen 04).
3. **Parse** — `ReceiptParser.parse(...)` menghasilkan `ParsedReceipt`.
4. **Strategi cadangan** (hanya mesin non-raw, untuk hasil lemah):
   koreksi perspektif → crop ke daerah kertas → binarisasi lokal;
   hasil dengan skor parser terbaik yang dipertahankan. Skor
   (`ReceiptParser.scoreOf`) memberi bobot besar pada total tercetak
   yang cocok dengan jumlah item.
5. **Gagal dengan pesan berguna** — jika tidak ada teks: deteksi apakah
   ada kertas struk sama sekali (`containsReceiptPaper`), lalu
   `OcrFailure` dengan petunjuk berbeda untuk "struk tidak ada di foto"
   vs "foto gelap/buram".

   <!-- -->

Hasil akhir: `OcrScanResult` = berkas sumber + laporan kualitas +
teks mentah + `ParsedReceipt`.

## State Management

`ReceiptProvider` (`lib/providers/receipt_provider.dart`,
`ChangeNotifier`) menyimpan daftar `Receipt` yang tersimpan, menghitung
ringkasan bulanan, dan mengekspos operasi `loadData`, `updateReceipt`,
`deleteReceipt`. Di-instantiasi sekali lewat `ChangeNotifierProvider`
di `main.dart`.

## Penyimpanan

SQLite via `DatabaseHelper` (`lib/helpers/database_helper.dart`),
berkas `smart_receipt.db` versi 3:

| Tabel | Isi |
|---|---|
| `categories` | 9 kategori kanonik — di-seed dari `kAppCategories` |
| `receipts` | Satu struk: toko, tanggal, subtotal/pajak/total, pembayaran, nomor struk, `image_path`, `ocr_text`, status, catatan |
| `transactions` | Item per struk: nama, qty, harga, kategori (FK cascade; FK kategori) |
| `category_corrections` | Koreksi kategori oleh pengguna — data latih untuk retraining |

Migrasi: v1→v2 menambah tabel koreksi; v2→v3 memetakan ulang id kategori
legacy ke taksonomi kanonik (`_migrateToCanonicalCategories`).
`PRAGMA foreign_keys = ON` diaktifkan.

Foto struk **dipindah** dari direktori sementara ke dokumen aplikasi
(`receipt_image_store.dart`) karena iOS dapat menghapus berkas sementara
kapan saja.

Taksonomi kategori dikonsolidasi di `lib/models/categories.dart`
(`kAppCategories`, 9 entri) sebagai satu sumber kebenaran: seed
database, UI, dan pemetaan label model (15 kelas → 9 kategori) semua
merujuk daftar ini, sehingga id tidak bisa lagi bergeser antar-sistem.

## Hubungan dengan Folder `ML/`

`ML/ocr/` adalah **rujukan Python** (pelatihan + ekspor model) untuk
pipeline yang sama dengan `lib/ocr/classical/`. Aplikasi **tidak**
menjalankan Python; ia hanya membaca JSON hasil ekspor
(`assets/ocr/*.json`). Alur ketergantungan:

```
ML/ocr/render_dataset.py ──> chars.npz ──> train_char.py ──> char_clf.joblib
                                                          └─> export_char.py ──┐
ML/ocr/export_lm.py  ─────────────────────────────────────────────────────────┤
                                                                              ▼
                                            smart_receipt/assets/ocr/*.json
                                                                              │
                                                                              ▼
                                     lib/ocr/classical/ (port Dart) ──> APK
```

Paritas dijaga dua arah: `test/ocr_golden_test.dart` membandingkan fitur
dan probabilitas dengan vektor rujukan Python, dan
`test/ocr_pipeline_test.dart` mengevaluasi output end-to-end pada
fixture terkunci.

## Layanan (services)

| Berkas | Peran |
|---|---|
| `correction_training_service.dart` | Merekam koreksi kategori, membangun & mengekspor dataset latih (JSON) |
| `ml_classifier_service.dart` | Siklus hidup model klasifikasi: muat, prediksi, lampirkan praprosesor teks |
| `model_evaluator.dart` | Metrik per kelas: akurasi, presisi/recall/F1 (macro & weighted) |
| `receipt_image_store.dart` | Menyimpan foto struk permanen di dokumen aplikasi |
| `spending_analytics.dart` | Agregasi pengeluaran murni Dart (per bulan, per kategori) |
| `spending_forecaster.dart` | Regresi linear atas total bulanan + MAE/RMSE/R² |
| `text_normalizer.dart` | Praproses teks Indonesia (clean → tokenize → normalize) |
