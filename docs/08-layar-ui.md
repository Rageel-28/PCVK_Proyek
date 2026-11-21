# 08 — Layar & Lapisan UI

Aplikasi memiliki **13 berkas layar** di `lib/screens/`: 12 layar
bernomor ("Screen 1"–"Screen 12") ditambah **HomeDashboard** yang menjadi
wadah tab Beranda.

## Daftar Layar

| # | Berkas | Kelas | Peran |
|---|---|---|---|
| 1 | `splash_screen.dart` | `SplashScreen` | Logo & layar peluncuran |
| 2 | `scan_receipt.dart` | `ScanReceiptScreen` | Ambil foto struk (kamera/galeri) |
| 3 | `ocr_processing.dart` | `OcrProcessingScreen` | Jalankan pipeline OCR + progres |
| 4 | `review_transaction.dart` | `ReviewTransactionScreen` | Konfirmasi & edit item sebelum simpan |
| 5 | `category_classification.dart` | `CategoryClassificationScreen` | Klasifikasi kategori per item |
| 6 | `transaction_saved.dart` | `TransactionSavedScreen` | Konfirmasi tersimpan |
| 7 | `transaction_history.dart` | `TransactionHistoryScreen` | Daftar struk tersimpan |
| 8 | `transaction_detail.dart` | `TransactionDetailScreen` | Detail satu struk + foto |
| 9 | `analysis.dart` | `AnalysisScreen` | Analisis pengeluaran per periode |
| 10 | `spending_prediction.dart` | `SpendingPredictionScreen` | Prediksi pengeluaran bulanan |
| 11 | `empty_state.dart` | `EmptyStateView` / `EmptyStateScreen` | Tampilan kosong yang dapat dipakai ulang |
| 12 | `settings.dart` | `SettingsScreen` | Pengaturan: tema, data & ML |
| — | `home_dashboard.dart` | `HomeDashboard` | Beranda (tab 0, tanpa nomor di kode) |

> Urutan alur scan adalah **2 → 3 → 5 → 4 → 6**: klasifikasi dijalankan
> **sebelum** layar review agar pengguna sudah melihat usulan kategori
> saat mengoreksi. (Nomor layar tidak mengikuti urutan alur; nomor adalah
> label "Screen N" di komentar kode.)

## Alur Navigasi

```
SplashScreen
  └─> HomeDashboard (tab Beranda)
        ├─[FAB Scan]─> ScanReceiptScreen
        │                 └─> OcrProcessingScreen
        │                       └─> CategoryClassificationScreen
        │                             └─> ReviewTransactionScreen
        │                                   └─> TransactionSavedScreen
        │                                         └─> kembali ke Beranda
        ├─[tab Riwayat]─> TransactionHistoryScreen ─> TransactionDetailScreen
        ├─[tab Analisis]─> AnalysisScreen ─> SpendingPredictionScreen
        └─[tab Pengaturan]─> SettingsScreen
```

Navigasi tab diatur oleh `lib/navigation/app_nav.dart`
(`navigateTab`, `openScan`) dan bottom nav (`AppBottomNav`) dengan
tombol Scan di tengah (index 2). Transisi halaman memakai
`InstantPushPageRoute` (`lib/navigation/app_route.dart`) — halaman baru
**langsung muncul** saat push, tetapi tetap beranimasi saat pop.

### Detail Tiap Layar Penting

**ScanReceiptScreen** — mengambil foto via `image_picker` (kamera dibatasi
~2400 px). Sebelum meneruskan ke OCR, gambar melewati gerbang
`ImagePreprocessor.containsReceiptPaper()`: jika tidak ada kertas struk di
foto, pengguna diberi dialog konfirmasi. Lalu `pushReplacement` ke layar
pemrosesan.

**OcrProcessingScreen** — memuat `ClassicalOcrEngine.load()` **sekali**
(memuat model cukup berat) lalu menyimpannya untuk percobaan berikutnya.
Menampilkan checklist tiga tahap (praproses → OCR → parsing), preview
foto, peringatan kualitas gambar (`ImageQualityReport.issues`), teks
mentah hasil OCR, dan tampilan gagal (`_FailureView`) dengan tombol
coba-ulang.

**CategoryClassificationScreen** — menjalankan klasifikasi TF-IDF +
Naive Bayes atas setiap nama item (lihat [05](05-klasifikasi-kategori.md)),
lalu meneruskan hasil ke layar review.

**ReviewTransactionScreen** — item dapat diedit (nama, qty, harga,
kategori). **Total diambil dari struk** (`_receiptTotal`) sebagai sumber
kebenaran; selisih terhadap jumlah item ditampilkan (`_totalAdjustment`).
Mengubah kategori item merekam `CategoryCorrection` untuk retraining.
Tombol simpan memanggil `ReceiptProvider.addReceipt()`.

**SettingsScreen** — profil, preferensi (mode gelap via `ThemeController`,
notifikasi, mata uang IDR, bahasa), dan bagian "Data & Machine Learning"
yang menampilkan jumlah koreksi serta tile "Latih Ulang Model".

## Tema

- `lib/theme/app_colors.dart` — `AppPalette` (±40 warna per mode terang/
  gelap), `AppColors` (fasad statis ke palet aktif), `ThemeScope`
  (InheritedWidget yang memicu repaint global saat tema berganti).
- `lib/theme/theme_controller.dart` — `ThemeController extends
  ChangeNotifier`; default `ThemeMode.system`, mendengarkan perubahan
  mode gelap sistem (`WidgetsBindingObserver` di `main.dart`), dan
  menyimpan pilihan ke `shared_preferences` (key `theme_mode`).
- `main.dart` membangun `ThemeData` Material 3 dari palet: `ColorScheme.
  fromSeed(seedColor: palette.navy)` dengan `scaffoldBackgroundColor`
  dari palet.

## Widget

| Berkas | Isi |
|---|---|
| `widgets/app_widgets.dart` | `AppStatusBar`, `AppTopBar`, `CircleIconButton`, `SectionHeader`, `PrimaryButton`, `SecondaryButton`, `CategoryPill`, `IconBox`, `AppBottomNav`, `FloatingNavBody` |
| `widgets/charts.dart` | `TrendLineChart` + `_LinePainter` — grafik garis digambar dengan `CustomPainter` (**tanpa paket grafik**) |
| `widgets/receipt_mock.dart` | `ReceiptMock` — ilustrasi kertas struk sebagai placeholder |
| `widgets/receipt_photo_preview.dart` | `ReceiptPhotoPreview` (thumbnail → viewer) dan `FullScreenReceiptViewer` (pinch-zoom & pan) |

## Dua Lapisan Model Data

Proyek memisahkan model **UI** dan model **database**:

| Lapisan | Berkas | Isi |
|---|---|---|
| UI | `models/models.dart` | `Category`, `kCategories`, `Transaction`, `TransactionItem`, helper `rupiah()`, `formatDate()`, `formatDateTime()`, `categoryByName()` |
| Database | `models/receipt_model.dart` | `Category`, `Transaction`, `Receipt`, `MonthlyTotal`, `CategoryCorrection` — masing-masing dengan `toMap()`/`fromMap()` |

`data/receipt_adapter.dart` (`toUiTransaction`) adalah **satu-satunya
jembatan** antara keduanya; juga menghasilkan id tampilan (`TRX007`).

`kCategories` (UI) **diturunkan** dari `kAppCategories` (kanonik), jadi
nama dan id selalu sama dengan seed SQLite; ikon Material dipetakan per
id dan warna diambil dari palet aktif agar tetap terbaca di mode terang
maupun gelap.

Format angka rupiah diimplementasikan manual (`rupiah(1420000)` →
`Rp1.420.000`), tanpa `intl`, agar tidak bergantung pada locale runtime.
