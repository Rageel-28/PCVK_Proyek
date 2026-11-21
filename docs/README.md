# Dokumentasi SmartReceipt Analyzer

Dokumentasi proyek PCVK — pengenalan teks struk belanja (OCR) yang dibangun
dari nol. Mulai dari [Ringkasan Proyek](01-ringkasan-proyek.md) bila baru
mengenal proyek ini.

## Peta Bacaan

**Ingin memahami sistem secara menyeluruh**
01 → 02 → 03 → 04 → 05

**Ingin menjalankan / mengubah kode**
07 → 13 (peta kode) → 02

**Ingin menilai kualitas ilmiah**
03 → 06 → 14 → 11 → 12

**Ingin melihat antarmuka aplikasi**
08 → 09 → 10

## Daftar Dokumen

| # | Dokumen | Ringkas |
|---|---|---|
| 01 | [Ringkasan Proyek](01-ringkasan-proyek.md) | Latar belakang, tujuan, ruang lingkup, metode, hasil |
| 02 | [Arsitektur Aplikasi](02-arsitektur-aplikasi.md) | Struktur `lib/`, alur scan, state, SQLite, hubungan ke `ML/` |
| 03 | [Pipeline OCR Klasik](03-pipeline-ocr.md) | Praproses → segmentasi → fitur 1844-d → LinearSVC → decoding |
| 04 | [Parser Struk & Koreksi Teks](04-parser-struk.md) | Rekonstruksi baris, item/total, rekonsiliasi aritmetika |
| 05 | [Klasifikasi Kategori](05-klasifikasi-kategori.md) | TF-IDF + Naive Bayes, margin review, taksonomi |
| 06 | [Evaluasi & Pengujian](06-evaluasi-pengujian.md) | CER/WER, paritas Python↔Dart, daftar uji |
| 07 | [Build & Rilis](07-build-rilis.md) | Prasyarat, dependensi, CI iOS, melatih ulang |
| 08 | [Layar & Lapisan UI](08-layar-ui.md) | Layar, navigasi, tema, widget, model UI/DB |
| 09 | [Analitik & Prediksi](09-analitik-prediksi.md) | Agregasi pengeluaran & regresi linear |
| 10 | [Praproses Citra Aplikasi](10-praproses-aplikasi.md) | Kualitas citra, deteksi kertas, variasi retry |
| 11 | [Metodologi Pelatihan](11-metodologi-pelatihan.md) | Dataset, augmentasi, training, ekspor model |
| 12 | [Keputusan Desain](12-keputusan-desain.md) | 14 keputusan arsitektur dan alasannya |
| 13 | [Glosarium & Peta Kode](13-glosarium-peta-kode.md) | Istilah teknis & panduan "saya ingin mengubah…" |
| 14 | [Contoh Per Langkah: Struk Nyata & Fixture](14-contoh-struk.md) | Bedah pipeline tahap demi tahap dengan keluaran nyata |

## Konvensi Dokumen

- Nama kelas/fungsi/berkas ditulis dalam `code`.
- Angka evaluasi bersumber dari `ML/README.md` dan uji Flutter; bila ada
  konflik antar sumber, ditandai eksplisit di dokumen terkait.
- Rujukan karya kuliah (proposal, SKPL) ada di folder `Document/`.
