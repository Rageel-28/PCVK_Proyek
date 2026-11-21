# 01 — Ringkasan Proyek

## Latar Belakang

SmartReceipt Analyzer adalah aplikasi pencatat pengeluaran berbasis Flutter
yang membaca **struk belanja thermal** melalui kamera, mengekstrak item,
harga, dan total secara otomatis, lalu mengklasifikasikan setiap item ke
kategori pengeluaran. Proyek ini lahir dari PBL (Project-Based Learning)
SmartReceipt Analyzer dan dikembangkan lebih jauh untuk mata kuliah
**Pengolahan Citra (PCVK)**.

Tuntasan akademik proyek ini: **OCR struk belanja dibangun dari nol** —
tanpa library OCR siap pakai (Tesseract, ML Kit, dll.) dan tanpa deep
learning. Seluruh tahap pengolahan citra — grayscale, binarisasi,
segmentasi baris/kata/karakter, ekstraksi fitur, klasifikasi, hingga
decoding teks — diimplementasikan sebagai kode sendiri.

## Tujuan

1. Membangun pipeline OCR klasik end-to-end (citra → teks terstruktur)
   yang seluruh komponennya ditulis sendiri.
2. Menyajikan hasil OCR sebagai transaksi terstruktur: item, kuantitas,
   harga satuan, subtotal, total, dan metode pembayaran.
3. Mengklasifikasikan setiap item ke salah satu dari 9 kategori
   pengeluaran menggunakan model klasik (TF-IDF + Naive Bayes) yang
   berjalan sepenuhnya di perangkat.
4. Menyimpan riwayat belanja dan menyajikan analisis serta prediksi
   pengeluaran bulanan.

## Ruang Lingkup

| Dalam lingkup | Di luar lingkup |
|---|---|
| OCR struk thermal (digit 0–9, huruf besar A–Z, simbol `.&-/%+`) — 42 kelas | Huruf kecil, aksen, multi-bahasa |
| 42 kelas karakter → decoding kata dengan model bahasa | Deep learning (CNN, transformer, dll.) |
| Parser struk Indonesia (format nominal, label TOTAL/DPP/PPN, dll.) | Struk multi-halaman, foto banyak struk sekaligus |
| Klasifikasi 15 label → pemetaan ke 9 kategori aplikasi | Sinkronisasi cloud, autentikasi pengguna |
| Penyimpanan lokal SQLite + analitik & prediksi pengeluaran | Integrasi pembayaran/e-wallet |

## Metode Singkat

Pipeline dibagi menjadi dua bagian besar:

1. **Pengolahan citra & pengenalan karakter** — citra diubah menjadi
   citra biner (binarisasi Sauvola adaptif), dipecah menjadi baris →
   kata → glyph, setiap glyph dinormalisasi ke kanvas 32×32 dan
   diekstrak fiturnya (HOG + zoning + proyeksi + topologi lubang,
   total 1844 dimensi), lalu diklasifikasikan oleh classifier linear
   (LinearSVC + softmax). Deret probabilitas per glyph didekode menjadi
   teks dengan **beam search** yang dibantu model bahasa n-gram
   (order 5) dan leksikon — lihat dokumen
   [03 — Pipeline OCR Klasik](03-pipeline-ocr.md).

2. **Pemahaman struk** — teks mentah per baris (beserta kotak geometrinya)
   direkonstruksi menjadi baris visual, diklasifikasi menjadi
   item/total/pembayaran/diskon, dan dipadukan agar jumlah item cocok
   dengan total tercetak — lihat dokumen
   [04 — Parser Struk](04-parser-struk.md). Klasifikasi kategori item
   dibahas di [05 — Klasifikasi Kategori](05-klasifikasi-kategori.md).

> **Catatan penting:** logika OCR klasik dikembangkan sebagai rujukan
> Python di `ML/ocr/`. Implementasi produksi di Dart **harus
> menghasilkan output yang identik** dengan rujukan Python — paritas ini
> dijaga oleh parameter terpusat (`lib/ocr/classical/params.dart`) dan
> pengujian vektor emas (`test/ocr_golden_test.dart`).

## Hasil Utama

- Aplikasi Flutter lengkap: 13 berkas layar, tema terang/gelap,
  penyimpanan lokal, grafik, dan prediksi pengeluaran.
- Pipeline OCR klasik 100% kode sendiri (Dart murni, tanpa dependensi
  komputer vision) yang berjalan di isolate terpisah sehingga UI tetap
  responsif.
- Parser struk yang tahan terhadap kesalahan OCR: rekonstruksi baris dari
  geometri, pencocokan label berbasis jarak edit, dan rekonsiliasi
  aritmetika terhadap total tercetak.
- Akurasi pada fixture terkunci `receipts_v2` (10 struk, 138 baris):
  **CER 0,08%, WER 0,46%, exact-line 98,6%** (rujukan Python). Port Dart
  mencapai CER 0,00% dengan exact-line 135/138 (Python 136/138). Dua foto
  struk asli terbaca seluruh harganya dengan benar.
- 21 file pengujian otomatis + 2 skrip pemeriksaan manual, dengan
  fixtures foto struk asli.
