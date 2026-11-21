# PCVK_Proyek — SmartReceipt Analyzer

Proyek mata kuliah **Pengolahan Citra** (PCVK): pengenalan teks struk
belanja (OCR) yang dibangun **dari nol**, tanpa library OCR siap pakai
dan tanpa deep learning.

Proyek ini merupakan turunan dari **PBL SmartReceipt Analyzer** — aplikasi
Flutter pencatat pengeluaran yang membaca struk belanja dan
mengklasifikasikan item ke kategori secara otomatis.

---

## Status (per 19 Nov 2026)

> Tanggal status ini mengikuti tanggal working tree saat dokumentasi
> ditulis. Repo berada dalam keadaan mid-migration: sebagian perubahan
> (pipeline pure-ML baru, aset `assets/ocr/`, port Dart) belum di-commit
> dan hanya ada di working tree.

Pipeline OCR klasik **pure-ML** sudah lengkap dan berjalan di aplikasi:

- **Rujukan Python** (`ML/ocr/`) — melatih model dan mengekspor JSON.
- **Port Dart** (`smart_receipt/lib/ocr/classical/`, 16 file) — pipeline
  yang sama, diimplementasikan ulang dalam Dart murni dan dijalankan di
  isolate terpisah di perangkat.
- Keduanya dijaga identik oleh **uji paritas** (`flutter test`).

Yang **dihapus** dari repo: implementasi OCR klasik versi lama (fitur &
parameter berbeda), Google ML Kit, dataset COCO (771 gambar), CSV sintetis
PBL, dan aset SVM lama. Metode HOG + LinearSVC **tetap dipakai** — tetapi
dalam implementasi baru di `ML/ocr/` dan `lib/ocr/classical/`.

| Komponen | Status |
|---|---|
| `ML/ocr/` — pipeline Python pure-ML (rujukan + pelatihan) | ✅ Ada |
| `smart_receipt/lib/ocr/classical/` — port Dart (produksi) | ✅ Aktif |
| Aset model (`assets/ocr/`: char_clf, lm, lexicon, templates) | ✅ Termasuk build |
| Parser struk (`receipt_parser.dart`) | ✅ Aktif |
| Klasifikasi kategori (TF-IDF + Naive Bayes on-device) | ✅ Aktif |
| Folder `ML/` lama (dataset COCO, CSV sintetis, SVC lama) | ❌ Dihapus (riwayat di git) |

> Dokumentasi lengkap ada di folder [`docs/`](docs/README.md). Mulai dari
> [Ringkasan Proyek](docs/01-ringkasan-proyek.md).

---

## Struktur Repositori

```
PCVK_Proyek/
├── ML/                         ← pipeline OCR pure-ML (Python)
│   ├── ocr/                    ← config, praproses, fitur, latih, decode, evaluasi
│   ├── tests/                  ← pytest
│   └── README.md               ← detail metodologi & hasil evaluasi
├── smart_receipt/              ← aplikasi Flutter
│   ├── lib/
│   │   ├── ocr/                ← antarmuka OCR + parser + klasifikasi
│   │   │   └── classical/      ← port Dart pipeline (produksi)
│   │   ├── screens/            ← 13 berkas layar aplikasi
│   │   ├── services/           ← analitik, prediksi, evaluasi model
│   │   ├── providers/          ← state management (provider)
│   │   ├── models/             ← model data & taksonomi kategori
│   │   └── helpers/            ← SQLite (sqflite)
│   ├── assets/                 ← model JSON (OCR & klasifikasi)
│   └── test/                   ← pengujian + fixtures
├── docs/                       ← dokumentasi proyek
├── testdata/real/              ← foto struk asli untuk pengujian manual
└── Document/                   ← dokumen kuliah (proposal PBL, SKPL, audit)
```

---

## Memulai

```bash
cd smart_receipt
flutter pub get
flutter run        # Android / iOS / desktop
```

Menjalankan pengujian:

```bash
cd smart_receipt && flutter test        # port Dart + paritas
cd ML && python3 -m pytest tests/ -q    # rujukan Python
```

Melatih ulang pipeline Python dari nol: lihat
[07 — Build & Rilis](docs/07-build-rilis.md#5-melatih-ulang-pipeline-ml)
atau [`ML/README.md`](ML/README.md).

---

## Dokumentasi

| Dokumen | Isi |
|---|---|
| [01 — Ringkasan Proyek](docs/01-ringkasan-proyek.md) | Latar belakang, tujuan, ruang lingkup, metode |
| [02 — Arsitektur Aplikasi](docs/02-arsitektur-aplikasi.md) | Struktur Flutter, alur scan, state, penyimpanan |
| [03 — Pipeline OCR Klasik](docs/03-pipeline-ocr.md) | Praproses → segmentasi → fitur → classifier → decoding |
| [04 — Parser Struk](docs/04-parser-struk.md) | Rekonstruksi baris, ekstraksi item/total, koreksi teks |
| [05 — Klasifikasi Kategori](docs/05-klasifikasi-kategori.md) | TF-IDF + Naive Bayes on-device, taksonomi, fallback |
| [06 — Evaluasi & Pengujian](docs/06-evaluasi-pengujian.md) | Hasil CER/WER, strategi uji, paritas Python↔Dart |
| [07 — Build & Rilis](docs/07-build-rilis.md) | Dependensi, CI iOS, catatan platform |
| [08 — Layar & Lapisan UI](docs/08-layar-ui.md) | Layar, navigasi, tema, widget, model UI/DB |
| [09 — Analitik & Prediksi](docs/09-analitik-prediksi.md) | Agregasi pengeluaran & regresi linear |
| [10 — Praproses Citra Aplikasi](docs/10-praproses-aplikasi.md) | Analisis kualitas, deteksi kertas, variasi retry |
| [11 — Metodologi Pelatihan](docs/11-metodologi-pelatihan.md) | Dataset, augmentasi, training, ekspor model |
| [12 — Keputusan Desain](docs/12-keputusan-desain.md) | 14 keputusan arsitektur beserta alasannya |
| [13 — Glosarium & Peta Kode](docs/13-glosarium-peta-kode.md) | Istilah & panduan "saya ingin mengubah…" |
| [14 — Contoh Per Langkah](docs/14-contoh-struk.md) | Bedah pipeline tahap demi tahap pada struk asli & fixture |

---

## Lisensi Data

Dataset lama (sintetis PBL dan COCO `OCR Struk Belanja` CC BY 4.0) sudah
dihapus dari repo ini; sisa jejaknya hanya di riwayat git.
