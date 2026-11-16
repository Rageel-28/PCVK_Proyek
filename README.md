# PCVK_Proyek — SmartReceipt Analyzer

Proyek mata kuliah **Pengolahan Citra** (PCVK): pengenalan teks struk
belanja (OCR) yang dibangun **dari nol**, tanpa library OCR siap pakai
dan tanpa deep learning.

Proyek ini merupakan turunan dari **PBL SmartReceipt Analyzer** — aplikasi
Flutter pencatat pengeluaran yang membaca struk belanja dan
mengklasifikasikan item ke kategori secara otomatis.

---

## Fokus Proyek Ini

| Aspek | Keterangan |
|---|---|
| **Mata kuliah** | Pengolahan Citra |
| **Komponen yang dikerjakan** | Tahap OCR: dari foto struk menjadi teks |
| **Pendekatan** | Pengolahan citra klasik + machine learning klasik |
| **Dilarang** | Library OCR siap pakai, deep learning |

Modul utama: **[`ML/ocr_classical/`](ML/ocr_classical/)**

---

## Ringkasan Hasil

Angka di bawah diukur pada **struk asli** (`ML/Dataset Citra/`), bukan pada
citra render sendiri. Rincian dan protokolnya:
**[HASIL_DATA_ASLI.md](ML/ocr_classical/HASIL_DATA_ASLI.md)**

### Per baris item (28 baris bertranskripsi manual)

| Kelompok | CER | WER | Exact match |
|---|---:|---:|---:|
| Semua | **7,13%** | 18,62% | 71,43% |
| Render bersih (20 baris) | **0,14%** | 1,67% | 95,00% |
| **Foto asli** (8 baris) | **24,59%** | 61,01% | 12,50% |

### Tingkat dokumen (9 struk)

| Kelompok | CER | WER |
|---|---:|---:|
| Render bersih (5 struk) | **0,82%** | 5,60% |
| **Foto asli** (4 struk) | **104,12%** | 138,87% |

CER foto > 100% berarti segmentasi baris menghasilkan lebih banyak karakter
daripada teks acuan: lipatan dan QR pada kertas kusut terpecah menjadi baris
palsu. Ini kelemahan nyata dan dicatat terbuka.

| Metrik lain | Nilai |
|---|---:|
| Akurasi klasifikasi karakter (validasi) | **99,52%** |
| Ukuran model | 141 KB (JSON) |
| Paritas HOG + SVM Python ↔ Dart | **100% identik** |
| Test suite Flutter | **182 lolos** |


---

## Pipeline OCR

```
Foto struk
   ├─ [1]  Deteksi area kertas        Otsu + connected components
   ├─ [2]  Grayscale + denoise        median blur 3x3
   ├─ [3]  Binarisasi                 Otsu / Sauvola
   ├─ [4]  Upscale adaptif            tinggi baris → ~26 px
   ├─ [5]  Deskew                     profil proyeksi horizontal
   ├─ [6]  Segmentasi baris           proyeksi + Otsu 1D
   ├─ [7]  Segmentasi karakter        connected components + gabung pecahan
   ├─ [8]  Normalisasi                32x32
   ├─ [9]  Ekstraksi fitur            HOG manual (324 dimensi)
   ├─ [10] Klasifikasi karakter       Linear SVM + batasan kandidat per kata
   ├─ [11] Rekonstruksi baris         spasi dari lebar karakter
   └─ [12] Koreksi leksikon           opsional (default mati, lihat catatan)
```

> **Catatan tahap 12.** Koreksi kamus diukur **menurunkan** akurasi pada
> struk asli (CER 7,46% → 10,96%): nama produk yang benar seperti `GAM`
> dan `BACTERIAL` ikut "dikoreksi" menjadi `GA` dan `BATERAI`. Karena itu
> defaultnya mati; aktifkan hanya bila kamusnya memang mencakup kosakata
> struk yang dibaca.


---

## Sudah berjalan di aplikasi

Pipeline ini **menggantikan Google ML Kit** dan berjalan di Flutter
tanpa library OCR apa pun:

| Komponen | Status |
|---|---|
| Dependensi ML Kit | **dihapus** dari `pubspec.yaml` |
| `MlKitOcrEngine` | **dihapus** |
| `ClassicalOcrEngine` | **12 modul Dart** di `lib/ocr/classical/` |
| Aset model | `ocr_char_svm.json` (141 KB) + `ocr_lexicon.json` (34 KB) |
| Klasifikasi TF-IDF + Naive Bayes | **tidak diubah** — hanya masukannya kini teks dari OCR klasik |
| Koreksi leksikon | **nonaktif default** (terukur menurunkan akurasi pada struk asli) |


### Verifikasi paritas Dart vs Python

| Uji | Hasil |
|---|---|
| HOG + SVM (16 patch, toleransi 1e-4) | **100% identik** |
| Teks baris, struk render asli | **100,00%** kesepakatan karakter |
| Teks baris, foto asli | **85,25%** kesepakatan karakter |
| Jalur aplikasi pada 6 struk asli | **6/6** menghasilkan teks + total |
| Seluruh test suite | **181 test lolos** |

---

## Struktur Repositori

```
PCVK_Proyek/
├── ML/
│   ├── ocr_classical/          ← PROYEK PCVK (fokus utama)
│   │   ├── config.py
│   │   ├── render_synthetic.py     ← data sintetis PELENGKAP saja
│   │   ├── build_real_chars.py     ← panen karakter dari transkripsi manual
│   │   ├── real_data.py            ← akses struk asli + bbox COCO
│   │   ├── evaluate_real.py        ← evaluasi pada struk ASLI
│   │   ├── transcripts_real.json   ← transkripsi manual 9 struk
│   │   ├── resize_cubic.py         ← resize/rotasi identik dengan Dart
│   │   ├── preprocess.py
│   │   ├── paper.py
│   │   ├── deskew.py
│   │   ├── segment.py
│   │   ├── features.py
│   │   ├── train_char.py
│   │   ├── lexicon.py
│   │   ├── ocr_pipeline.py
│   │   ├── export_lexicon.py
│   │   ├── export_real_parity.py
│   │   ├── evaluate.py             ← ablasi pada data sintetis (lampiran)
│   │   ├── visualize.py
│   │   ├── HASIL_DATA_ASLI.md      ← LAPORAN HASIL (acuan)
│   │   ├── DOKUMENTASI.md
│   │   ├── models/
│   │   └── output/
│   │
│   ├── Dataset Citra/          ← 622 foto struk asli Indomaret/Alfamart
│   ├── Receipt Indomaret.v1i.coco.zip
│   ├── Alfamart Indomaret Struk.v1i.coco.zip
│   ├── Dataset_SmartReceipt_Sintetis.csv
│   ├── train_real_model.py     ← pelatihan model kategori (dari PBL)
│   └── uts/                    ← laporan UTS PBL
│
├── smart_receipt/              ← aplikasi Flutter (dari PBL)
│   └── lib/
│       ├── ocr/                ← OCR + klasifikasi
│       ├── services/
│       └── screens/
│
└── Document/                   ← dokumen proposal & SKPL
```

---

## Memulai

### 1. Instalasi dependensi

```powershell
cd ML
python -m pip install -r requirements_ocr.txt
```

### 2. Jalankan pipeline

```powershell
cd ML/ocr_classical

# 1. Data sintetis PELENGKAP (format struk Indonesia: koma ribuan, titik dua, DPN/PPN)
python render_synthetic.py --max-lines 2500 --samples-per-line 2

# 2. Panen karakter ASLI berlabel manual dari transkripsi
python build_real_chars.py

# 3. Latih classifier
python train_char.py

# 4. Evaluasi pada struk ASLI (acuan pelaporan)
python evaluate_real.py                # per baris item
python evaluate_real.py --mode doc     # per dokumen penuh

# 5. Ekspor aset untuk aplikasi
python export_lexicon.py
python export_real_parity.py
Copy-Item models/char_svm.json ..\..\smart_receipt\assets\ocr_char_svm.json

# 6. OCR satu gambar
python ocr_pipeline.py path\ke\struk.jpg
```

Ablasi pada data sintetis (`python evaluate.py`) tetap tersedia sebagai
lampiran, tetapi **bukan acuan** hasil.

### 3. Aplikasi Flutter (opsional)

```powershell
cd smart_receipt
flutter pub get
flutter run
```

---

## Pemenuhan Syarat Tugas

| Syarat | Status | Bukti |
|---|---|---|
| Tanpa library OCR siap pakai | ✅ Terpenuhi | Tidak ada pytesseract/easyocr/paddleocr/ML Kit |
| Tanpa deep learning | ✅ Terpenuhi | Hanya LinearSVC (SVM klasik) |
| Tanpa library sama sekali | ⚠️ Sebagian | Masih memakai OpenCV & NumPy sebagai alat dasar |

Penjelasan lengkap beserta daftar algoritma yang ditulis manual ada di
**[DOKUMENTASI.md § 2](ML/ocr_classical/DOKUMENTASI.md#2-pemenuhan-syarat-tugas)**.

---

## Algoritma yang Ditulis dari Nol

| Algoritma | Berkas |
|---|---|
| HOG 324 dimensi (tanpa `skimage.hog`) | `features.py` |
| Sauvola thresholding | `preprocess.py` |
| Otsu 1D untuk profil baris | `segment.py` |
| Deskew profil proyeksi | `deskew.py` |
| Segmentasi karakter (CC + gabung pecahan + split lembah) | `segment.py` |
| Bicubic fixed-point 11 bit (identik dengan Dart) | `resize_cubic.py` |
| Rotasi nearest + pembulatan setengah ke genap | `resize_cubic.py` |
| Levenshtein | `lexicon.py` |
| Spasi kata adaptif berbasis lebar karakter | `ocr_pipeline.py` |
| Batasan kandidat karakter per konteks kata | `config.py`, `ocr_pipeline.py` |

---

## Batasan yang Diketahui

| Batasan | Dampak terukur | Penyebab |
|---|---|---|
| Segmentasi baris pada foto kusut | CER dokumen foto **104,1%** | lipatan, bayangan, QR pecah jadi baris palsu |
| Cetakan pudar | CER baris foto **24,6%** | stroke hilang, glyph terbelah, `0`/`O` nyaris identik |
| Ukuran sampel uji kecil | 28 baris, 9 dokumen | waktu transkripsi manual |
| Render bukan foto | 500 dari 622 gambar | dataset aslinya memang template |
| Koreksi kamus tidak dipakai | CER naik 7,46% → 10,96% | nama produk asli jarang ada di kamus |

Batasan ini didokumentasikan terbuka sebagai bahan analisis laporan,
bukan disembunyikan. Rincian pengukuran:
**[HASIL_DATA_ASLI.md](ML/ocr_classical/HASIL_DATA_ASLI.md)**

---

## Lisensi Data

- Dataset struk Indomaret/Alfamart: CC BY 4.0 (Roboflow, `universe.roboflow.com/jupar`)
- Dataset sintetis: dibuat sendiri untuk proyek PBL
- Dataset COCO `OCR Struk Belanja`: CC BY 4.0 (via Roboflow)

