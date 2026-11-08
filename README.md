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

| Metrik | Nilai |
|---|---:|
| Akurasi klasifikasi karakter | **99,73%** |
| Exact match baris (Sauvola + deskew + leksikon) | **72,00%** |
| CER (Character Error Rate) | 29,75% |
| WER (Word Error Rate) | 46,08% |
| Ukuran model | 117 KB (JSON) |

Diuji pada 150 baris sintetis ber-degradasi. Detail lengkap:
**[DOKUMENTASI.md](ML/ocr_classical/DOKUMENTASI.md)**

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
   ├─ [7]  Segmentasi karakter        connected components + split lembah
   ├─ [8]  Normalisasi                32x32
   ├─ [9]  Ekstraksi fitur            HOG manual (324 dimensi)
   ├─ [10] Klasifikasi karakter       Linear SVM
   ├─ [11] Rekonstruksi baris         spasi dari median celah
   └─ [12] Koreksi leksikon           Levenshtein (3.641 kata)
```

---

## Struktur Repositori

```
PCVK_Proyek/
├── ML/
│   ├── ocr_classical/          ← PROYEK PCVK (fokus utama)
│   │   ├── config.py
│   │   ├── render_synthetic.py
│   │   ├── preprocess.py
│   │   ├── paper.py
│   │   ├── deskew.py
│   │   ├── segment.py
│   │   ├── features.py
│   │   ├── train_char.py
│   │   ├── lexicon.py
│   │   ├── ocr_pipeline.py
│   │   ├── evaluate.py
│   │   ├── visualize.py
│   │   ├── DOKUMENTASI.md      ← dokumentasi teknis lengkap
│   │   ├── models/
│   │   └── output/
│   │
│   ├── Dataset_SmartReceipt_Sintetis.csv
│   ├── Dataset_Katalog_Indonesia_Sintetis_v1.csv
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

# Bangun data latih karakter
python render_synthetic.py --max-lines 2500 --samples-per-line 2

# Latih classifier
python train_char.py

# Evaluasi + ablasi
python evaluate.py

# Visualisasi untuk laporan
python visualize.py

# OCR satu gambar
python ocr_pipeline.py path\ke\struk.jpg
```

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
| Segmentasi karakter (CC + merge + split) | `segment.py` |
| Levenshtein | `lexicon.py` |
| Spasi kata adaptif | `ocr_pipeline.py` |

---

## Batasan yang Diketahui

- Font tipis masih terpecah saat binarisasi → segmentasi berlebih
- Huruf bersambung akibat blur thermal → dua huruf menjadi satu
- Foto struk asli masih di bawah ML Kit (domain berbeda dari data sintetis)
- Dataset COCO berisi struk Malaysia/Italia, bukan Indonesia

Batasan ini didokumentasikan terbuka sebagai bahan analisis laporan,
bukan disembunyikan.

---

## Lisensi Data

- Dataset sintetis: dibuat sendiri untuk proyek PBL
- Dataset COCO `OCR Struk Belanja`: CC BY 4.0 (via Roboflow)
