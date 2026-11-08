# OCR Klasik — Tanpa Library OCR, Tanpa Deep Learning

Pipeline pengenalan teks struk belanja yang dibangun **dari nol** menggunakan
teknik pengolahan citra klasik dan machine learning klasik.

> **Untuk mata kuliah Pengolahan Citra.** Tidak memakai Google ML Kit,
> Tesseract, atau library OCR siap pakai apa pun. Tidak memakai neural
> network (CNN/YOLO/TFLite). Semua tahap adalah algoritma klasik yang
> dapat dijelaskan dan direproduksi.

**Dokumentasi teknis lengkap:** [`DOKUMENTASI.md`](DOKUMENTASI.md)
(penjelasan tiap tahap, cara kerja HOG & SVM, pemenuhan syarat tugas,
hasil pengukuran, ablasi, batasan, dan glosarium).

---

## Alur pipeline

```
Foto struk
   │
   ├─ [1] Deteksi area kertas      Otsu + connected components + rasio aspek
   ├─ [2] Grayscale + denoise      median blur 3x3
   ├─ [3] Binarisasi               Otsu global / Sauvola adaptif
   ├─ [4] Upscale adaptif          tinggi baris -> 26 px
   ├─ [5] Deskew                   profil proyeksi horizontal
   ├─ [6] Segmentasi baris         proyeksi horizontal + Otsu 1D
   ├─ [7] Segmentasi karakter      connected components + split lembah
   ├─ [8] Normalisasi              32x32, rasio aspek dijaga
   ├─ [9] Ekstraksi fitur          HOG manual (324 dimensi)
   ├─ [10] Klasifikasi karakter    Linear SVM
   ├─ [11] Rekonstruksi baris      spasi dari median celah
   └─ [12] Koreksi leksikon        Levenshtein ke 3.641 kata
```

Tidak ada satu pun tahap yang memakai model pre-trained atau API.

---

## Hasil

Diuji pada **150 baris sintetis** (render dari dataset, dengan degradasi
noise/blur/kontras):

| Konfigurasi | CER | WER | Exact match |
|---|---:|---:|---:|
| Otsu | 21,33% | 51,39% | 64,00% |
| Sauvola | 31,14% | 50,89% | 66,67% |
| Otsu + deskew | 21,18% | 54,61% | 62,67% |
| Otsu + deskew + leksikon | 21,04% | 49,52% | 66,67% |
| **Sauvola + deskew + leksikon** | **29,75%** | **46,08%** | **72,00%** |

**Akurasi klasifikasi karakter: 99,73%** (validasi 8.019 sampel).

Catatan penting:

- Akurasi **karakter** tinggi karena setiap karakter dipotong rapi.
- **CER/WER** lebih rendah karena kesalahan menumpuk: satu huruf salah
  merusak satu kata, dan kesalahan segmentasi merusak seluruh baris.
- Ini fenomena normal pada OCR klasik dan justru menjadi bahan analisis
  yang jujur untuk laporan.

---

## Struktur file

| File | Fungsi |
|---|---|
| `config.py` | Semua parameter (charset, ukuran, ambang, hyperparameter) |
| `render_synthetic.py` | Render 6.026 teks jadi gambar + label karakter otomatis |
| `preprocess.py` | Grayscale, denoise, binarisasi Otsu/Sauvola/adaptive |
| `paper.py` | Deteksi & potong area kertas struk |
| `deskew.py` | Estimasi & koreksi kemiringan |
| `segment.py` | Segmentasi baris & karakter |
| `features.py` | HOG manual (324 dimensi) |
| `train_char.py` | Latih Linear SVM, ekspor `.joblib` + `.json` |
| `lexicon.py` | Koreksi kata (Levenshtein) |
| `ocr_pipeline.py` | Pipeline lengkap + CLI |
| `evaluate.py` | CER, WER, ablasi per tahap |
| `visualize.py` | Gambar tiap tahap, confusion matrix, grafik ablasi |

Folder:
- `data/` — dataset karakter & baris uji (`.npz`)
- `models/` — `char_svm.joblib`, `char_svm.json`
- `output/` — hasil evaluasi & visualisasi

---

## Cara menjalankan

```powershell
cd ML/ocr_classical

# 1. Bangun data latih karakter (~5-10 menit)
python render_synthetic.py --max-lines 2500 --samples-per-line 2

# 2. Latih classifier karakter
python train_char.py

# 3. Evaluasi + ablasi
python evaluate.py

# 4. Visualisasi untuk laporan
python visualize.py

# 5. OCR satu gambar
python ocr_pipeline.py path\ke\struk.jpg
```

---

## Keputusan desain penting

### 1. Label otomatis dari render

Dataset hanya berisi teks, bukan gambar berlabel. Solusinya: render teks
memakai font monospace, lalu catat posisi tinta setiap huruf. Label
karakter didapat **gratis** tanpa anotasi manual, dan jumlahnya tak terbatas.

**Kunci:** bbox pelatihan harus **tight** (dari piksel tinta), bukan dari
metrik font. Saat inferensi, segmentasi juga menghasilkan tight bbox.
Kalau tidak sama, proporsi huruf berbeda dan akurasi jatuh drastis
(huruf "T" terbaca "Y").

### 2. Upscale adaptif

Teks kecil (tinggi 10-15 px) sulit disegmentasi: stroke 1 px mudah
terputus. Citra diperbesar sampai tinggi baris ~26 px. Ini menaikkan
jumlah baris tersegmentasi dari 22/30 menjadi 28/30.

### 3. Ambang baris hibrida

- **Citra bersih**: ada baris yang benar-benar kosong → pakai ambang absolut.
- **Foto struk**: tekstur kertas membuat hampir semua baris bertinta →
  pakai Otsu 1D pada profil baris.

Pembeda: rasio baris kosong. Di bawah 5% → dianggap foto.

### 4. HOG ditulis manual

`features.py` tidak memakai `skimage.hog`. Implementasi manual dipilih
karena versi Dart nanti harus meniru perhitungan yang sama persis.
Rumus lengkap ada di docstring modul.

### 5. Spasi dari median celah

Celah dalam kata dan antar kata saling tumpang tindih. Ambang tetap gagal.
Yang stabil: median celah baris itu, lalu celah > 2× median = batas kata.

---

## Batasan yang diketahui

| Batasan | Penyebab | Dampak |
|---|---|---|
| Font tipis terpecah | Stroke 1 px hilang saat binarisasi | Segmentasi berlebih (1 huruf jadi 3-5 komponen) |
| Huruf bersambung | Blur thermal menyatukan tinta | Dua huruf jadi satu bbox |
| Foto struk asing | Dataset COCO berisi struk Malaysia/Italia | Leksikon Indonesia tidak membantu |
| Garis pemisah | Rule horizontal menyatukan baris | Sudah ditangani Otsu 1D, tapi belum sempurna |

**Rekomendasi pengembangan:** augmentasi erosi/ dilatasi lebih agresif,
segmentasi berbasis contour dengan analisis proyeksi vertikal per komponen,
dan pengujian pada foto struk Indonesia asli.

---

## Data uji COCO

`ML/OCR Struk Belanja.v2i.coco/` berisi 771 foto struk dengan anotasi
bounding box (4 kelas: `struk_belanja`, `info_toko`, `item_belanja`,
`total`).

**Perhatian:** isinya struk Malaysia (`MR. D.I.Y. SDN BHD`, `TESCO MALAYSIA`)
dan Italia (`RISTORANTE BELLAGIO`), bukan struk Indonesia. Dataset ini
cocok untuk menguji ketahanan terhadap foto asli, tetapi **tidak cocok**
untuk mengukur akurasi pada domain Indonesia karena kosakatanya berbeda.

---

## Ekspor untuk aplikasi Flutter

`models/char_svm.json` (117 KB) berisi:

```json
{
  "charset": "0123456789ABC...",
  "char_size": 32,
  "hog": {"cell": 8, "bins": 9, "block": 2},
  "svm": {
    "classes": [...],
    "coef": [[...]],
    "intercept": [...]
  },
  "confidence_threshold": 0.55
}
```

Aplikasi cukup menghitung `argmax(coef @ hog + intercept)`. Tidak perlu
library ML apa pun di sisi Dart.

---

## Ringkasan kontribusi

| Tahap | Sumber |
|---|---|
| Deteksi kertas | Ditulis sendiri (Otsu + CC) |
| Binarisasi | Ditulis sendiri (Otsu, Sauvola) |
| Deskew | Ditulis sendiri (profil proyeksi) |
| Segmentasi | Ditulis sendiri (CC + split lembah) |
| HOG | Ditulis sendiri (manual, 324 dim) |
| Klasifikasi karakter | Ditulis sendiri (Linear SVM) |
| Koreksi leksikon | Ditulis sendiri (Levenshtein) |
| **Baca teks (OCR)** | **Tidak memakai library siap pakai** |
