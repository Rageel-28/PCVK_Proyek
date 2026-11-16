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
   ├─ [9] Ekstraksi fitur          HOG 324 dim + Zoning 4x4 16 dim = 340
   ├─ [10] Penskalaan              StandardScaler (mean 0, variansi 1)
   ├─ [11] Klasifikasi karakter    Linear SVM
   ├─ [12] Rekonstruksi baris      spasi dari median celah
   └─ [13] Koreksi leksikon        Levenshtein ke 3.641 kata
```

Tidak ada satu pun tahap yang memakai model pre-trained atau API.

### Fitur: HOG + Zoning

| Kelompok | Dimensi | Yang direkam |
|---|---:|---|
| HOG | 324 | **Arah tepi** (bentuk goresan); sel 8x8, 9 bin, blok 2x2 L2-Hys |
| Zoning / Grid | 16 | **Sebaran tinta**; kanvas dibagi grid 4x4, tiap zona = kepadatan tinta 0..1 |
| **Gabungan** | **340** | `np.concatenate([hog, zoning])` |

Keduanya saling melengkapi: HOG membuang posisi demi arah, zoning membuang
arah demi posisi. `L` dan `7` dapat punya histogram gradien mirip, tetapi
zonanya berbeda jelas.

**StandardScaler wajib.** HOG bernilai ~0..0.6, zoning ~0..1 dengan rata-rata
~0.2. Tanpa penyamaan skala, dimensi bernilai absolut lebih besar mendominasi
margin SVM. Scaler di-fit pada data latih lalu parameternya diekspor ke JSON,
sehingga Dart menerapkan transformasi yang sama.

**PCA diuji dan tidak dipakai.** Dimensi 340 hampir seluruhnya berguna; PCA
justru menurunkan akurasi:

| Konfigurasi | Dimensi | Akurasi validasi |
|---|---:|---:|
| Tanpa PCA | 340 | **99,92%** |
| PCA 200 | 200 | 99,88% |
| PCA 120 | 120 | 99,71% |
| PCA 64 | 64 | 99,71% |
| PCA 32 | 32 | 99,05% |

---

## Hasil

**Perbandingan varian fitur** (data latih & validasi identik, seed 42;
35.432 / 4.832 sampel):

| Konfigurasi | Dimensi | Akurasi karakter |
|---|---:|---:|
| HOG saja, tanpa scaler | 324 | 99,81% |
| HOG saja, + StandardScaler | 324 | 99,83% |
| HOG+Zoning, tanpa scaler | 340 | 99,90% |
| **HOG+Zoning, + StandardScaler** | **340** | **99,92%** |

**GridSearchCV LinearSVC** (subsampel 12.000, 3-fold) memilih **C = 1,0**:

| C | Akurasi CV |
|---:|---:|
| 0,01 | 98,98% |
| 0,1 | 99,62% |
| **1,0** | **99,69%** |
| 10,0 | 99,68% |

Gamma tidak disertakan karena kernel yang dipakai linear. RBF memang punya
gamma, tetapi bobotnya tidak dapat diekspor sebagai satu matriks `coef`,
sehingga inferensi Dart akan jauh lebih berat.

**Perbandingan end-to-end** pada 120 baris uji yang sama
(`python experiment_endtoend.py`):

| Model | CER | WER | Exact match |
|---|---:|---:|---:|
| HOG saja | **30,25%** | 62,18% | 61,67% |
| **HOG+Zoning+Scaler** | 30,70% | **60,65%** | **62,50%** |
| Selisih | −0,44 poin | **+1,53 poin** | **+0,83 poin** |

Ini hasil yang campuran, dan dilaporkan apa adanya:

- **WER membaik 1,53 poin** — kata utuh lebih sering benar, dan inilah yang
  dilihat pengguna.
- **CER sedikit memburuk 0,44 poin** — jumlah karakter salah bertambah
  sedikit pada token yang sudah hampir benar.

Pola ini sama dengan yang tercatat pada koreksi leksikal: menarik token ke
bentuk yang lebih "khas" menyelamatkan kata utuh tetapi kadang mengubah
karakter yang sebenarnya sudah benar. Karena aplikasi menampilkan hasil per
kata, **WER lebih relevan** daripada CER.

Diuji pada **120 baris sintetis** (render dari dataset, dengan degradasi
noise/blur/kontras):

| Konfigurasi | CER | WER | Exact match |
|---|---:|---:|---:|
| Otsu | 32,35% | 66,32% | 55,00% |
| Sauvola | 50,97% | 69,10% | 55,00% |
| Otsu + deskew | 31,63% | 68,26% | 54,17% |
| Otsu + deskew + leksikon | 30,70% | 60,65% | 62,50% |
| **Sauvola + deskew + leksikon** | **47,72%** | **57,67%** | **63,33%** |

**Akurasi klasifikasi karakter: 99,92%** (validasi 4.832 sampel).

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
| `features.py` | HOG 324 dim + Zoning 4x4 16 dim, plus fungsi gabungan |
| `train_char.py` | Latih HOG+Zoning -> Scaler -> (PCA) -> SVM, GridSearchCV C |
| `experiment_pca.py` | Uji jumlah komponen PCA (hasil: tidak dipakai) |
| `experiment_endtoend.py` | Bandingkan HOG saja vs HOG+Zoning pada data uji sama |
| `lexicon.py` | Koreksi kata (Levenshtein) |
| `ocr_pipeline.py` | Pipeline lengkap + CLI |
| `evaluate.py` | CER, WER, ablasi per tahap |
| `visualize.py` | Gambar tiap tahap, confusion matrix, grafik ablasi |
| `export_golden_vectors.py` | Ekspor vektor uji paritas untuk Dart |
| `export_test_images.py` | Ekspor gambar uji end-to-end untuk Dart |

Folder:
- `data/` — dataset karakter, baris uji, golden vectors (`.npz`, `.json`)
- `models/` — `char_svm.joblib`, `char_svm.json`, `ocr_lexicon.json`
- `output/` — hasil evaluasi & visualisasi

---

## Implementasi di aplikasi (Dart)

Pipeline ini sudah diport ke Flutter dan **menggantikan Google ML Kit**.

| Berkas Dart | Padanan Python |
|---|---|
| `lib/ocr/classical/gray_image.dart` | buffer citra + resize bicubic + rotasi |
| `lib/ocr/classical/binarization.dart` | `preprocess.py` |
| `lib/ocr/classical/connected_components.dart` | `cv2.connectedComponents` |
| `lib/ocr/classical/paper_detector.dart` | `paper.py` |
| `lib/ocr/classical/deskew.dart` | `deskew.py` |
| `lib/ocr/classical/segmenter.dart` | `segment.py` |
| `lib/ocr/classical/hog.dart` | `features.py` (bagian HOG) |
| `lib/ocr/classical/zoning.dart` | `features.py` (bagian Zoning) |
| `lib/ocr/classical/char_classifier.dart` | `train_char.py` (scaler + PCA + SVM) |
| `lib/ocr/classical/lexicon_corrector.dart` | `lexicon.py` |
| `lib/ocr/classical/classical_ocr_engine.dart` | `ocr_pipeline.py` |

Aset: `assets/ocr_char_svm.json` (156 KB: 340 bobot + scaler) dan
`assets/ocr_lexicon.json` (34 KB).

### Verifikasi paritas

| Uji | Cakupan | Hasil |
|---|---|---|
| `test/ocr_classical_parity_test.dart` | HOG, Zoning, vektor gabungan, dan prediksi pada 16 patch | **100% identik** dengan Python (toleransi 1e-4) |
| `test/ocr_classical_pipeline_test.dart` | 8 gambar end-to-end | **8/8 teks identik** dengan Python |

Perbedaan numerik yang harus disamakan agar paritas tercapai:

1. **Pembulatan resize** — Python/cv2 membulatkan setengah ke nilai genap
   (`47 * 1.5 = 70.5` menjadi **70**, bukan 71).
2. **Urutan denoise** — citra yang diperbesar adalah grayscale yang
   **sudah** didenoise, bukan citra mentah.
3. **Konvensi rotasi** — `cv2.warpAffine` memakai `src = M⁻¹ · dst`;
   tanda suku sinus harus dibalik.
4. **Tepi Sauvola** — OpenCV `BORDER_REPLICATE` membagi dengan luas
   jendela penuh (`win × win`), bukan jumlah piksel yang terlihat.
5. **Interpolasi resize** — `INTER_CUBIC` memakai kernel Catmull-Rom
   dengan `a = -0.75`, bukan bilinear.
6. **PNG grayscale 1 kanal** — `pixel.g`/`pixel.b` bernilai 0, sehingga
   luma harus dibaca lewat kanal merah setelah normalisasi.

### Kontrak keluaran

`ClassicalOcrEngine.recognize()` mengembalikan **satu `OcrLine` per
kelompok kata**, bukan satu baris selebar gambar. `receipt_parser.dart`
mengelompokkan `OcrLine` berdasarkan `centerY` lalu memakai `left`/`right`
untuk memisahkan kolom nama dan harga; bila seluruh baris dikirim sebagai
satu `OcrLine`, kolom harga hilang dan total struk tidak terbaca.

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

# 6. Ekspor data uji paritas untuk Dart
python export_golden_vectors.py
python export_test_images.py
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
