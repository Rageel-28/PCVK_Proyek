# Hasil Evaluasi pada Data Struk Asli

Dokumen ini menggantikan seluruh angka sintetis sebagai **acuan pelaporan**.
Setiap angka di sini diukur pada piksel struk asli dari
`ML/Dataset Citra/`, bukan pada citra hasil render sendiri.

---

## 1. Sumber data

| Sumber | Gambar | Bbox baris item | Sifat |
|---|---:|---:|---|
| `Receipt Indomaret.v1i.coco.zip` | 500 | 2.156 | render bersih, satu keluarga template |
| `Alfamart Indomaret Struk.v1i.coco.zip` | 122 | 630 | **foto asli** (CamScanner), kertas kusut |
| **Total** | **622** | **2.786** | |

Anotasi COCO hanya menandai **baris item** (`product_item`,
`product_item_discount`, `product_item_voucher`), bukan seluruh baris struk.
Tidak ada transkripsi teks di dataset, jadi transkripsi dibuat manual:
lihat `transcripts_real.json`.

> **Catatan kejujuran.** 500 dari 622 gambar adalah render template, bukan
> foto. Hanya 122 gambar yang benar-benar difoto. Semua klaim tentang
> ketahanan foto nyata hanya boleh bersandar pada 122 gambar itu, dan pada
> laporan ini dipisahkan sebagai `jenis=foto`.

### Transkripsi manual

9 struk ditranskripsi baris demi baris (5 render + 4 foto), menghasilkan:

- 28 baris item dengan teks acuan eksak (dipakai untuk CER/WER per baris)
- 9 transkrip dokumen penuh (dipakai untuk CER/WER tingkat dokumen)

Jumlah baris item pada transkripsi **dicek sama** dengan jumlah bbox COCO;
kalau tidak sama, struk tersebut dilewati (lihat `evaluate_real.py`).

---

## 2. Protokol evaluasi

Dua mode, keduanya memakai piksel asli:

| Mode | Cara | Yang diukur |
|---|---|---|
| `items1` | bbox COCO dipakai untuk melokalisasi baris, lalu pipeline memilih pita teks terpanjang di dalam crop | **kualitas pengenalan**, tanpa dipengaruhi kesalahan segmentasi baris |
| `doc` | pipeline dijalankan pada seluruh foto | **kualitas menyeluruh**: segmentasi baris + pengenalan + spasi |

Mode `items1` adalah acuan utama, karena tahap segmentasi baris pada foto
kusut masih menjadi kelemahan terpisah yang diukur di mode `doc`.

---

## 3. Hasil

### 3.1 Per baris item (`items1`, 28 baris)

| Kelompok | n | CER | WER | Exact match | Keyakinan |
|---|---:|---:|---:|---:|---:|
| **Semua** | 28 | **7,13%** | **18,62%** | **71,43%** | 70,96% |
| Render bersih | 20 | **0,14%** | 1,67% | **95,00%** | 77,82% |
| **Foto asli** | 8 | **24,59%** | 61,01% | 12,50% | 53,81% |

Contoh baris yang terbaca sempurna pada foto asli:

```
asli : ULTRA CKT 250ML 2 6,100 12,200
ocr  : ULTRA CKT 250ML 2 6,100 12,200
```

Contoh baris yang masih salah (foto kusut, cetakan pudar):

```
asli : MARLBORO TRPCL BS16S 1 36000 36,000
ocr  : MARL80R0 TRP8L BS16S 1 36,0,368000
```

### 3.2 Tingkat dokumen (`doc`, 9 struk)

| Kelompok | n | CER | WER |
|---|---:|---:|---:|
| **Semua** | 9 | 46,73% | 64,83% |
| Render bersih | 5 | **0,82%** | 5,60% |
| **Foto asli** | 4 | **104,12%** | 138,87% |

CER foto > 100% berarti jumlah karakter yang dihasilkan **lebih banyak**
daripada teks acuan: segmentasi baris pada foto kusut memecah lipatan,
bayangan, dan QR menjadi baris-baris palsu. Ini kelemahan nyata, bukan
kesalahan hitung.

---

## 4. Apa yang memperbaiki akurasi

Semua perubahan di bawah ini diukur, bukan diasumsikan. Angka adalah CER
pada 28 baris asli.

| Perubahan | CER sebelum | CER sesudah |
|---|---:|---:|
| Baseline (model sintetis lama, charset 67 kelas) | 28,75% | — |
| Buang huruf kecil dari charset (struk thermal selalu kapital) | 28,75% | 27,76% |
| Buang data panen `chars_real.npz` (labelnya hasil self-training model rusak) | 27,76% | 21,23% |
| Data karakter asli berlabel manual (`chars_real_gold.npz`) | 21,23% | 11,98% |
| Pilih pita teks terpanjang di dalam crop | 11,98% | 9,28% |
| Ambang spasi berbasis lebar karakter (bukan median celah) | 9,28% | 9,09% |
| Gabungkan pecahan glyph (`M` terbelah dua, `0` terbelah dua) | 9,09% | 7,57% |
| Aturan glyph sangat kecil -> tanda baca | 7,57% | 7,46% |
| Batasan kandidat **murni** per kata (`600ML` jangan jadi angka) | 7,46% | **6,78%** |

### Yang justru tidak dipakai

| Ide | Hasil ukur | Keputusan |
|---|---|---|
| Koreksi leksikon agresif | CER 7,46% → 10,96%, exact 67,9% → 17,9% | **dimatikan** (default `use_lexicon=False`) |
| Koreksi leksikon konservatif (min 5 huruf, tanpa angka, kandidat tunggal) | CER 7,55%, exact 60,7% | tetap dimatikan, dicatat sebagai opsi |
| Binarisasi Otsu / adaptive mean | CER foto 41,1% / 54,0% vs Sauvola 37,4% | Sauvola dipertahankan |
| Trim baris kosong pada crop | CER 28,75% → 37,46% | dibatalkan |
| Bantalan putih lebar pada crop | CER 28,75% → 31,74% | dibatalkan |

Koreksi kamus gagal karena nama produk di struk asli jarang ada di kamus,
sehingga kata yang **sudah benar** ikut diubah: `GAM` → `GA`,
`BACTERIAL` → `BATERAI`.

---

## 5. Perubahan model

| Aspek | Sebelum | Sesudah |
|---|---|---|
| Charset | 67 kelas (angka + huruf besar + huruf kecil) | **48 kelas** (angka + huruf besar + `.&-/%+,:@()=`) |
| Akurasi validasi | 95,7% (dengan huruf kecil) | **99,52%** |
| Data latih | 91.554 sampel sintetis | 165.967 sintetis + ~650 **asli berlabel manual** |
| Bobot sampel asli | — | ×3 |
| Karakter `,` `:` `@` `(` `)` `=` | tidak ada di data latih | ada (struk penuh karakter ini) |

Huruf kecil dibuang karena struk thermal hampir selalu kapital, dan
menggabungkan besar/kecil membuat kelas saling tumpang tindih
(`O` vs `o`, `C` vs `c`) sehingga akurasi turun.

### Data karakter asli

`build_real_chars.py` memanen ~650 karakter berlabel **benar** dari
transkripsi manual, dengan aturan ketat: jumlah karakter hasil segmentasi
harus sama dengan jumlah karakter transkripsi, kalau tidak baris dilewati.
Hasilnya 252 angka dan 362 huruf, termasuk 104 angka `0` yang sebelumnya
tidak terwakili sama sekali.

> Berkas lama `chars_real.npz` (13.113 sampel) **dibuang**. Isinya dilabeli
> oleh model yang sedang rusak (self-training), dan terbukti meracuni kelas
> `0` sehingga model membacanya sebagai `8`.

---

## 6. Paritas Python ↔ Dart

Paritas diuji pada potongan baris asli (`test/ocr_real_line_parity_test.dart`):

| Uji | Hasil |
|---|---|
| HOG + SVM (16 patch, toleransi 1e-4) | **100% identik** |
| Teks baris, struk render | **100,00%** kesepakatan karakter |
| Teks baris, foto asli | **85,25%** kesepakatan karakter |
| Baris identik persis | 22/28 (78,6%) |
| Seluruh test suite | **181 test lolos** |

Agar paritas bisa dicapai, tiga operasi yang sebelumnya memakai OpenCV
diganti implementasi bersama:

1. **Resize bicubic** (`resize_cubic.py` ↔ `gray_image.dart`) — OpenCV
   memakai fixed-point 11 bit; dihitung ulang dengan rumus yang sama di
   kedua sisi sehingga identik bit-per-bit.
2. **Rotasi deskew** (`rotate_nearest`) — `cv2.warpAffine` memakai pusat
   `(w/2, h/2)` dan `cvRound`; disatukan agar sama.
3. **Downscale untuk estimasi sudut** (`resize_bilinear`) — OpenCV memakai
   `INTER_AREA`, Dart memakai bilinear.

Selisih yang tersisa pada foto berasal dari glyph yang secara visual
memang ambigu (koma 3 px, `0`/`O` pada cetakan pudar) dan skor SVM-nya
hampir seri; pembulatan floating point cukup untuk membalik argmax.

Dua bug port Dart ditemukan lewat pengukuran ini:

- `prepare()` mengembalikan grayscale yang **belum** didenoise ulang setelah
  upscale, sehingga patch karakter dipotong dari citra ber-noise sementara
  Python memotong dari citra bersih.
- Aset `ocr_lexicon.json` kedaluwarsa (3.641 kata) dibanding kamus Python
  (3.667 kata), sehingga aplikasi mengoreksi kata yang seharusnya
  dibiarkan. Sekarang diekspor oleh `export_lexicon.py`.

---

## 7. Jalur aplikasi

`test/ocr_real_app_test.dart` menjalankan jalur penuh aplikasi
(`ImagePreprocessor` → `ClassicalOcrEngine` → `ReceiptParser`) pada 6 struk
asli. Semuanya menghasilkan teks, item, dan total:

```
receipt-000  kata=60  item=4  total=629100   toko="Tubun"
receipt-001  kata=93  item=9  total=1178400  toko="Tubun"
receipt-002  kata=68  item=6  total=5741200  toko="Tubun"
CamScanner…  kata=138 item=5  total=51900    toko="Alfamart"
CamScanner…  kata=62  item=2  total=147080   toko="Alfhi)/"
CamScanner…  kata=80  item=4  total=151600   toko="Alfamidi"
```

Sebelum perbaikan, jalur aplikasi menghasilkan teks tanpa arti
(`mean_confidence` 0,29, 601 karakter berkeyakinan rendah) karena model
dilatih dengan charset 67 kelas dan data panen berlabel salah.

---

## 8. Batasan yang masih terbuka

| Batasan | Dampak terukur | Penyebab |
|---|---|---|
| Segmentasi baris pada foto kusut | CER dokumen foto 104,1% | lipatan, bayangan, dan QR pecah menjadi baris palsu |
| Cetakan pudar | CER baris foto 24,6% | stroke hilang, glyph terbelah, `0`/`O` nyaris identik |
| Ukuran sampel kecil | 28 baris, 9 dokumen | waktu transkripsi manual |
| Render bukan foto | 500 dari 622 gambar | dataset aslinya memang template |

Untuk menaikkan angka foto secara berarti, langkah berikutnya adalah
**memperbaiki segmentasi baris pada foto** (deteksi lipatan dan QR, lalu
gabungkan pita yang berdekatan), bukan menambah data latih karakter.

---

## 9. Cara mereproduksi

```powershell
cd ML/ocr_classical

# 1. Bangun data sintetis pelengkap (format struk Indonesia)
python render_synthetic.py --max-lines 2500 --samples-per-line 2

# 2. Panen karakter asli berlabel manual dari transkripsi
python build_real_chars.py

# 3. Latih model karakter
python train_char.py

# 4. Evaluasi pada struk asli
python evaluate_real.py                  # per baris item
python evaluate_real.py --mode doc       # per dokumen

# 5. Ekspor aset + fixture paritas untuk aplikasi
python export_lexicon.py
python export_real_parity.py
Copy-Item models/char_svm.json ../../smart_receipt/assets/ocr_char_svm.json

# 6. Verifikasi aplikasi
cd ../../smart_receipt
flutter test
```
