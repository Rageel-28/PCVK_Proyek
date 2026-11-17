# Panduan Pelabelan Glyph Struk Thermal

Langkah untuk memperbaiki OCR pada struk **foto thermal** (Indomaret, Alfamart,
dsb.). Diagnosis sebelumnya membuktikan penyebab kegagalan adalah *domain gap*:
model hanya pernah melihat font sistem (Consolas/Courier), sedangkan struk
memakai font dot-matrix printer thermal yang tidak punya padanan di Windows
(korelasi maksimum 0,54).

Satu-satunya perbaikan yang terbukti adalah **melatih dengan glyph thermal
nyata**. Panduan ini menjelaskan cara mendapatkannya.

---

## 1. Citra mana yang dilabeli?

Dataset: `ML/OCR - Struk Belanja.v10-raw-dataset.coco` — **1.505 citra**
(train 1.204, valid 151, test 150).

### Yang dipakai: 71 struk Indonesia

Dari 1.505 citra, **71 sudah teridentifikasi sebagai struk Indonesia** dari
nama berkasnya (`struk-*`, `nota-*`, `*indomaret*`, `*kasir*`):

| Split | Indonesia | Total |
|---|---:|---:|
| train | 51 | 1.204 |
| valid | 11 | 151 |
| test | 9 | 150 |
| **Total** | **71** | **1.505** |

Sisanya 1.434 citra bernama angka polos (`0_jpg`, `100_jpg`) dan isinya
campur Malaysia/Amerika/Indonesia — tidak bisa dipisah dari nama berkas,
jadi **tidak dipakai**.

Contoh isi 71 citra tersebut: Indomaret (Cidatar–Kab Garut), TOKO. H. ENDANG,
TOKO BANG-AN ARITONANG, TOKO KOMPUTER (Subang), sy_beautyskin (Kendal, "Kasir
Pintar"), SANTA SUPERMARKET, MAHKOTA TEKSTIL.

Penanda bahasanya khas Indonesia: `TERIMA KASIH`, `ANDA HEMAT`,
`TOTAL BELANJA`, `TUNAI`, `KEMBALI`, `PPN`, `Rp`.

### Kenapa bukan dataset v2i

`OCR Struk Belanja.v2i.coco` (771 citra) hampir seluruhnya struk Malaysia
(RM, GST, Selangor, Johor) dan Amerika. Hanya ±4 citra bertanda Indonesia.
v10 jauh lebih baik untuk tujuan ini.

### Penting: anotasi COCO TIDAK memuat teks

Dataset ini berformat COCO, tetapi anotasinya **hanya kotak area**, bukan
transkripsi teks. Isi tiap anotasi (terverifikasi dari berkas JSON):

```
{'id': 1, 'image_id': 0, 'category_id': 2,
 'bbox': [32.23, 92.55, 216.43, 18.47], 'iscrowd': 0,
 'area': 3997.46, 'segmentation': []}
```

| Kategori | Arti |
|---|---|
| `OCR-Struk-Belanja` | seluruh struk |
| `nama_toko` | kotak berisi nama toko |
| `tanggal_waktu` | kotak berisi tanggal/jam |
| `line_item` | kotak berisi baris barang |
| `total_belanja` | kotak berisi total |

Tidak ada kunci `text`, `transcription`, maupun `label`. Artinya anotasi
itu memberi tahu **DI MANA** teks berada, bukan **APA** isinya.

Model OCR memerlukan label **per karakter** — untuk setiap glyph harus
diketahui huruf apa. Karena itu teks setiap baris tetap harus diketik
manusia. Anotasi COCO hanya bisa dipakai untuk:

- memvalidasi bahwa segmentasi kita kena area yang benar, dan
- mengurutkan prioritas baris yang dilabeli (`line_item`, `total_belanja`).

Total anotasi: 6.637 (train) + 844 (valid) + 790 (test) = **8.271 kotak**.
Bila kelak Anda ingin memakai anotasi ini untuk pelabelan semi-otomatis,
kotak `line_item` bisa dicocokkan dengan hasil segmentasi kita — tetapi
teksnya tetap perlu diketik.

### Pilih berdasarkan skor, bukan urutan folder

```bash
cd ML/ocr_classical/diagnostic
python rank_labeling_candidates.py            # semua 71 citra Indonesia
python rank_labeling_candidates.py --split train   # hanya split train
```

Menghasilkan `labeling_candidates.csv` berisi, per citra:

| Kolom | Arti |
|---|---|
| `glyphs` | jumlah glyph yang bisa diekstrak (makin banyak makin cepat terkumpul) |
| `median_glyph_h` | tinggi glyph median; **< 12 px sulit dibaca** |
| `median_aspect` | rasio aspek; thermal ≈ 0,30–0,40, laser ≈ 0,50–0,65 |
| `score` | gabungan ketiganya (0–1) |
| `quality` | **A** = layak, **B** = marginal, **C** = tidak layak |

**Target yang dikejar:** struk Indomaret Anda punya rasio aspek 0,33 dan tinggi
glyph ±50 px setelah diperbesar. Utamakan `quality = A`, lalu `B` bila perlu.

Urutan di CSV juga menjadi **urutan antrean** di alat pelabelan, jadi citra
terbaik muncul lebih dulu.

### Berapa banyak?

| Target | Citra | Glyph | Perkiraan waktu |
|---|---:|---:|---|
| Uji kelayakan | **10–20** | ±3.000–6.000 | 30–60 menit |
| Memakai seluruh set | 71 | ±15.000–20.000 | 3–5 jam |

Mulai dari 10–20 citra. Bila akurasi pada struk asli naik berarti, lanjutkan.

---

## 2. Cara melabeli

```bash
cd ML/ocr_classical/diagnostic
python label_lines.py
```

### Tampilan

- Atas: seluruh citra, baris aktif disorot merah
- Tengah: baris aktif **diperbesar** agar mudah dibaca
- Bawah: kotak ketik + jumlah glyph yang terdeteksi

### Alur kerja

1. Baca baris yang disorot pada gambar perbesaran.
2. Ketik teksnya **tanpa spasi** — `JAVANATEHMLATI350`
3. Tekan **Enter** → tersimpan, lanjut baris berikutnya.

| Tombol | Fungsi |
|---|---|
| **Enter** | simpan & lanjut |
| **Lewati baris** | lewati satu baris (mis. tidak terbaca) |
| **Lewati citra** | tolak seluruh citra ini — pakai bila citra terlalu miring, tertekuk, atau buram |
| **Undo** / Ctrl+Z | batalkan label terakhir |
| **ASCII** / Tab | lihat glyph sebagai teks ASCII untuk memastikan segmentasi |
| **Esc** | keluar, progres tersimpan |

### Citra miring, tertekuk, atau buram

Alat ini menjalankan **koreksi kemiringan (deskew) otomatis** sebelum
segmentasi: sudut dicari dengan memaksimalkan variansi profil proyeksi,
rentang ±25° (foto HP sering lebih miring daripada scan).

Hasil terukur pada citra yang paling buruk di dataset:

| Citra | Tanpa deskew | Sesudah deskew |
|---|---:|---:|
| `struk-14…89120d` (miring −6,9°) | 3 baris, 47 glyph | **14 baris, 343 glyph** |

Bila setelah deskew hasilnya masih kacau, layar menampilkan peringatan
**PERIKSA**:

- `citra dimiringkan X°` — kemiringan besar
- `baris ini sangat tinggi - mungkin 2 baris menyatu`
- `N glyph dalam satu baris - mungkin kacau`

Untuk kasus itu, tekan **Lewati citra**. Ini keputusan yang sah: **label
yang salah lebih merusak data latih daripada kekurangan beberapa citra.**
Anda tidak perlu memaksakan semua 71 citra terpakai.

### Aturan yang harus dipatuhi

**Jumlah karakter yang diketik harus sama dengan jumlah glyph yang
terdeteksi.** Alat ini selalu menampilkan `glyph terdeteksi: N`.

- Cocok → label dipetakan 1:1 ke setiap glyph
- Tidak cocok → tetap tersimpan, tetapi ditandai **MISMATCH** saat ekspor dan
  **tidak ikut** ke data latih

Ini disengaja. Bila dipaksakan, satu huruf bergeser akan menggeser **seluruh**
label di baris itu dan merusak data. Lebih baik satu baris hilang.

Bila sering mismatch, tekan **ASCII** — biasanya ada bintik debu yang dihitung
sebagai glyph, atau dua huruf menyatu.

---

## 3. Hasilnya

```bash
python label_lines.py --stats     # progres
python label_lines.py --export    # bangun ulang dataset glyph
```

| Berkas | Isi |
|---|---|
| `labels/lines_labeled.csv` | `image, line_index, text` — sumber kebenaran |
| `labels/glyph_labels.csv` | `image, line_index, glyph_index, label, top, left, height, width` |

`glyph_labels.csv` inilah yang dipakai untuk melatih model baru.

---

## 4. Setelah pelabelan

Data siap dipakai untuk melatih ulang:

1. Potong glyph memakai koordinat di `glyph_labels.csv`
2. Normalisasi 32×32 (fungsi `normalize_char` yang sama)
3. Ekstrak HOG + Zoning → 340 dimensi
4. StandardScaler → Linear SVM
5. **Uji pada struk Indomaret Anda**, bukan pada data sintetis

Yang perlu dijaga: **pembagian data harus per citra**, bukan per glyph.
Glyph dari citra yang sama punya kondisi cetak identik; bila dibagi per glyph,
model hanya menghafal citra dan angka akurasinya menipu.

---

## 5. Berkas pendukung

| Berkas | Fungsi |
|---|---|
| `dataset_source.py` | Sumber citra: dataset v10, filter 71 struk Indonesia |
| `rank_labeling_candidates.py` | Nilai & urutkan citra mana yang layak dilabeli |
| `make_contact_sheet.py` | Lembar kontak: lihat 12 citra sekaligus |
| `label_lines.py` | GUI pelabelan per baris |
| `diag_*.py`, `experiment_*.py` | Skrip diagnosis penyebab kegagalan OCR |

### Mengubah sumber dataset

Semua path dataset terpusat di `dataset_source.py`:

```python
DATASET = ROOT / "ML" / "OCR - Struk Belanja.v10-raw-dataset.coco"
INDONESIA_MARKERS = ("struk", "nota", "indomaret", "alfamart",
                     "alfamidi", "kasir")
```

Bila ingin memakai dataset lain atau menambah penanda merek, cukup ubah dua
konstanta itu — kedua alat lain mengikutinya.

Cek cepat isi dataset:

```bash
python dataset_source.py
```
