# 10 — Praproses Citra di Sisi Aplikasi

`lib/ocr/image_quality.dart` menyediakan dua hal: **analisis kualitas**
citra (dipakai semua mesin) dan **rangkaian transformasi** untuk
menyiapkan gambar pada mesin yang tidak melakukan praproses sendiri.

> Mesin klasik (`ClassicalOcrEngine`) meminta **citra asli**
> (`needsRawImage = true`) karena ia melakukan deskew, binarisasi
> Sauvola, dan upscale sendiri. Pada jalur itu, `ImagePreprocessor` hanya
> menjalankan `analyze()` untuk laporan kualitas di UI, **tanpa mengubah**
> gambar.

## 1. Laporan Kualitas (`ImageQualityReport`)

| Field | Arti |
|---|---|
| `brightness` | Luminansi rata-rata (0 hitam – 255 putih) |
| `blurScore` | Variansi Laplacian — makin tinggi makin tajam |
| `rotationDegrees` | Derajat rotasi yang diterapkan agar teks tegak |
| `width`, `height` | Dimensi yang dianalisis |
| `preparedPath` | Berkas citra yang benar-benar dikirim ke mesin OCR |
| `lowLight`, `blurry`, `rotated` | Ambang lulus/tidak |
| `hasIssue` | `lowLight || blurry || rotated` |
| `issues` | Daftar pesan siap tampil di UI |

Ambang:

- Blur: `kBlurThreshold = 45` (variansi Laplacian) — di bawahnya dianggap
  buram dan dinilai lebih baik memberi peringatan daripada gagal.
- Cahaya: `kLowLightThreshold = 82` (luminansi rata-rata).

## 2. Analisis (`analyze`)

`analyze(sourcePath:)` mengukur kecerahan, ketajaman, dan kemiringan
tanpa menulis berkas baru; `preparedPath` menunjuk ke berkas sumber.
Dipakai untuk mesin `needsRawImage = true`.

Estimasi kemiringan (`_estimateSkewDegrees`) memakai **profil proyeksi**
dua iterasi, dibatasi ±15°.

## 3. Penyiapan untuk Mesin Lain (`prepare`)

`prepare(sourcePath:, outputPath:)` menulis citra siap-OCR ke
`outputPath`:

1. **Bake orientation** (EXIF) agar rotasi kamera tidak tertinggal.
2. Ukur kecerahan & blur; estimasi kemiringan lalu rotasi bila perlu.
3. Batasi dimensi terpanjang ke **3000 px** (`_maxDimension`). Batas ini
   menjaga memori (bitmap ~27 MB) sekaligus tetap cukup besar untuk foto
   galeri di mana struk hanya menempati sebagian bingkai.
4. Normalisasi kecerahan adaptif (gamma tersolusi menuju rata-rata 128)
   bila citra gelap.
5. **Unsharp mask** (radius 2, amount 1,2) hanya bila terdeteksi buram.

Tidak ada upscale buta maupun CLAHE — keduanya merusak binarisasi dan
segmentasi mesin klasik.

## 4. Variasi untuk Pemindaian Ulang (`OcrService._retryOcr`)

Strategi cadangan yang tersedia (dipilih otomatis oleh `OcrService`):

| Metode | Fungsi |
|---|---|
| `preparePerspectiveCorrected` | Pangkas area struk dan luruskan distorsi proyektif |
| `prepareCrop` | Potong ke daerah kertas/objek dan perbesar (padding 4%) |
| `prepareBinarized` | Binarisasi adaptif lokal (rata-rata jendela 31, ambang rata-rata−8) + pembersihan bintik gelap di tepi |
| `prepareRotated` | Putar 90° untuk foto galeri yang tersimpan menyamping |
| `prepareSharpened` | Pertajam ulang |

## 5. Deteksi Kertas Struk

- `detectInkBounds(path)` — bounding box area tinta (ambang persentil 2
  untuk tinta, persentil 60 untuk kertas) pada grayscale 400 px.
- `detectPaperBounds(path)` — bounding box blob terang terbesar
  (4-konektivitas), memakai `_brightLevel` (akumulasi histogram ≥ 4% dari
  putih ke bawah). Filter: rasio fill ≥ 0,55, aspek 0,4–4,5, area ≥ 5%.
- `containsReceiptPaper(path)` — gerbang "ada struk di foto atau tidak",
  dipakai di `ScanReceiptScreen` dan saat OCR menghasilkan teks kosong.

`ImageRegion { left, top, right, bottom; width; height }` adalah tipe
kotak yang dikembalikan.

> `_brightLevel` lahir dari perbaikan regresi: versi pertama memakai
> persentil sederhana sehingga **semua foto asli ditolak**; akumulasi
> histogram membuat ambang tahan terhadap gradasi pencahayaan dan noise
> JPEG. Regresi ini dijaga `test/paper_gate_test.dart`.

## 6. Peruul `ImagePreprocessor` (`package:image`)

`ImagePreprocessor` menepati batas: paket `image` hanya dipakai untuk
**dekode, resize, rotasi, dan konversi format** berkas citra — bukan
untuk algoritma pengenalan. Seluruh algoritma yang menentukan akurasi
OCR (Sauvola, HOG, segmentasi, decoding) berada di
`lib/ocr/classical/` dan ditulis sendiri.
