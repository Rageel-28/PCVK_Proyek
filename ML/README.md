# ML — Pipeline OCR Klasik (pure machine learning)

Semua tahap OCR di sini dilatih dari nol dengan metode **machine learning
klasik**: HOG + SVM linear, n-gram language model, dan aturan bentuk. Tidak ada
pustaka OCR siap pakai (Tesseract, EasyOCR, PaddleOCR, ML Kit, API OCR) dan
tidak ada deep learning.

Target: Android + iOS, total model < 20 MB.

## Struktur

```
ML/
├── requirements.txt
├── ocr/
│   ├── config.py             charset, dimensi fitur, parameter, path
│   ├── corpus.py             korpus baris struk sintetis (label otomatis)
│   ├── render_dataset.py     render → tight bbox → normalisasi → augmentasi
│   ├── features.py           HOG + zoning + projection + hole (1844-d)
│   ├── train_char.py         LinearSVC + temperature scaling
│   ├── export_char.py        ekspor JSON untuk port Dart + golden vectors
│   ├── preprocess.py         deskew → Sauvola → upscale adaptif
│   ├── lines.py              deteksi pita baris
│   ├── rules.py              klasifikasi teks vs garis vs barcode
│   ├── words.py              segmentasi kata
│   ├── chars.py              segmentasi glyph (+ pemecahan & penggabungan)
│   ├── lm.py                 n-gram karakter
│   ├── lexicon.py            weighted Levenshtein + confusion cost
│   ├── decode.py             beam search, koreksi, normalisasi token
│   ├── templates.py          pencocokan baris tetap (alamat, footer, label)
│   ├── pipeline.py           recognize_page — jalur lengkap citra → teks
│   ├── export_lm.py          ekspor lm/lexicon/templates JSON
│   ├── evaluate.py           CER/WER/exact-line/metrik baris barang
│   ├── synth_receipts.py     generator struk held-out
│   ├── data/                 dataset hasil render (gitignored)
│   ├── models/               model hasil latih + ekspor JSON
│   └── output/               grafik, laporan, artefak (gitignored)
└── tests/                    pytest
```

## Alur lengkap

```bash
python3 -m pip install -r ML/requirements.txt
cd ML

# F0 — dataset
python3 -m ocr.render_dataset --n-lines 6000     # → ocr/data/chars.npz

# F1 — classifier karakter
python3 -m ocr.train_char                        # → ocr/models/char_clf.joblib
python3 -m ocr.export_char                       # → char_clf.json + golden_vectors.json

# F2/F3 — LM, leksikon, templat
python3 -m ocr.export_lm                         # → lm.json, lexicon.json, templates.json

# Evaluasi
python3 -m ocr.evaluate                          # fixture terkunci receipts_v2
python3 -m ocr.synth_receipts --count 16 --seed 777
python3 -m ocr.evaluate --fixtures ocr/output/synth_receipts

# Uji
python3 -m pytest tests/ -q
```

## Hasil

Fixture terkunci `smart_receipt/test/fixtures/receipts_v2` (10 struk, 138
baris, 107 baris barang):

| metrik | nilai |
|---|---|
| CER | 0.08% |
| WER | 0.46% |
| exact-line | 98.6% (136/138) |
| nama / harga / baris barang utuh | 100.0% / 98.1% / 98.1% |

Dua baris yang meleset sama-sama `4` tertutup (serif, atasnya membentuk
segitiga) yang terbaca `0` — `14.500`→`10.500` dan `9.454`→`9.450`. Bentuk
itu hanya muncul pada font sintetis fixture; `4` pada struk thermal asli
terbuka dan **seluruh harga struk asli terbaca benar**. Port Dart meleset
satu glyph tambahan di batas yang sama (`4.700`→`0.700`) karena beda
floating-point, jadi 135/138.

Fixture ini dipakai selama pengembangan, jadi anggaplah ia **batas atas**.
Set held-out (`synth_receipts`, font 14–22 px, rotasi ±2.5°, blur, noise,
pencahayaan tidak rata) — proksi domain-sejati yang tidak pernah dilihat
saat penyetelan — memberi **CER 1.54%, WER 5.71%, exact-line 90.1%,
nama 96.0%, harga 92.0%, baris utuh 89.3%**.

### Struk asli (foto ponsel)

Dua foto struk asli di `testdata/real/` dipakai untuk menguji di luar domain
sintetis. Font printer thermal jauh lebih sempit daripada font vektor
(aspek huruf ~0.42 vs ~0.85), jadi data latih sengaja diberi variasi
**penyempitan horizontal** (`AUG_CONDENSE_*`). Tanpa itu huruf `A` terbaca
`.` dan `D` terbaca `0`.

Struk asli juga mencampur huruf besar dan kecil (`Total Item`, `Kembalian`),
padahal charset sengaja tetap kapital. Huruf kecil dirender apa adanya
tetapi **dilabeli** versi kapitalnya, dalam pola realistis per baris
(kapital semua / Title Case / kecil semua; `AUG_LOWERCASE_*`). Huruf `l`
dikecualikan karena bentuknya identik dengan `I` — melatihnya sebagai `L`
membuat `TOTAL` terbaca `TOTAI`.

| struk | hasil |
|---|---|
| Alfamart (439×927, WhatsApp) | header `ALFAMART SIDOMORO GRESIK`, nama barang + semua harga benar |
| Indomaret (946×1216) | nama barang + semua harga benar, baris total benar |

Keterbatasan yang masih ada: `0`/`O` pada font tanpa garis miring masih
ambigu (dibantu aturan kata `normalize_alpha_token` dan leksikon), baris
total campuran huruf kecil (`Total Belanja`) kadang masih meleset, dan
`:`/`,`/`(`/`)` tidak ada di charset sehingga terbaca `.`.

**Foto di atas latar gelap.** Struk yang dipegang di atas kain/meja gelap
dulu menghasilkan 0 baris: patch solid global di `sauvola` memakai
persentil-2 sebagai acuan tinta, dan di citra seperti itu persentil-2 jatuh
ke **latar**, bukan tinta — seluruh latar jadi tinta dan semua baris menyatu.
Solusinya (di `preprocess._paper_mask`): cari blob terang terbesar (ambang
`0.60 × persentil-90`, open 9, close 21); bila blob itu **tidak** menyentuh
keempat tepi bingkai (artinya ada latar), potong citra ke blob itu dan buang
tinta di luar mask yang sudah di-erode. Struk yang memenuhi bingkai
(menyentuh 4 tepi) diproses seperti biasa tanpa pemotongan, jadi tidak ada
regresi. Uji: struk Indomaret asli berlatar gelap (`struk121.jpeg`) dari 0
menjadi 25 baris, `TOTAL BELANJA : 86,000` terbaca dan total parser 86.000
benar.


## Arsitektur

```
citra → [potong daerah kertas bila ada latar gelap]
      → deskew → Sauvola → upscale adaptif → buang tinta di luar kertas
      → deteksi pita baris (teks / garis / barcode)
      → segmentasi kata → segmentasi glyph
      → classifier karakter (LinearSVC)
      → beam search + LM n-gram + leksikon (cadangan)
      → pencocokan templat baris tetap
      → teks
```

### Keputusan desain yang menentukan

- **Konvensi citra**: `255 = tinta`, `0 = kertas`.
- **Bbox tight dari piksel tinta**, bukan metrik font — supaya distribusi
  latih sama dengan hasil segmentasi saat inferensi.
- **Booster kelas langka**: `Q`, `V`, `Z`, `&`, `+` ditambah lewat baris
  sintetis sampai ≥ 2000 sampel.
- **Cap kelas dominan**: baris `-----` membuat `-` menguasai ~44% dataset.
- **Split bebas kebocoran**: `text_index` menandai baris sumber tiap karakter,
  jadi pembagian train/val/test per baris, bukan per karakter.
- **Satu model linear + temperature scaling.** Ensemble tidak menambah akurasi
  dan lebih sulit diport ke Dart.
- **HOG `cell=4`** (bukan 8) supaya goresan nol bergaris miring tertangkap.
- **Deskew wajib.** Struk difoto miring 0.6–2.0°; tanpa deskew satu baris
  tersebar ke puluhan baris piksel dan `detect_lines` menggabungkan baris.
- **Ambang celah kata dari lebar sel huruf**, bukan median celah: pada
  `2 X 9.800` celah antar huruf 23 px dan celah kata 44 px.
- **Token angka tidak pernah dikoreksi leksikon**, hanya huruf yang bentuknya
  sama (`O`→`0`, `I`→`1`, `S`→`5`) dan hanya bila hasilnya berbentuk angka.
- **Ukuran kemasan bukan angka** (`250G`, `12S`, `200ML`).
- **Leksikon hanya cadangan**, dilewati bila classifier yakin (≥ 0.95).
- **Templat baris tetap** (alamat, footer, label) dengan penjaga tanda tangan
  angka, supaya alamat boleh diperbaiki tetapi harga tidak.

## Port Dart (aplikasi)

Pipeline yang sama hidup di `smart_receipt/lib/ocr/classical/` dan dipakai
lewat `ClassicalOcrEngine implements OcrEngine`. Model dibaca dari
`smart_receipt/assets/ocr/` (~1.3 MB total, jauh di bawah batas 20 MB).

| uji paritas | hasil |
|---|---|
| 200 golden vector | fitur selisih maks 5e-6, probabilitas < 1e-4, prediksi 200/200 |
| `receipts_v2` (10 struk, 138 baris) | CER 0.00%, exact 138/138 |
| held-out `synth_receipts` (16 struk) | CER 2.36%, exact 85.2% |

Jalankan `cd smart_receipt && flutter test` untuk uji paritas lengkap.

### Dua detail OpenCV yang wajib ditiru

Port ini sempat meleset jauh (CER 15.5%) bukan karena classifier, melainkan
karena dua hal di praproses:

- `bilateralFilter` memakai kernel **lingkaran** (offset dengan jarak >
  radius dibuang) dan tepi `BORDER_REFLECT_101`;
- rotasi deskew memakai `INTER_CUBIC`, bukan bilinear.

Selisih 1-2 level abu-abu saja cukup untuk memunculkan bintik biner yang
menjatuhkan median tinggi baris, lalu huruf terpecah dua.

## Format model

Semua model diekspor sebagai JSON agar bisa diport ke Dart tanpa konversi
biner. `golden_vectors.json` menyimpan vektor fitur + label untuk uji paritas
implementasi Dart terhadap Python.
