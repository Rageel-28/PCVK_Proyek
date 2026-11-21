# 13 — Glosarium & Peta Kode

## Glosarium

| Istilah | Arti |
|---|---|
| **OCR** | Optical Character Recognition — pengenalan teks dari citra |
| **Struk thermal** | Struk dari printer thermal minimarket; cetakan tipis, huruf sempit (aspek ~0,42) |
| **Sauvola** | Algoritma binarisasi adaptif lokal; ambang bergantung pada rata-rata & simpangan baku jendela |
| **Binarisasi** | Mengubah grayscale menjadi citra 2 nilai (tinta/kertas) |
| **Deskew** | Koreksi kemiringan citra agar baris teks horizontal |
| **Upscale adaptif** | Memperbesar citra sampai tinggi baris mencapai target (48 px), maksimal 5× |
| **Connected components** | Komponen piksel tinta yang saling tersambung (8-konektivitas) — dasar segmentasi glyph |
| **Pita baris (band)** | Deretan piksel horizontal yang berisi satu baris teks |
| **Glyph** | Satu karakter yang terlihat (hasil segmentasi) |
| **HOG** | Histogram of Oriented Gradients — fitur distribusi arah gradien |
| **Zoning** | Fitur per kotak grid (densitas & centroid) |
| **Topologi lubang** | Fitur jumlah & posisi lubang tertutup dalam glyph (`O`, `B`, `8`) |
| **LinearSVC** | Support Vector Classifier linear multikelas |
| **Temperature scaling** | Kalibrasi probabilitas dengan membagi skor dengan suhu sebelum softmax |
| **N-gram karakter** | Model bahasa atas barisan n karakter (di sini order 5) |
| **Beam search** | Pencarian terbatas yang menyimpan k terbaik di setiap langkah |
| **Leksikon** | Kamus kata sebagai cadangan koreksi |
| **Weighted Levenshtein** | Jarak edit dengan biaya substitusi berbeda per pasangan karakter |
| **Margin** | Selisih skor kelas teratas dan kedua; sinyal untuk memutuskan review |
| **TF-IDF** | Term frequency × inverse document frequency — bobot token |
| **Multinomial Naive Bayes** | Klasifikator probabilistik dengan asumsi independensi fitur |
| **`char_wb`** | Mode n-gram scikit-learn yang tidak melintasi batas kata |
| **Fixture** | Data uji tetap (foto struk + ground truth) |
| **Golden vector** | Vektor rujukan Python untuk menguji port Dart |
| **CER / WER** | Character / Word Error Rate |
| **Exact-line** | Persentase baris yang terbaca persis benar |
| **Isolate** | Unit konkurensi Dart dengan memori terpisah |
| **`needsRawImage`** | Flag mesin OCR yang meminta citra asli tanpa praproses aplikasi |
| **Paritas** | Sifat dua implementasi (Python & Dart) yang menghasilkan output sama |
| **Log-joint** | `class_log_prior + Σ tfidf·feature_log_prob` — skor gabungan Naive Bayes sebelum softmax |
| **Softmax** | Mengubah skor arbitrer menjadi distribusi probabilitas (`eˣ/Σeˣ`) |
| **Temperature scaling** | Membagi skor dengan konstanta suhu `T` sebelum softmax agar probabilitas terkalibrasi |
| **In-sample** | Metrik dievaluasi pada data yang sama dipakai melatih (bukan uji tersembunyi) |
| **R²** | Koefisien determinasi: proporsi variansi yang dijelaskan model (1 = sempurna) |
| **Fixture terkunci** | Data uji tetap yang tidak boleh dipakai menyetel parameter — jadi hasilnya **batas atas**, bukan akurasi produksi |
| **BR-08 / SKPL** | Aturan bisnis di dokumen SKPL (`Document/SKPL_SmartReceipt_Analyzer.docx`); bukan di `docs/` |
| **F0–F3** | Fase alur di `ML/README.md`: F0 dataset, F1 classifier, F2/F3 model bahasa/leksikon/templat |
| **Aspek huruf** | Perbandingan lebar/tinggi huruf; thermal ~0,42 vs font vektor ~0,85 |

## Peta Kode: Saya Ingin…

| Kebutuhan | Berkas utama |
|---|---|
| Memahami alur OCR dari awal | `lib/ocr/ocr_service.dart`, `lib/ocr/classical/pipeline.dart` |
| Mengubah parameter OCR | `lib/ocr/classical/params.dart` (dan `ML/ocr/config.py`) |
| Mengubah praproses citra OCR | `lib/ocr/classical/preprocess.dart` |
| Mengubah fitur karakter | `lib/ocr/classical/features.dart` |
| Mengubah classifier karakter | `lib/ocr/classical/char_classifier.dart` |
| Mengubah decoding/LM | `lib/ocr/classical/decode.dart`, `lm.dart`, `lexicon.dart` |
| Mengubah parser struk | `lib/ocr/receipt_parser.dart` |
| Mengubah koreksi pasca-OCR | `lib/ocr/ocr_text_corrector.dart` |
| Mengubah klasifikasi kategori | `lib/ocr/tfidf_classifier.dart` |
| Mengubah daftar kategori | `lib/models/categories.dart` |
| Mengubah praproses aplikasi | `lib/ocr/image_quality.dart` |
| Mengubah layar | `lib/screens/*.dart` |
| Mengubah penyimpanan | `lib/helpers/database_helper.dart` |
| Mengubah agregasi | `lib/services/spending_analytics.dart` |
| Mengubah prediksi | `lib/services/spending_forecaster.dart` |
| Melatih ulang model | `ML/ocr/train_char.py`, `export_char.py`, `export_lm.py` |
| Menjalankan evaluasi | `ML/ocr/evaluate.py` |

## Peta Kode: Komponen Utama

```
lib/
├── main.dart ...................... entry point
├── ocr/
│   ├── ocr_engine.dart ........... antarmuka OcrEngine
│   ├── ocr_service.dart .......... orkestrasi scan
│   ├── image_quality.dart ........ analisis & penyiapan citra
│   ├── ocr_text_corrector.dart ... koreksi karakter
│   ├── receipt_parser.dart ....... teks → ParsedReceipt
│   ├── tfidf_classifier.dart ..... klasifikasi kategori
│   ├── receipt_classifier.dart ... fallback klasifikasi
│   └── classical/
│       ├── params.dart ........... parameter (identik dgn Python)
│       ├── preprocess.dart ....... grayscale/denoise/Sauvola/upscale
│       ├── lines.dart, rules.dart  deteksi & filter baris
│       ├── words.dart, chars.dart  segmentasi
│       ├── char_image.dart ....... normalisasi 32×32
│       ├── features.dart ......... fitur 1844-d
│       ├── char_classifier.dart .. LinearSVC
│       ├── lm.dart, lexicon.dart . model bahasa & kamus
│       ├── decode.dart ........... beam search
│       ├── templates.dart ........ templat baris
│       ├── pipeline.dart ......... recognizePage
│       └── classical_ocr_engine.dart  OcrEngine + isolate
├── screens/ ...................... 13 berkas layar
├── services/ ..................... analitik, prediksi, evaluasi
├── providers/ .................... state (ReceiptProvider)
├── models/ ....................... model UI + DB + taksonomi
├── data/ ......................... adaptor DB ↔ UI
├── helpers/ ...................... SQLite
├── navigation/ ................... route & tab
├── theme/ ........................ palet & tema
└── widgets/ ...................... komponen UI
```

## Indeks Dokumentasi

| # | Dokumen |
|---|---|
| 01 | [Ringkasan Proyek](01-ringkasan-proyek.md) |
| 02 | [Arsitektur Aplikasi](02-arsitektur-aplikasi.md) |
| 03 | [Pipeline OCR Klasik](03-pipeline-ocr.md) |
| 04 | [Parser Struk & Koreksi Teks](04-parser-struk.md) |
| 05 | [Klasifikasi Kategori](05-klasifikasi-kategori.md) |
| 06 | [Evaluasi & Pengujian](06-evaluasi-pengujian.md) |
| 07 | [Build & Rilis](07-build-rilis.md) |
| 08 | [Layar & Lapisan UI](08-layar-ui.md) |
| 09 | [Analitik & Prediksi](09-analitik-prediksi.md) |
| 10 | [Praproses Citra Aplikasi](10-praproses-aplikasi.md) |
| 11 | [Metodologi Pelatihan](11-metodologi-pelatihan.md) |
| 12 | [Keputusan Desain](12-keputusan-desain.md) |
| 13 | Glosarium & Peta Kode (dokumen ini) |
| 14 | [Contoh Per Langkah: Struk Nyata & Fixture](14-contoh-struk.md) |
