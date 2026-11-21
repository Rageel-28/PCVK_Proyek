# 05 — Klasifikasi Kategori

Setelah item struk diekstrak, setiap nama item diklasifikasikan ke salah
satu dari **9 kategori** pengeluaran. Klasifikasi berjalan sepenuhnya
di perangkat (on-device), memakai model **TF-IDF + Multinomial Naive
Bayes** yang dilatih offline dan diekspor ke JSON.

## Taksonomi: Tiga Lapis yang Disatukan

Sebelumnya proyek membawa tiga daftar kategori yang saling bertentangan:

| Sumber | Jumlah | Contoh |
|---|---|---|
| Seed SQLite lama | 8 | `Makanan & Minuman`, `Transport` |
| Daftar UI lama | 9 | `Makanan`, `Transportasi` |
| Label model ML | 15 | `Rokok & Tembakau` |

Id diterjemahkan **berdasarkan posisi** antar-daftar, sehingga model
`Transportasi` bisa diam-diam terpetakan ke UI `Minuman`. Sekarang semua
merujuk satu sumber: `lib/models/categories.dart`.

### 9 Kategori Aplikasi (`kAppCategories`)

| Id | Nama | Ikon | Warna |
|---|---|---|---|
| 1 | Makanan | 🍔 | `#FF6B6B` |
| 2 | Minuman | 🥤 | `#4ECDC4` |
| 3 | Rumah Tangga | 🛒 | `#45B7D1` |
| 4 | Perawatan Diri | 🧴 | `#B45BCB` |
| 5 | Kesehatan | ⚕️ | `#FF7675` |
| 6 | Transportasi | 🚗 | `#FFA502` |
| 7 | Elektronik | 💻 | `#74B9FF` |
| 8 | Pakaian | 👕 | `#DB6B8C` |
| 9 | Lainnya | 📌 | `#6C5CE7` |

`kFallbackCategoryId = 9` (`Lainnya`) dipakai untuk label model yang
tidak punya kategori khusus, termasuk fallback "Perlu Review".

## Model: TF-IDF + Naive Bayes (`tfidf_classifier.dart`)

Model dilatih dengan scikit-learn-style (`TfidfVectorizer` char n-gram +
`MultinomialNB`) lalu diekspor ke `assets/model_kategori_produk.json`
(~3 MB). `TfidfNaiveBayesEngine` hanya **menjalankan** model, mencerminkan
transformasi saat latih secara identik agar skor reprodusibel:

1. **Ekstraksi n-gram karakter** (2–4 gram, word-boundary aware seperti
   `char_wb` scikit-learn)
2. **TF sublinear** → `1 + ln(tf)`
3. **Pembobotan IDF** → `tf × idf`
4. **Normalisasi L2**
5. **Multinomial NB** → `class_log_prior + Σ tfidf·feature_log_prob`
6. **Softmax** atas skor kelas untuk posterior

### Keputusan Review: *Margin*, Bukan Posterior

Posterior mentah model ini **jenuh mendekati 1,000** (median 1,000),
sehingga tidak bisa dipakai untuk memutuskan apakah prediksi perlu
ditinjau. Sinyal yang dipakai adalah **margin** = selisih skor
log-joint kelas teratas dan kelas kedua (`s1 − s2`); karena vektor TF-IDF
dinormalisasi L2, margin ini kira-kira bebas skala.

- `defaultReviewMargin = 4,0`
- Kalibrasi pada set evaluasi dummy: margin < 4 menandai ~12% item,
  yang memiliki **error rate ~22%**, vs ~1,6% untuk item yang tidak
  ditandai.
- Nilai `confidence` yang ditampilkan UI diturunkan dari margin sebagai
  batas bawah, jadi angka di layar selalu konsisten dengan status
  "perlu review".

### Pemetaan 15 Kelas Model → 9 Kategori

`TfidfNaiveBayesEngine.classToCategoryId` memetakan label model
(mis. `Makanan`→1, `Tagihan & Utilitas`→9, `Rokok & Tembakau`→9) —
berdasarkan **nama**, bukan posisi indeks.

## Alur di Layar Klasifikasi

`CategoryClassificationScreen`:

1. `MLClassifierService.loadModel()` memuat model + tabel normalisasi
   teks (`text_normalization_id.json`); tidak pernah melempar — bila
   gagal, klasifikasi jatuh ke fallback.
2. Untuk setiap item, nama **diprediksi dua kali**: teks mentah dan hasil
   `OcrAbbreviationExpander.expand()` (mis. `GG`→`GUDANG GARAM`,
   `S/ROTI`→`SARI ROTI`). Prediksi dengan **margin lebih besar**
   dipertahankan.
3. Hasil ditampilkan; item dengan `needsReview` ditandai agar
   dikonfirmasi di layar review.

## Fallback Berbasis Aturan (`receipt_classifier.dart`)

`ReceiptClassifier` adalah klasifier leksikon kecil yang eksplisit,
hanya dipakai bila model TF-IDF tidak bisa dimuat. Skoring: bobot per
kata (mis. `indomie` 3 untuk Makanan), dicocokkan dengan tokenisasi
(potongan prefix untuk OCR terpotong), diaktifkan softmax suhu 1,5.
Confidence-nya heuristik, bukan terkalibrasi — model utama selalu
diutamakan.

## Umpan Balik & Retraining

Aplikasi memiliki siklus perbaikan model:

1. Pengguna mengubah kategori item di layar review/detail →
   `CorrectionTrainingService.recordCorrection` menyimpan ke tabel
   `category_corrections`.
2. `buildTrainingDataset()` / `exportTrainingDataset()` mengekspor koreksi
   dalam format `{samples: [{text, label}]}` — sama dengan
   `eval_dataset_dummy.json`.
3. Di luar aplikasi: dataset digabung, model dilatih ulang, JSON
   diekspor ulang.
4. `ModelEvaluator` menghitung akurasi, presisi, recall, dan F1 (macro &
   weighted) per kelas untuk verifikasi.

`SettingsScreen` menampilkan jumlah koreksi dan status "Latih Ulang
Model v1.0" agar siklus ini transparan bagi pengguna.
