# Laporan UTS PBL Pembelajaran Mesin

**SmartReceipt Analyzer — Klasifikasi Kategori Item Struk Belanja**

**Kelompok SmartReceipt Analyzer**

**Rakagali · Aldi · Rijal · Noval**

**21 Oktober 2026**

---

## Selayang Pandang

SmartReceipt Analyzer adalah aplikasi pencatatan pengeluaran yang memanfaatkan *Optical Character Recognition* (OCR) untuk membaca struk belanja, lalu mengklasifikasikan setiap item ke dalam kategori secara otomatis. Permasalahan utama yang dihadapi adalah teks hasil OCR sangat bising: nama produk terpotong, disingkat, salah baca, serta bercampur dengan nominal harga, ukuran kemasan, dan kode PLU. Solusi yang ditawarkan adalah model klasifikasi teks yang berjalan *offline* di perangkat (*on-device*), tanpa server dan tanpa koneksi internet, melalui tahapan pra-pengolahan teks, ekstraksi fitur TF-IDF *character n-gram*, dan algoritma pembelajaran mesin yang ringan. Pada UTS ini kelompok membandingkan empat model, yaitu Multinomial Naive Bayes, *k-Nearest Neighbors* (kNN), *Support Vector Machine* (SVM), dan *Random Forest*. Hasil terbaik pada data uji diperoleh kNN dengan akurasi 77,34% dan macro-F1 0,7365. Meskipun kNN juga paling cepat pada tahap pelatihan, keunggulan itu tidak relevan karena pelatihan hanya dilakukan sekali. Pada tahap inferensi yang berjalan di setiap pemindaian struk, kNN justru sekitar 109 kali lebih lambat daripada Naive Bayes dan harus menyimpan seluruh data latih di memori. Oleh sebab itu Multinomial Naive Bayes dipilih sebagai kandidat produksi: inferensinya 0,0010 milidetik per item, ukuran modelnya 3,0 MB, dan selisih akurasinya hanya 3,16 poin persentase dari kNN, sehingga model dapat diekspor menjadi berkas JSON dan dijalankan langsung di dalam aplikasi Flutter.

---

## Dataset

Data yang digunakan adalah data sintetis berbahasa Indonesia yang merepresentasikan teks item pada struk belanja. Terdapat dua berkas utama, yaitu `Dataset_SmartReceipt_Sintetis.csv` sebagai dataset utama dan `Dataset_Katalog_Indonesia_Sintetis_v1.csv` sebagai tambahan data latih dari katalog produk retail. Dataset utama memuat 6.026 baris, 1.419 produk unik yang tergabung dalam 1.027 grup produk, serta 15 kategori target. Dataset katalog memuat 1.190 baris dan 170 produk unik.

### Struktur Data

Setiap baris data merepresentasikan satu variasi penulisan dari sebuah produk. Kolom `receipt_text` berperan sebagai fitur masukan, sedangkan kolom `kategori` berperan sebagai label target. Kolom `nama_produk` dipakai untuk mencegah kebocoran data antarbagian, dan kolom `split` menandai pembagian data.

**Tabel 1. Kolom pada dataset utama**

| Kolom | Tipe | Deskripsi | Contoh Nilai |
|---|---|---|---|
| id | teks | Identitas baris | R000001 |
| product_id | teks | Identitas produk | P0001 |
| nama_produk | teks | Nama baku produk | Indomie Goreng |
| receipt_text | teks | Teks menyerupai hasil OCR, menjadi fitur masukan | INDOMIE GRG SPL 85G |
| kategori | teks | Label target kategori belanja | Makanan |
| gaya | teks | Gaya penulisan (singkatan, terpotong, dan sebagainya) | abbr_trunc |
| grup | teks | Kelompok produk sejenis | indomie |
| split | teks | Penanda train, val, atau test | train |
| sumber | teks | Asal data | sintetis |

Lima belas kategori target adalah Makanan, Minuman, Perawatan Diri, Rumah Tangga, Elektronik, Transportasi, Kesehatan, Pakaian & Fashion, Hiburan & Hobi, Bayi & Anak, Pendidikan & ATK, Tagihan & Utilitas, Hewan Peliharaan, Lainnya, dan Rokok & Tembakau.

**Tabel 2. Contoh baris data**

| receipt_text | kategori | gaya | split |
|---|---|---|---|
| Indomie Goreng 85 G | Makanan | clean | train |
| INDOMIE GRG | Makanan | abbr | train |
| INDOMIEGORENG | Makanan | nospace | train |
| INDOMIE AYAM BAWAN | Makanan | trunc | test |
| AQUA BTL 1500ML | Minuman | abbr | train |
| PARACETAMOL 500MG | Kesehatan | clean | train |
| 635110203830 INDOMIE AYAM BAWAN | Makanan | kode_plu | test |
| WHISKAS MAKANA | Hewan Peliharaan | trunc | test |
| GUDANG GARAM S | Rokok & Tembakau | abbr_trunc | test |
| PERTALITE | Transportasi | clean | train |

### Sebaran Kategori

Jumlah data tiap kategori tidak sama. Kategori Makanan memiliki 950 baris, sedangkan Rokok & Tembakau hanya 187 baris, sehingga kategori dengan data sedikit lebih sulit dikenali model.

**Tabel 3. Sebaran data per kategori pada dataset utama**

| Kategori | Jumlah Baris | Proporsi |
|---|---:|---:|
| Makanan | 950 | 15,8% |
| Elektronik | 560 | 9,3% |
| Perawatan Diri | 546 | 9,1% |
| Rumah Tangga | 501 | 8,3% |
| Transportasi | 451 | 7,5% |
| Minuman | 421 | 7,0% |
| Pakaian & Fashion | 366 | 6,1% |
| Kesehatan | 343 | 5,7% |
| Hiburan & Hobi | 332 | 5,5% |
| Tagihan & Utilitas | 299 | 5,0% |
| Bayi & Anak | 292 | 4,8% |
| Pendidikan & ATK | 282 | 4,7% |
| Hewan Peliharaan | 271 | 4,5% |
| Lainnya | 225 | 3,7% |
| Rokok & Tembakau | 187 | 3,1% |
| **Total** | **6.026** | **100,0%** |

### Karakteristik Teks

Teks pada dataset sengaja dibuat menyerupai keluaran OCR struk thermal yang tidak sempurna. Panjang teks berkisar antara 3 sampai 40 karakter dengan rata-rata 14,9 karakter, dan rata-rata 2,5 kata per baris. Variasi gaya penulisan mencakup sembilan pola, yaitu teks bersih (1.421 baris), singkatan (498), singkatan sekaligus terpotong (984), penghilangan kata (939), penggabungan tanpa spasi (733), penghilangan huruf vokal (529), pemotongan kata (716), penyisipan kode PLU (203), dan penulisan kapital (3).

### Pembagian Data

Dataset telah dilengkapi kolom `split` yang membagi data menjadi tiga bagian, yaitu data latih (train), data validasi (val), dan data uji (test). Pembagian ini bersifat tetap agar hasil evaluasi dapat direproduksi: 4.474 baris train (74,2%), 762 baris validasi (12,6%), dan 790 baris uji (13,1%).

---

## Pra Pengolahan

Seluruh anggota tim menggunakan fungsi pra-pengolahan yang sama, yaitu implementasi Python dari aturan aplikasi Flutter yang tersimpan pada berkas `smart_receipt/assets/text_normalization_id.json`. Kesamaan aturan ini penting agar distribusi teks yang dilihat model saat pelatihan identik dengan teks yang diterima model saat aplikasi dipakai. Pra-pengolahan dilakukan dalam tujuh tahap berurutan.

**Tahap 1 — Case folding.** Seluruh teks diubah menjadi huruf kecil agar penulisan kapital dan non-kapital dianggap sama.

```
INDOMIE GRG SPL 85G  →  indomie grg spl 85g
```

**Tahap 2 — Pembersihan nominal.** Pola harga seperti Rp 4.500, Rp4500, atau 120.000 dihapus karena tidak membawa informasi kategori.

```
aqua btl 600ml rp 4.500  →  aqua btl 600ml
```

**Tahap 3 — Pembersihan satuan dan kode.** Ukuran kemasan (85G, 600ML, 2PCS, 24S), kode PLU atau barcode, serta angka lepas dihapus.

```
635110203830 indomie ayam bawan  →  indomie ayam bawan
```

**Tahap 4 — Perapian huruf berulang.** Pengulangan karakter lebih dari dua kali diringkas menjadi dua, untuk mengatasi noise cetak thermal.

```
aquaaa  →  aquaa
```

**Tahap 5 — Pembersihan simbol.** Karakter non-alfanumerik dihapus dan spasi berlebih dirapikan.

```
indomie,,  goreng!!  →  indomie goreng
```

**Tahap 6 — Normalisasi merek dan sinonim.** Merek dipetakan ke kata generik agar model tidak perlu menghafal ribuan merek.

```
indomie → mie instan;  aqua → air mineral;  paracetamol → obat
```

**Tahap 7 — Pembuangan stopword dan token pendek.** Kata tanpa daya beda (dan, untuk, total, harga) serta token kurang dari dua karakter dibuang.

```
mie instan untuk total  →  mie instan
```

### Contoh Hasil

**Tabel 4. Contoh hasil pra-pengolahan**

| Teks Struk Asli | Hasil Pra-Pengolahan |
|---|---|
| INDOMIE GRG SPL 85G | mie instan grg spl |
| AQUA BTL 600ML Rp 4.500 | air mineral btl |
| PARACETAMOL 2PCS | obat |
| 635110203830 INDOMIE AYAM BAWAN | mie instan ayam bawan |
| PAMPERS NB 24S | popok nb |
| PERTAMAX 10LTR | bensin |
| TOKEN LISTRIK PLN 200000 | listrik |
| RINSO DETERJENBUBUK 800 GR | deterjen deterjenbubuk |
| WHISKAS MAKANA | whiskas makana |
| GUDANG GARAM S | rokok garam |

> **Catatan:** tidak digunakan augmentasi data pada eksperimen UTS. Variasi gaya teks (singkatan, pemotongan, penggabungan tanpa spasi, penghilangan vokal, kode PLU, dan noise OCR) sudah tersedia secara bawaan pada dataset sintetis, sehingga augmentasi tambahan tidak diperlukan agar perbandingan antarmodel tetap adil.

### Implementasi

Agar dapat diaudit, pra-pengolahan dipecah menjadi tiga fungsi terpisah pada implementasi Python, yaitu `clean()`, `tokenize()`, dan `normalize_tokens()`. Fungsi `normalize()` pada `pipeline_utils.py` menggabungkan ketiganya dan dipakai oleh seluruh notebook anggota tim.

---

## Ekstraksi Fitur

Ekstraksi fitur dikunci identik untuk keempat model agar perbandingan bersifat adil. Satu-satunya yang berbeda antareksperimen hanyalah algoritma klasifikasinya. Metode yang digunakan adalah *Term Frequency–Inverse Document Frequency* (TF-IDF) berbasis *character n-gram* dengan pembatas kata (`char_wb`).

### Alasan Pemilihan Character N-Gram

Pendekatan *character n-gram* dipilih karena teks struk sangat rentan terhadap kesalahan baca OCR. Sebagai contoh, kata "indomie" dapat terbaca "indome" atau "1nd0m1e". Apabila menggunakan pendekatan kata utuh (*word-level*), satu huruf yang salah menyebabkan kata tersebut tidak dikenali sama sekali. Dengan potongan karakter, sebagian besar potongan tetap sama sehingga kemiripan pola huruf masih dapat ditangkap model.

Contoh pemotongan `char_wb` pada kata "mie" (diberi padding spasi menjadi `" mie "`):

```
Potongan 2 karakter : " m"   "mi"   "ie"   "e "
Potongan 3 karakter : " mi"  "mie"  "ie "
Potongan 4 karakter : " mie" "mie "
```

Setiap kata menghasilkan 9 potongan. Seluruh potongan dari data latih dikumpulkan menjadi kosakata 13.170 fitur.

### Pembobotan TF-IDF

```
TF  = 1 + ln(jumlah kemunculan potongan dalam satu item)
IDF = ln( (1 + N) / (1 + df) ) + 1
N   = jumlah dokumen latih,  df = jumlah dokumen yang memuat potongan tersebut

Bobot TF-IDF = TF x IDF
Normalisasi L2: setiap nilai dibagi akar jumlah kuadrat seluruh nilai
```

Pembobotan sublinear TF dipakai agar potongan yang muncul sepuluh kali tidak dianggap sepuluh kali lebih penting dibandingkan potongan yang muncul sekali. Sementara itu, smooth IDF mencegah pembagian dengan nol dan menstabilkan bobot pada potongan yang sangat langka.

**Tabel 5. Konfigurasi TF-IDF yang dikunci untuk seluruh model**

| Parameter | Nilai | Keterangan |
|---|---|---|
| analyzer | char_wb | N-gram karakter dalam batas kata |
| ngram_range | (2, 4) | Panjang potongan 2 sampai 4 karakter |
| sublinear_tf | True | TF = 1 + ln(frekuensi) |
| norm | l2 | Normalisasi L2 pada setiap vektor |
| max_features | 20.000 | Batas maksimum jumlah fitur |
| Data untuk fit | Hanya data latih | Mencegah kebocoran informasi dari data uji |

### Hasil Ekstraksi

Dari data latih terbentuk matriks TF-IDF berukuran 5.569 baris × 13.170 kolom. Meskipun batas maksimum fitur ditetapkan 20.000, seluruh korpus hanya menghasilkan 13.170 potongan unik sehingga batas tersebut tidak tercapai. Matriks bersifat sangat jarang (*sparse*) dengan hanya 199.050 elemen tidak nol, atau setara kepadatan 0,271%. Nilai IDF bergerak antara 2,430 dan 8,932.

Sebagai gambaran konkret, untuk teks "mie instan grg spl" model hanya mengaktifkan 44 dari 13.170 fitur.

**Tabel 6. Contoh pembobotan TF-IDF untuk teks "mie instan grg spl"**

| Potongan | Indeks | IDF | TF-IDF | Tafsiran |
|---|---:|---:|---:|---|
| " spl" | 1.375 | 8,9320 | 0,2629 | Sangat langka, sangat khas |
| "pl " | 9.623 | 7,5457 | 0,2221 | Khas untuk kata berakhiran "pl" |
| "spl" | 11.228 | 7,5457 | 0,2221 | Ciri kuat kata "spl" |
| "grg " | 5.239 | 6,7348 | 0,1982 | Khas singkatan "grg" |
| " grg" | 562 | 6,6807 | 0,1966 | Khas singkatan "grg" |
| "mie " | 7.752 | 6,0988 | 0,1795 | Kata kunci kategori Makanan |
| " mie" | 925 | 6,0142 | 0,1770 | Kata kunci kategori Makanan |
| " m" | 887 | 2,7561 | 0,0811 | Terlalu umum, daya beda rendah |
| " s" | 1.261 | 2,4527 | 0,0722 | Terlalu umum, daya beda rendah |
| "an" | 2.238 | 2,4300 | 0,0715 | Paling umum di korpus |

> **Cara membaca tabel:** kolom IDF menunjukkan kelangkaan sebuah potongan di seluruh korpus. Nilai tinggi berarti potongan tersebut jarang muncul dan khas untuk kategori tertentu. Kolom TF-IDF menunjukkan pengaruh akhir potongan itu pada item yang sedang diperiksa. Jika seluruh 44 nilai dikuadratkan, dijumlahkan, lalu diakarkan, hasilnya tepat 1,0 sebagai bukti normalisasi L2 berhasil.

---

## Pembuatan Data Latih

Pembagian data memakai kolom `split` bawaan dataset dengan tambahan pengaman kebocoran data (*leakage guard*) berbasis produk. Produk yang muncul pada split validasi atau uji tidak diizinkan masuk ke data latih, termasuk apabila produk tersebut juga terdapat pada dataset katalog. Dengan demikian, evaluasi benar-benar mengukur kemampuan model mengklasifikasikan produk baru yang belum pernah dilihat, bukan sekadar menghafal baris data.

Data latih akhir merupakan gabungan dari 4.474 baris split train dataset utama dan 1.095 baris dataset katalog yang lolos penyaringan produk, sehingga totalnya 5.569 baris. Data validasi sebanyak 762 baris hanya dipakai untuk memantau performa, sedangkan data uji sebanyak 790 baris tidak disentuh sama sekali selama pelatihan maupun penyetelan model.

**Tabel 7. Perbandingan jumlah data latih dan data uji**

| Bagian Data | Sumber | Jumlah Baris | Proporsi | Keterangan |
|---|---|---:|---:|---|
| Data latih | Split train dataset utama | 4.474 | 62,6% | Dipakai untuk belajar |
| Data latih | Katalog tersaring | 1.095 | 15,3% | Produk val/test dibuang |
| **Data latih total** | **Gabungan** | **5.569** | **77,9%** | **Dipakai melatih model** |
| Data validasi | Split val dataset utama | 762 | 10,7% | Pemantauan performa |
| Data uji | Split test dataset utama | 790 | 11,1% | Evaluasi akhir |
| **Total** | — | **7.121** | **100%** | Termasuk data katalog |

Untuk menguji kestabilan hasil, dilakukan pula eksperimen tambahan dengan tiga rasio pembagian berbeda menggunakan `GroupShuffleSplit` agar kelompok produk tidak terpecah antarbagian, serta validasi silang lima lipatan berbasis kelompok (`StratifiedGroupKFold`).

**Tabel 8. Pengaruh rasio pembagian data terhadap performa Naive Bayes**

| Rasio / Metode | Jml. Data Latih | Jml. Data Uji | Akurasi | Macro-F1 |
|---|---:|---:|---:|---:|
| 70:30 | 4.249 | 1.777 | 75,75% | 0,7521 |
| 80:20 | 4.856 | 1.170 | 77,09% | 0,7489 |
| 90:10 | 5.441 | 585 | 76,92% | 0,7261 |
| 5-fold CV | ± 4.821 | ± 1.205 | 76,04% | 0,7539 |

Akurasi pada berbagai skema pembagian data berada pada rentang 75,75% sampai 77,09% dengan rata-rata validasi silang 76,04% dan simpangan baku 2,56 poin. Kestabilan ini menunjukkan bahwa hasil evaluasi tidak bergantung pada satu pembagian data tertentu. Angka akurasi pada Tabel 8 sedikit lebih tinggi daripada hasil pada Tabel 10 karena skema ini melatih model pada seluruh dataset utama tanpa penyaringan katalog, sehingga model berpeluang melihat lebih banyak variasi produk.

---

## Pembuatan Model

Empat anggota tim membuat satu model klasifikasi berbeda di atas fitur TF-IDF yang sama. Seluruh model dilatih pada data latih yang sama dan diuji pada 790 baris data uji yang sama, sehingga perbandingan mencerminkan perbedaan algoritma, bukan perbedaan data.

**Tabel 9. Hyperparameter keempat model**

| Anggota | Model | Hyperparameter Utama | Alasan Pemilihan |
|---|---|---|---|
| Rakagali | Multinomial Naive Bayes | alpha = 0,03; fit_prior = True | Cepat, ringan, mudah diekspor |
| Aldi | k-Nearest Neighbors | n_neighbors = 5; weights = distance; metric = cosine | Sederhana, cocok untuk kemiripan teks |
| Rijal | Support Vector Machine | LinearSVC; C = 1,0; class_weight = balanced; max_iter = 5.000 | Efektif pada data berdimensi tinggi |
| Noval | Random Forest | n_estimators = 300; max_features = sqrt; class_weight = balanced_subsample | Tahan terhadap outlier dan fitur berlebih |

### Multinomial Naive Bayes

Naive Bayes menghitung skor setiap kategori sebagai jumlah log prior kelas dengan total perkalian antara nilai TF-IDF dan log likelihood fitur. Model memilih kategori dengan skor tertinggi. Meskipun mengasumsikan setiap fitur saling mandiri, asumsi ini membuat perhitungan menjadi sangat ringan dan tetap akurat untuk klasifikasi teks.

```
skor(kategori) = log P(kategori) + Σ [ TF-IDF_i × log P(potongan_i | kategori) ]

kategori terpilih = argmax skor(kategori)
```

Parameter alpha = 0,03 berperan sebagai penghalus Laplace. Tanpa penghalus, potongan yang belum pernah muncul pada suatu kategori saat pelatihan akan bernilai nol dan menyebabkan seluruh skor kategori tersebut runtuh hanya karena satu potongan. Nilai alpha yang kecil dipilih agar penghalusan cukup mencegah nilai nol tanpa mengubah distribusi hasil secara berlebihan.

### k-Nearest Neighbors

kNN tidak membangun model pada tahap pelatihan; ia hanya menyimpan seluruh data latih. Saat prediksi, jarak antara item baru dan seluruh item latih dihitung menggunakan kemiripan kosinus, lalu lima tetangga terdekat dipilih. Kategori ditentukan melalui suara terbanyak dengan pembobotan jarak, sehingga tetangga yang lebih dekat memiliki pengaruh lebih besar.

### Support Vector Machine

SVM linear mencari bidang pemisah (*hyperplane*) yang memisahkan kategori dengan margin maksimum. Parameter C = 1,0 mengatur keseimbangan antara lebar margin dan toleransi kesalahan, sedangkan `class_weight = balanced` dipakai untuk meredam ketidakseimbangan jumlah data antarkategori.

### Random Forest

Random Forest membangun 300 pohon keputusan, masing-masing dilatih pada sampel data bootstrap dan subset fitur acak. Prediksi akhir ditentukan melalui voting mayoritas antar pohon. Pendekatan ini tahan terhadap outlier, tetapi kurang sesuai untuk vektor TF-IDF yang renggang dan berdimensi tinggi karena waktu pelatihannya jauh lebih lama.

### Mekanisme Penolakan pada Tahap Produksi

Model Naive Bayes yang dipilih untuk produksi dilengkapi mekanisme penolakan berbasis margin. Model menghitung selisih skor antara peringkat pertama dan peringkat kedua; apabila selisih tersebut di bawah ambang 4,0, prediksi ditandai sebagai **Perlu Review** agar pengguna dapat mengonfirmasi atau mengoreksi kategorinya. Mekanisme ini penting karena probabilitas mentah Naive Bayes pada data teks cenderung mendekati 100% sehingga tidak informatif untuk menentukan tingkat keyakinan.

Sebagai gambaran, untuk teks "mie instan ayam bawan" model menghasilkan skor Makanan −48,8521 dan skor runner-up Pakaian & Fashion −63,1947, sehingga marginnya 14,3427 dan prediksi langsung diterima. Sebaliknya, untuk teks "minyak tropical" skor Kesehatan −49,0142 hanya sedikit lebih tinggi daripada Makanan −49,1830, dengan margin 0,1688, sehingga prediksi ditandai Perlu Review. Pada kasus kedua tersebut model memang salah memprediksi, dan mekanisme penolakan berhasil mencegah kesalahan masuk ke laporan pengeluaran pengguna.

---

## Evaluasi

Evaluasi dilakukan pada 790 baris data uji yang tidak pernah dilihat model selama pelatihan. Metrik yang digunakan adalah akurasi, presisi, recall, dan F1-score. Presisi, recall, dan F1 dihitung dengan rata-rata *macro* agar setiap kategori berbobot setara tanpa memandang jumlah sampelnya, sehingga kategori kecil seperti Rokok & Tembakau tidak tenggelam oleh kategori besar seperti Makanan. Selain itu dilaporkan pula F1 *weighted* yang memperhitungkan jumlah sampel tiap kategori.

**Tabel 10. Hasil evaluasi pada data uji (790 baris)**

| Model | Akurasi | Presisi (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|---|---:|---:|---:|---:|---:|
| k-Nearest Neighbors (kNN) | 77,34% | 0,7387 | 0,7563 | 0,7365 | 0,7728 |
| Support Vector Machine (SVM) | 74,68% | 0,7069 | 0,7262 | 0,7050 | 0,7417 |
| **Multinomial Naive Bayes** | **74,18%** | **0,7087** | **0,7213** | **0,7063** | **0,7444** |
| Random Forest | 72,91% | 0,7266 | 0,6899 | 0,6946 | 0,7180 |

Akurasi dan F1 hanya bercerita tentang kualitas prediksi. Untuk memilih model yang dijalankan di ponsel, biaya komputasi juga harus diukur.

**Tabel 11. Biaya komputasi dan ukuran model**

| Model | Waktu Latih | Waktu Inferensi | Ukuran Model | Catatan |
|---|---:|---:|---:|---|
| Multinomial Naive Bayes | 0,01 s | 0,0010 ms/item | 3,0 MB | Hanya menyimpan tabel bobot |
| k-Nearest Neighbors (kNN) | 0,005 s | 0,1066 ms/item | 2,3 MB | Harus menyimpan seluruh data latih |
| Support Vector Machine (SVM) | 0,17 s | 0,0007 ms/item | 1,5 MB | Paling cepat saat inferensi |
| Random Forest | 3,73 s | 0,0884 ms/item | 160,6 MB | 300 pohon, terlalu besar untuk ponsel |

> **Cara pengukuran:** waktu pelatihan dan inferensi diukur pada perangkat pengujian yang sama dengan konfigurasi hyperparameter identik seperti pada `perbandingan_model.ipynb`. Setiap pengukuran diulang tiga kali untuk pelatihan dan tujuh kali untuk inferensi, lalu diambil nilai median agar tidak terpengaruh fluktuasi sistem. Ukuran model diukur dari berkas hasil serialisasi, di mana kNN membutuhkan seluruh matriks data latih sehingga ukurannya tidak dapat diperkecil.

Tabel 11 menunjukkan hal yang penting dan sering terlewat: kNN memang paling cepat pada tahap pelatihan, yaitu sekitar 0,005 detik, karena kNN tidak benar-benar belajar apa pun. Ia hanya menyalin seluruh data latih ke dalam memori. Akan tetapi, pada tahap inferensi, kNN harus menghitung kemiripan antara item baru dan seluruh 5.569 baris data latih, sehingga waktunya menjadi 0,1066 milidetik per item, atau sekitar 109 kali lebih lambat daripada Naive Bayes.

Perbedaan ini menjadi krusial karena konteks pemakaiannya berbeda. Pelatihan dilakukan sekali saja di komputer, sedangkan inferensi terjadi pada setiap item struk yang dipindai pengguna. Satu struk berisi rata-rata 10 sampai 20 item, sehingga selisih 0,106 milidetik per item berlipat menjadi sekitar 2,1 milidetik per struk. Angka itu masih kecil, tetapi kNN juga menuntut seluruh matriks data latih tetap berada di memori perangkat, yang berarti konsumsi RAM lebih besar dan berkas model tidak dapat dikecilkan menjadi sekadar tabel bobot.

Dengan demikian, keunggulan kNN pada kolom waktu latih tidak dapat dijadikan alasan untuk memilihnya. Kriteria yang relevan bagi aplikasi bergerak adalah waktu inferensi, ukuran model, dan konsumsi memori; pada ketiga kriteria tersebut Naive Bayes atau SVM lebih unggul. Random Forest gugur dengan jelas karena ukurannya mencapai 160,6 MB dan inferensinya juga lebih lambat daripada Naive Bayes.

kNN memperoleh akurasi tertinggi pada data uji, yaitu 77,34%, diikuti SVM 74,68%, Multinomial Naive Bayes 74,18%, dan Random Forest 72,91%. Meskipun demikian, urutan tersebut berubah pada data validasi.

**Tabel 12. Hasil evaluasi pada data validasi (762 baris)**

| Model | Akurasi Validasi | Macro-F1 Validasi |
|---|---:|---:|
| Support Vector Machine (SVM) | 86,09% | 0,8525 |
| Multinomial Naive Bayes | 82,41% | 0,8180 |
| k-Nearest Neighbors (kNN) | 80,18% | 0,8022 |
| Random Forest | 77,30% | 0,7737 |

Perbedaan urutan ini menarik untuk dicermati. SVM paling baik pada data validasi (86,09%), tetapi turun ke peringkat kedua pada data uji. Sebaliknya, kNN justru memimpin pada data uji. Pola tersebut mengindikasikan bahwa SVM lebih stabil ketika produk uji masih mirip dengan data latih, sedangkan kNN lebih baik dalam mengenali produk yang benar-benar baru. Naive Bayes sendiri menempati posisi kedua pada data validasi dan ketiga pada data uji, dengan selisih yang relatif kecil di kedua skenario, sehingga dapat dikatakan paling konsisten.

### Analisis Kesalahan Multinomial Naive Bayes

**Tabel 13. Performa per kategori Multinomial Naive Bayes pada data uji**

| Kategori | Presisi | Recall | F1 | Jumlah Uji | Catatan |
|---|---:|---:|---:|---:|---|
| Makanan | 0,9245 | 0,8596 | 0,8909 | 171 | Paling kuat, data terbanyak |
| Pakaian & Fashion | 0,9048 | 0,8636 | 0,8837 | 44 | Kosakata sangat khas |
| Transportasi | 0,8197 | 0,8475 | 0,8333 | 59 | Kata kunci jelas |
| Hewan Peliharaan | 0,8684 | 0,7857 | 0,8250 | 42 | Cukup baik |
| Perawatan Diri | 0,8235 | 0,8235 | 0,8235 | 51 | Seimbang |
| Hiburan & Hobi | 0,8947 | 0,7234 | 0,8000 | 47 | Recall perlu ditingkatkan |
| Elektronik | 0,7778 | 0,8182 | 0,7975 | 77 | Baik |
| Tagihan & Utilitas | 0,6905 | 0,9355 | 0,7945 | 31 | Recall sangat tinggi |
| Rokok & Tembakau | 0,7727 | 0,7391 | 0,7556 | 23 | Data sedikit |
| Lainnya | 0,6829 | 0,7778 | 0,7273 | 36 | Kategori sisa, wajar lemah |
| Bayi & Anak | 0,4902 | 0,8065 | 0,6098 | 31 | Sering tertukar dengan Perawatan Diri |
| Pendidikan & ATK | 0,5882 | 0,5000 | 0,5405 | 40 | Tumpang tindih dengan Lainnya |
| Rumah Tangga | 0,6591 | 0,4143 | 0,5088 | 70 | Recall paling rendah |
| Minuman | 0,4359 | 0,4250 | 0,4304 | 40 | Sering tertukar dengan Makanan |
| Kesehatan | 0,2979 | 0,5000 | 0,3733 | 28 | Paling lemah |
| **Rata-rata macro** | **0,7087** | **0,7213** | **0,7063** | **790** | — |

Tiga kategori dengan performa terbaik adalah Makanan (F1 0,8909), Pakaian & Fashion (0,8837), dan Transportasi (0,8333). Ketiganya memiliki kosakata yang sangat khas sehingga mudah dipisahkan. Sebaliknya, Kesehatan (0,3733), Minuman (0,4304), dan Rumah Tangga (0,5088) menjadi kategori terlemah.

Kategori Kesehatan paling sering tertukar karena kosakatanya sangat pendek dan tumpang tindih dengan kategori lain; kata seperti "obat" atau "vitamin" juga lazim muncul pada item Perawatan Diri. Kategori Minuman sering tertukar dengan Makanan karena banyak minuman kemasan memiliki kata kunci yang mirip, misalnya "teh" atau "susu". Sementara itu, kategori Rumah Tangga memiliki recall hanya 0,4143 karena banyak produknya yang berupa nama merek yang telah dinormalisasi menjadi kata generik, sehingga kehilangan daya pembeda terhadap kategori lain.

Pola kesalahan lain yang teramati adalah produk yang sebenarnya sudah dikenal, tetapi teksnya terpotong atau mengandung singkatan yang tidak terdapat pada data latih. Sebagai contoh, "MINYAK TROPICAL" diprediksi sebagai Kesehatan, bukan Makanan, karena kata "tropical" lebih sering muncul pada konteks minyak kesehatan pada data latih. Kesalahan seperti ini justru tertangkap oleh mekanisme Perlu Review karena marginnya sangat kecil.

### Perbandingan dengan Penelitian Sejenis

Hasil ini sejalan dengan temuan umum pada klasifikasi teks pendek berbahasa Indonesia. TF-IDF character n-gram terbukti mampu menangani noise OCR dengan baik, sedangkan Naive Bayes konsisten menjadi pilihan yang paling efisien secara komputasi meskipun bukan yang paling akurat. Keunggulan kNN pada data uji juga masuk akal karena sifatnya yang berbasis kemiripan lokal, sehingga cocok untuk produk baru yang masih memiliki tetangga dekat pada data latih.

---

## Kesimpulan

1. **Keseragaman pipeline.** Pipeline TF-IDF character n-gram yang sama dapat digunakan oleh empat algoritma berbeda dan menghasilkan performa yang kompetitif, dengan rentang akurasi 72,91% sampai 77,34% pada data uji.
2. **Performa antarmodel.** kNN unggul pada data uji dengan akurasi 77,34% dan macro-F1 0,7365, sedangkan SVM unggul pada data validasi dengan akurasi 86,09%. Perbedaan ini menunjukkan bahwa pemilihan model sebaiknya mempertimbangkan jenis data yang akan dihadapi di lapangan.
3. **Keunggulan waktu latih kNN tidak menentukan.** kNN memang paling cepat pada tahap pelatihan (sekitar 0,005 detik) karena hanya menyalin data latih tanpa belajar apa pun. Namun keunggulan tersebut tidak relevan karena pelatihan hanya sekali. Pada tahap inferensi, kNN memerlukan 0,1066 milidetik per item, yaitu sekitar 109 kali lebih lambat daripada Naive Bayes, dan tetap harus menyimpan seluruh data latih di memori.
4. **Model terpilih.** Multinomial Naive Bayes dipilih sebagai model produksi karena inferensinya 0,0010 milidetik per item, ukuran modelnya 3,0 MB, dan bobotnya dapat diekspor menjadi berkas JSON yang berjalan langsung di perangkat tanpa koneksi internet. Akurasinya 74,18%, hanya terpaut 3,16 poin persentase dari kNN.
5. **Model yang kurang sesuai.** Random Forest kurang sesuai untuk vektor TF-IDF yang renggang dan berdimensi tinggi. Waktu latihnya 3,73 detik, inferensinya 0,0884 milidetik per item, ukuran modelnya mencapai 160,6 MB sehingga tidak layak dibundel ke aplikasi ponsel, dan recall-nya terendah (0,6899).
6. **Mitigasi kesalahan.** Mekanisme Perlu Review berbasis margin terbukti efektif menangkap prediksi yang tidak dapat dipercaya. Pada contoh nyata, prediksi yang salah memiliki margin hanya 0,1688 sehingga berhasil ditandai untuk dikonfirmasi pengguna.
7. **Titik lemah.** Kategori Kesehatan, Minuman, dan Rumah Tangga masih menjadi titik lemah dengan F1 di bawah 0,52. Pengembangan berikutnya perlu menambah data dan memperkaya aturan normalisasi untuk ketiga kategori tersebut.
8. **Saran pengembangan.** Pengembangan lanjutan yang disarankan meliputi penyetelan hyperparameter pada data validasi, penambahan aturan normalisasi singkatan kasir, penambahan data struk asli, serta eksperimen pembobotan kelas untuk kategori minoritas.

---

## Kontribusi Anggota Tim

Seluruh anggota terlibat dalam verifikasi pembagian data, peninjauan hasil evaluasi, dan penyusunan kesimpulan.

**Tabel 14. Kontribusi anggota tim**

| Nama Anggota | Kontribusi |
|---|---|
| Rakagali | Menyusun pipeline bersama (pra-pengolahan, TF-IDF, pembagian data dengan leakage guard), membuat model Multinomial Naive Bayes dalam versi scikit-learn dan versi manual dari nol, menyusun notebook perbandingan model, serta menyusun laporan. |
| Aldi | Membuat dan mengevaluasi model k-Nearest Neighbors, menganalisis pengaruh nilai k dan metrik cosine terhadap akurasi, serta menyusun analisis kesalahan kNN. |
| Rijal | Membuat dan mengevaluasi model Support Vector Machine, menganalisis pengaruh hyperparameter C dan class_weight, serta membandingkan performa SVM pada data uji dan validasi. |
| Noval | Membuat dan mengevaluasi model Random Forest, menganalisis pengaruh jumlah pohon dan perilakunya pada data teks renggang berdimensi tinggi. |
| Semua anggota | Memverifikasi pembagian data dan leakage guard, mereview hasil evaluasi dan confusion matrix, serta menyusun kesimpulan bersama. |

---

## Lampiran: Implementasi di Aplikasi

Sebagai bukti bahwa hasil penelitian dapat diterapkan, model Naive Bayes diekspor menjadi berkas JSON yang memuat daftar kelas, kosakata, bobot IDF, log prior, dan matriks log likelihood. Aplikasi Flutter kemudian memuat berkas tersebut dan menjalankan vektorisasi serta perhitungan Naive Bayes secara native di perangkat.

```
Foto struk → OCR → Pra-pengolahan → Vektorisasi TF-IDF
    → Skor Naive Bayes (15 kategori) → Uji margin (4,0)
    → Tampilkan kategori / tandai Perlu Review
```

Terdapat dua hal yang perlu dicatat pada tahap implementasi. Pertama, pra-pengolahan di aplikasi harus identik dengan pra-pengolahan saat pelatihan agar distribusi fitur yang diterima model konsisten. Kedua, aplikasi memiliki jalur cadangan berupa pengklasifikasi berbasis daftar kata kunci apabila berkas model gagal dimuat, sehingga aplikasi tetap dapat digunakan meskipun dengan akurasi yang lebih rendah.
