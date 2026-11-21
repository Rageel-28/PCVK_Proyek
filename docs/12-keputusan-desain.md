# 12 — Keputusan Desain

Dokumen ini mencatat keputusan arsitektur yang **menentukan** bentuk
proyek, alasan di baliknya, dan konsekuensinya. Berguna untuk menilai
proyek maupun untuk siapa pun yang melanjutkannya.

---

## KD-01 — OCR dibangun dari nol (tanpa library OCR, tanpa deep learning)

**Keputusan.** Seluruh pengenalan teks memakai metode klasik yang
ditulis sendiri: binarisasi Sauvola, segmentasi connected components,
HOG + LinearSVC, n-gram LM, dan beam search.

**Alasan.** Ini tuntutan akademik mata kuliah Pengolahan Citra — setiap
tahap harus bisa dijelaskan dan dipertanggungjawabkan, bukan kotak hitam
pustaka.

**Konsekuensi.** Akurasi lebih rendah daripada OCR komersial pada foto
sulit, dan seluruh beban pemeliharaan ada di proyek. Ditukar dengan
kontrol penuh dan penjelasan tiap tahap.

---

## KD-02 — Satu model linear, bukan ensemble

**Keputusan.** Classifier karakter memakai satu LinearSVC, bukan
soft-voting beberapa model.

**Alasan.** Diukur pada data validasi: LinearSVC **99,882%**,
LogisticRegression 99,861%, soft-voting keduanya 99,865%. Ensemble tidak
menambah akurasi sama sekali, tetapi menambah ukuran model dan waktu
inferensi di perangkat — serta jauh lebih sulit diport ke Dart.

**Konsekuensi.** Model tunggal tanpa redundansi; jika kelas tertentu
lemah, perbaikannya lewat data (booster kelas langka), bukan ensemble.

---

## KD-03 — Port Dart 1:1 dan uji paritas

**Keputusan.** Pipeline produksi adalah port Dart dari rujukan Python,
dengan parameter terpusat dan uji vektor emas.

**Alasan.** Python bagus untuk pelatihan & eksperimen, tidak bisa
dijalankan di ponsel. Menjaga keduanya identik membuat hasil eksperimen
Python dapat dipercaya sebagai rujukan perilaku aplikasi.

**Konsekuensi.** Setiap perubahan algoritma harus dilakukan **dua kali**
(Python & Dart). Uji paritas (`ocr_golden_test`) menjadi gerbang wajib.
Selisih floating-point masih bisa muncul di batas tertentu.

---

## KD-04 — Konvensi biner: 255 = tinta, 0 = kertas

**Keputusan.** Di seluruh pipeline klasik, piksel bernilai 255 berarti
tinta.

**Alasan.** Kebalikan dari banyak pustaka gambar; konvensi ini membuat
perhitungan "jumlah tinta" menjadi penjumlahan sederhana.

**Konsekuensi.** Membingungkan bagi pendatang baru — karena itu
didokumentasikan di doc comment `gray_image.dart` dan di
[03](03-pipeline-ocr.md).

---

## KD-05 — Mesin OCR sebagai antarmuka + `needsRawImage`

**Keputusan.** `OcrEngine` adalah antarmuka; mesin dipasang via
`OcrService(engine: ...)`. Mesin dapat menyatakan bahwa ia butuh citra
**asli**.

**Alasan.** Aplikasi tidak boleh bergantung pada satu implementasi
(evergreen untuk eksperimen mesin lain). Flag `needsRawImage`
menyelesaikan masalah nyata: pipeline klasik melakukan deskew, Sauvola,
dan upscale sendiri; memberinya gambar yang sudah diputar, di-unsharp,
dan di-JPEG ulang **menghilangkan goresan tipis** cetakan thermal —
gejalanya derau seperti `220M-....` dan `ROT1`.

**Konsekuensi.** `OcrService` tidak punya mesin default; pembuatan tanpa
engine memunculkan `OcrFailure("Mesin OCR belum tersedia.")`. Strategi
pemindaian ulang hanya aktif untuk mesin non-raw.

---

## KD-06 — OCR dijalankan di isolate terpisah

**Keputusan.** `ClassicalOcrEngine.recognize` menjalankan pipeline di
`Isolate.run`.

**Alasan.** OCR satu struk memakan puluhan detik. Di thread UI itu
memicu dialog "aplikasi tidak merespons" di Android.

**Konsekuensi.** Model harus dikirim sebagai teks JSON dan di-parse
ulang di dalam isolate (ratusan milidetik — jauh di bawah waktu OCR),
dan closure tidak boleh menangkap `this`.

---

## KD-07 — Total tercetak sebagai kebenaran; rekonsiliasi aritmetika

**Keputusan.** Parser menyimpan semua pembacaan angka yang plausibel
per item, lalu memilih kombinasi yang jumlahnya cocok dengan total
tercetak.

**Alasan.** Urutan kolom berbeda antarprinter (`qty unit total` vs
`total qty unit`). Tanpa rekonsiliasi, harga mudah tertempel pada item
yang salah.

**Konsekuensi.** Parser lebih kompleks, tetapi skor kecocokan total juga
menjadi sinyal kuat untuk memilih di antara strategi pemindaian ulang.

---

## KD-08 — Pencocokan label dengan jarak edit, bukan regex persis

**Keputusan.** Label seperti `TOTAL`, `DPP`, `PPN`, `TUNAI` dicocokkan
dengan Levenshtein ≤ 2 plus pembatasan huruf awal.

**Alasan.** OCR sering merusak label karena hurufnya tebal dan kecil:
`DPP`→`D+DP`, `PPN`→`P&DLO`, `TUNAI`→`TKWNAI`, `TOTAL`→`TKDTAL`.

**Konsekuensi.** Ambang harus dijaga agar nama produk tidak ikut
terfilter (`BOTOL` berjarak edit 2 dari `TOTAL`; `TEH` dari `TAX`) —
karena itu huruf awal dan panjang kata ikut diperiksa.

---

## KD-09 — Satu taksonomi kategori

**Keputusan.** `lib/models/categories.dart` menjadi satu-satunya sumber
9 kategori; UI, seed SQLite, dan pemetaan 15 kelas model merujuk ke sana.

**Alasan.** Tiga daftar lama (8, 9, 15 entri) diterjemahkan **berdasarkan
posisi**, sehingga model `Transportasi` bisa terpetakan ke UI `Minuman`
— bug diam-diam.

**Konsekuensi.** Perlu migrasi v2→v3 untuk memetakan ulang id legacy.

---

## KD-10 — Keputusan review memakai *margin*, bukan posterior

**Keputusan.** Item ditandai "perlu review" bila selisih skor kelas
teratas dan kedua < 4,0.

**Alasan.** Posterior model jenuh di ~1,000, jadi angka itu tidak
informatif. Margin < 4 berkorelasi dengan **error rate ~22%**, vs ~1,6%
untuk yang tidak ditandai.

**Konsekuensi.** Ambang 4,0 terikat pada model saat ini; model baru
memerlukan kalibrasi ulang.

---

## KD-11 — Foto struk disalin ke dokumen aplikasi

**Keputusan.** `ReceiptImageStore.persist()` menyalin foto ke
`documents/receipt_images/`.

**Alasan.** `image_picker` menaruh berkas di direktori sementara yang
boleh dihapus iOS kapan saja — thumbnail bekerja hari ini dan rusak
minggu depan.

**Konsekuensi.** Ada duplikasi berkas sementara yang harus dibersihkan
sistem; `remove()` hanya menghapus berkas yang memang dikelola aplikasi.

---

## KD-12 — Tanggal transaksi, bukan tanggal simpan

**Keputusan.** Semua agregasi memakai `Receipt.transactionDate`.

**Alasan.** Struk bisa dimasukkan terlambat; belanjanya tetap milik bulan
saat transaksi terjadi.

**Konsekuensi.** Total bulanan bisa berubah saat struk lama ditambahkan —
itu perilaku yang benar, bukan bug.

---

## KD-13 — Peringatan buram lebih baik daripada gagal

**Keputusan.** Ambang blur (`kBlurThreshold = 45`) dan cahaya
(`kLowLightThreshold = 82`) dibuat konservatif; citra yang meragukan
tetap diproses dengan peringatan.

**Alasan.** Pengguna masih bisa memperbaiki hasil di layar review;
menolak scan terlalu dini membuat aplikasi terasa tidak berguna.

**Konsekuensi.** Sebagian hasil akan buruk dan perlu koreksi manual.

---

## KD-14 — Prediksi hanya bila data cukup

**Keputusan.** Prediksi pengeluaran butuh minimal **3 bulan** berisi data
dan tidak pernah menampilkan angka prediksi bila tidak cukup.

**Alasan.** Garis regresi dari < 3 titik tidak bermakna; menampilkan
angka tetap adalah menyesatkan.

**Konsekuensi.** Pengguna baru tidak melihat fitur prediksi sampai
mengumpulkan cukup riwayat.
