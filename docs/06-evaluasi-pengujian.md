# 06 — Evaluasi & Pengujian

## 1. Hasil Evaluasi OCR

Metrik utama: **CER** (character error rate), **WER** (word error rate),
dan **exact-line** (persentase baris yang terbaca persis benar).
Dievaluasi dengan `ML/ocr/evaluate.py` ala standar OCR.

### Fixture terkunci `receipts_v2`

10 struk sintetis, 138 baris, 107 baris barang. Fixture ini dipakai
selama pengembangan, jadi anggaplah **batas atas**.

| Metrik | Nilai |
|---|---|
| CER | **0,08%** |
| WER | **0,46%** |
| Exact-line | **98,6%** (136/138) |
| Nama / harga / baris barang utuh | 100,0% / 98,1% / 98,1% |

Dua baris yang meleset sama-sama huruf `4` tertutup (serif) yang terbaca
`0`; bentuk itu hanya ada pada font sintetis fixture.

### Set held-out sintetis (`synth_receipts`)

16 struk, font 14–22 px, rotasi ±2,5°, blur, noise, pencahayaan tidak
rata — **tidak pernah dilihat** saat penyetelan parameter:

| Metrik | Nilai |
|---|---|
| CER | 1,54% |
| WER | 5,71% |
| Exact-line | 90,1% |
| Nama / harga / baris utuh | 96,0% / 92,0% / 89,3% |

### Struk asli (foto ponsel)

Dua foto di `testdata/real/`:

| Struk | Hasil |
|---|---|
| Alfamart (439×927) | header terbaca utuh, nama barang + **semua harga benar** |
| Indomaret (946×1216) | nama barang + semua harga benar, baris total benar |

Kasus khusus: struk asli di atas **latar gelap** (`struk121.jpeg`, tidak
disertakan di repo — lihat catatan di bawah) sebelum perbaikan
`_paper_mask` menghasilkan 0 baris; setelahnya 25 baris,
`TOTAL BELANJA : 86.000` terbaca benar.

> **Catatan reproduksi:** `testdata/real/` hanya berisi `alfamart_01.jpeg`
> dan `indomaret_01.png`. Foto uji latar gelap (`struk121.jpeg`) dan skrip
> `struk121_check.dart` merujuk lokasi di luar repo, sehingga hasil "0 → 25
> baris" tidak dapat direproduksi apa adanya. Untuk memverifikasi, letakkan
> foto struk berlatar gelap di `testdata/real/` dan sesuaikan path skrip.

## 2. Uji Paritas Python ↔ Dart

Port Dart harus menghasilkan output yang sama dengan rujukan Python:

| Uji | Hasil |
|---|---|
| 200 golden vector (fitur + probabilitas) | selisih fitur maks **5×10⁻⁶**, selisih probabilitas < 10⁻⁴, prediksi **200/200** |
| `receipts_v2` (138 baris) | CER 0,00%, exact **135/138** (Python: 136/138 — port Dart meleset 1 glyph tambahan) |
| held-out `synth_receipts` | CER 2,36%, exact 85,2% (Python: 1,54% / 90,1%) |

> Port Dart meleset 1 glyph lebih banyak daripada Python pada satu kasus
> batas font sintetis karena perbedaan floating-point (`4.700`→`0.700`),
> sehingga exact-line Dart 135/138 vs Python 136/138.
>
> Penurunan pada set held-out bukan berarti model berbeda: modelnya
> **sama**, hanya implementasinya (Dart vs Python). Perbedaan
> floating-point paling terasa pada citra sulit (blur, rotasi,
> pencahayaan tidak rata), bukan pada fixture yang sudah "bersih".
>
> **Catatan konflik angka:** `ML/README.md` memuat dua angka yang saling
> bertentangan — tabel "uji paritas" menyebut `exact 138/138`, sedangkan
> penjelasan di atasnya menyebut 135/138. Angka yang dipakai di sini
> (135/138) diambil dari penjelasan yang lebih rinci. Jalankan ulang
> `flutter test` untuk memastikan nilai sebenarnya saat ini.

## 3. Suite Pengujian Flutter (`smart_receipt/test/`)

21 file `*_test.dart` otomatis + 2 skrip manual (`real_receipt_check.dart`,
`struk121_check.dart`):

| Kelompok | File |
|---|---|
| **OCR & paritas** | `ocr_golden_test` (paritas vektor), `ocr_pipeline_test` (end-to-end + CER/WER), `ocr_gray_test` (regresi PNG 1-kanal), `ocr_text_corrector_test` (30), `ocr_abbreviations_test`, `paper_gate_test` (regresi deteksi kertas) |
| **Parser** | `receipt_parser_test` (23) — geometri/kolom/teks, stacked cells, nominal terpecah, total & pembulatan, diskon, merchant rusak, tanggal |
| **Klasifikasi** | `category_taxonomy_test` (16), `shipped_model_test`, `model_evaluator_test`, `text_preprocessing_test` |
| **Data & state** | `database_helper_test`, `receipt_provider_test`, `receipt_image_store_test`, `correction_training_test` |
| **Analitik** | `spending_forecaster_test` (11) |
| **Widget** | `route_animation_test`, `theme_toggle_test`, `receipt_photo_preview_test`, `widget_test` |
| **Diagnostik** | `diag_debug_test` (membantu debug parser, bukan gerbang mutu) |
| **Manual (bukan suite)** | `real_receipt_check.dart`, `struk121_check.dart` |

Fixtures: `fixtures/receipts_v2/` (10 struk + ground truth),
`fixtures/ocr_golden/golden_vectors.json`, `fixtures/model_kategori_produk.json`,
`fixtures/eval_dataset_dummy.json`, `fixtures/image.png`.

## 4. Pengujian Python (`ML/tests/`)

17 file pytest mencakup praproses, fitur, segmentasi, model, decode,
leksikon, templat, dan pipeline. Jalankan:

```bash
cd ML && python3 -m pytest tests/ -q
```

## 5. Cara Menjalankan

```bash
# Port Dart (semua uji Flutter)
cd smart_receipt && flutter test

# Rujukan Python
cd ML && python3 -m pytest tests/ -q

# Evaluasi ulang fixture terkunci & held-out
cd ML
python3 -m ocr.evaluate
python3 -m ocr.synth_receipts --count 16 --seed 777
python3 -m ocr.evaluate --fixtures ocr/output/synth_receipts
```

`testdata/real/` menyimpan dua foto struk asli untuk pemeriksaan manual
di luar domain sintetis.
