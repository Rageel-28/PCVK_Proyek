# 11 — Metodologi Pelatihan (`ML/`)

Folder `ML/` berisi pipeline Python yang **melatih** model dan
**mengekspor**-nya ke JSON untuk port Dart. Tidak ada pustaka OCR siap
pakai dan tidak ada deep learning.

## 1. Dependensi

`ML/requirements.txt`:

| Pustaka | Peran |
|---|---|
| `numpy` | Operasi matriks/vektor |
| `opencv-python-headless` | Render, resize, filter (dipakai saat latih/evaluasi) |
| `scikit-learn` | LinearSVC, tokenizer, metrik |
| `joblib` | Menyimpan/memuat model |
| `Pillow` | Render teks ke citra |
| `fonttools` | Menelusuri variasi font |
| `pandas`, `matplotlib` | Analisis & grafik laporan |
| `pytest` | Pengujian |

## 2. Alur Fase

Perintah lengkap ada di [07 — Build & Rilis](07-build-rilis.md#5-melatih-ulang-pipeline-ml)
dan [`ML/README.md`](../ML/README.md). Ringkasan fase:

| Fase | Skrip | Keluaran |
|---|---|---|
| F0 — dataset | `ocr.render_dataset` | `ocr/data/chars.npz` |
| F1 — classifier | `ocr.train_char` → `ocr.export_char` | `char_clf.json`, `golden_vectors.json` |
| F2/F3 — LM & leksikon | `ocr.export_lm` | `lm.json`, `lexicon.json`, `templates.json` |
| Evaluasi | `ocr.evaluate`, `ocr.synth_receipts` | Metrik CER/WER/exact |

Seluruh perintah dijalankan dari dalam folder `ML/`.

## 3. Dataset Karakter (`corpus.py`, `render_dataset.py`)

### Korpus

`corpus.py` membangkitkan baris struk sintetis dari kosakata produk
minimarket Indonesia dan format baris Indomaret/Alfamart. Karena teks
disusun sendiri, **label karakter didapat gratis** saat render.

Aturan: setiap baris hanya memakai karakter yang ada di `config.CHARSET`
plus spasi — jika tidak, ada kelas yang tidak pernah dilatih tetapi
muncul saat inferensi.

### Render & Augmentasi

`render_dataset.py`:

```
korpus teks
  → render tiap baris dengan font × ukuran
  → ambil bbox tight tiap karakter dari piksel tinta
  → normalisasi ke kanvas 32×32
  → augmentasi (blur, noise, rotasi, erosi/dilasi, …)
  → (X, y, text_index)
```

Kunci desain:

- **Bbox tight dari piksel tinta**, bukan metrik font — agar distribusi
  latih sama dengan hasil segmentasi saat inferensi.
- **Variasi penyempitan horizontal** (`AUG_CONDENSE_*`) karena font
  printer thermal jauh lebih sempit (aspek ~0,42 vs ~0,85). Tanpa ini
  huruf `A` terbaca `.` dan `D` terbaca `0` pada struk asli.
- **Pola huruf kecil berlabel kapital** (`AUG_LOWERCASE_*`): struk asli
  mencampur kapital dan huruf kecil (`Total Item`, `Kembalian`), padahal
  charset tetap kapital. Huruf kecil dirender apa adanya tetapi diberi
  label kapitalnya, mengikuti pola realistis per baris (kapital semua /
  Title Case / kecil semua). Huruf `l` dikecualikan karena bentuknya
  identik dengan `I`.
- `text_index` menandai baris sumber setiap karakter → pembagian
  train/val/test **per baris**, bukan per karakter (bebas kebocoran).
- **Booster kelas langka** (`Q`, `V`, `Z`, `&`, `+`) ditambah sampai
  ≥ 2000 sampel; kelas dominan (`-` dari baris `-----`, ~44% dataset)
  dibatasi.

## 4. Latihan Classifier (`train_char.py`)

**LinearSVC + temperature scaling.**

Pemilihan model dibuktikan secara terukur, bukan diwariskan:

| Model | Akurasi validasi |
|---|---|
| **LinearSVC** | **99,882%** |
| LogisticRegression | 99,861% |
| Soft-voting keduanya | 99,865% |

Ensemble tidak menambah akurasi sama sekali tetapi menambah ukuran
model dan biaya inferensi di perangkat. Satu model linear menang dari
sisi akurasi sekaligus kesederhanaan (dan jauh lebih mudah diport ke
Dart).

Keluaran: `char_clf.joblib` (artefak scikit-learn), `char_clf.npz`
(koefisien float32 untuk analisis), `char_clf.json` (siap-port),
`char_confusion.png` (matriks konfusi 42×42), `char_report.json`.

## 5. Ekspor (`export_char.py`, `export_lm.py`)

Semua model diekspor sebagai **JSON** agar bisa diport ke Dart tanpa
konversi biner. `export_char.py` juga menulis `golden_vectors.json` —
vektor fitur + label + probabilitas rujukan Python untuk uji paritas
implementasi Dart.

Total model yang dikirim ke perangkat:
`char_clf.json` ~734 KB + `lm.json` ~553 KB + `lexicon.json` ~28 KB +
`templates.json` ~1,2 KB ≈ **1,3 MB** (target desain < 20 MB).

## 6. Model Bahasa & Leksikon

- **`lm.py`** — n-gram karakter order 5 dengan interpolasi dan `floor`,
  token awal `\u0002` dan akhir `\u0003`.
- **`lexicon.py`** — `weightedLevenshtein(a, b, costs)` dengan biaya
  substitusi khusus untuk pasangan karakter yang mudah tertukar
  (mis. `O`↔`0`, `I`↔`1`, `S`↔`5`).
- **`templates.py`** — baris tetap (alamat, footer, label) dengan
  penjaga tanda tangan angka.

## 7. Evaluasi (`evaluate.py`)

Metrik yang dilaporkan:

- **CER** — character error rate setelah semua baris digabung.
- **WER** — word error rate.
- **exact-line** — persentase baris ground truth yang muncul persis.
- **item-accuracy** — persentase baris barang (nama + harga) yang benar.
  Ini metrik paling penting untuk pemakaian nyata: struk dibaca untuk
  mengambil daftar belanjaan, bukan transkripsi.

Himpunan `receipts_v2` adalah **test terkunci** — tidak boleh dipakai
melatih atau menyetel apa pun. Hasil lengkap ada di
[06 — Evaluasi & Pengujian](06-evaluasi-pengujian.md).

## 8. Generator Struk Held-out (`synth_receipts.py`)

`synth_receipts.py --count 16 --seed 777` membuat struk sintetis baru
(font 14–22 px, rotasi ±2,5°, blur, noise, pencahayaan tidak rata) yang
**tidak pernah dilihat** saat penyetelan parameter — proksi domain-sejati
untuk mengukur generalisasi.

## 9. Uji Python (`ML/tests/`)

17 file pytest mencakup praproses, fitur, segmentasi, model, decode,
leksikon, templat, dan pipeline:

```bash
cd ML && python3 -m pytest tests/ -q
```

## 10. Artefak yang Tidak Di-commit

`ML/ocr/data/` (dataset karakter, ~60 MB, dibangkitkan ulang) dan
`ML/ocr/output/*` (grafik/laporan) di-*gitignore* — keduanya bisa dibuat
ulang. Sebaliknya `ML/ocr/models/` (model hasil latih + JSON ekspor +
`golden_vectors.json`) **disimpan di repo** agar eksperimen dan uji
paritas dapat direproduksi tanpa melatih ulang.
