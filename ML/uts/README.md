# UTS PBL Machine Learning — SmartReceipt Analyzer

Perbandingan empat model klasifikasi kategori item struk belanja di atas fitur TF-IDF
yang sama. Setiap anggota mengerjakan satu notebook model; hasilnya digabungkan pada
notebook perbandingan dan laporan.

## Berkas

| Berkas | Isi |
|---|---|
| `rakagali_naive_bayes.ipynb` | Model Multinomial Naive Bayes (Rakagali) |
| `aldi_knn.ipynb` | Model k-Nearest Neighbors (Aldi) |
| `rijal_svm.ipynb` | Model Support Vector Machine (Rijal) |
| `noval_random_forest.ipynb` | Model Random Forest (Noval) |
| `perbandingan_model.ipynb` | Melatih keempat model dan membandingkan hasilnya |
| `pipeline_utils.py` | Pipeline bersama: pra-pengolahan, split data, TF-IDF, evaluasi |
| `hasil_perbandingan.csv` | Ringkasan metrik keempat model |
| `Laporan_UTS_PBL_ML.docx` / `.md` | Laporan UTS sesuai template |
| `Template Laporan UTS PBL ML.docx` | Template asli (tidak diubah) |

## Aturan yang Dikunci

Agar perbandingan adil, semua notebook memakai:

- Dataset yang sama: `Dataset_SmartReceipt_Sintetis.csv` + `Dataset_Katalog_Indonesia_Sintetis_v1.csv`.
- Split bawaan dataset dengan *leakage guard*: data latih 5.569 baris, data uji 790 baris, validasi 762 baris.
- Pra-pengolahan yang sama (`pipeline_utils.TextPreprocessor`).
- TF-IDF yang sama: `analyzer="char_wb"`, `ngram_range=(2, 4)`, `sublinear_tf=True`, `max_features=20000`.

Yang berbeda hanya algoritma klasifikasinya.

## Cara Menjalankan

1. Install dependensi:
   ```bash
   pip install -r ../requirements.txt
   ```
2. Buka salah satu notebook dari folder `ML/uts`, lalu jalankan semua sel
   (*Run All*). Notebook mengimpor `pipeline_utils.py` dari folder yang sama.
3. Untuk membandingkan seluruh model, jalankan `perbandingan_model.ipynb`.
   Hasilnya diperbarui ke `hasil_perbandingan.csv`.

Catatan: notebook tidak memerlukan library plotting; seluruh keluaran berupa tabel
dan teks agar dapat dijalankan dengan dependensi minimal.

## Hasil Singkat (data uji, 790 baris)

| Model | Akurasi | Macro-F1 | Waktu Latih |
|---|---:|---:|---:|
| k-Nearest Neighbors | 77,34% | 0,7365 | 0,13 s |
| Support Vector Machine | 74,68% | 0,7050 | 0,22 s |
| Multinomial Naive Bayes | 74,18% | 0,7063 | 0,04 s |
| Random Forest | 72,91% | 0,6946 | 3,86 s |

Multinomial Naive Bayes tetap dipilih untuk produksi karena paling ringan dan cepat
untuk inferensi *on-device* di aplikasi Flutter.
