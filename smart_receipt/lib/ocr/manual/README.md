# OCR Manual — Fitur 172 Dimensi + Random Forest

Pipeline OCR buatan sendiri yang dipakai aplikasi pada branch `rijalori`.
Seluruh tahap berjalan di perangkat; tidak memakai library OCR siap pakai
(ML Kit, Tesseract) dan tidak memakai deep learning.

## Alur

```
foto struk
  │
  ├─ grayscale (luma Rec.601)
  ├─ batas sisi terpanjang 2600 px        (resize bilinear gaya PIL)
  ├─ upscale adaptif: tinggi baris → 60 px (maks 6x, LANCZOS gaya PIL)
  ├─ Otsu global                          (satu ambang dari histogram)
  ├─ hapus garis tabel                    (baris/kolom ≥90% tinta, citra penuh)
  ├─ segmentasi baris                     (proyeksi horizontal, gap 4 px,
  │                                        tinggi 18–160 px)
  ├─ segmentasi glyph                     (connected components 4-arah +
  │                                        merge titik i/j)
  ├─ normalisasi 64x48                    (stretch, crop dari mask biner)
  ├─ fitur manual 172 dim                 (grid 12x12 + kontur 24 + bentuk 4)
  ├─ Random Forest 100 pohon              (70 kelas, traversal pohon)
  ├─ rekonstruksi spasi                   (celah > 2x median)
  └─ koreksi leksikal                     (tipe token + pola uang + kamus
                                           edit-1, deterministik)
```

## Berkas

| Berkas | Padanan Python |
|---|---|
| `manual_config.dart` | konstanta `annotate_lines.py` + `rebuild_feature_cache.py` |
| `pil_resize.dart` | `PIL.Image.resize` (BILINEAR & LANCZOS) |
| `manual_preprocess.dart` | `otsu_threshold`, `binarize`, `remove_rules`, `load_image` |
| `manual_segmenter.dart` | `detect_lines`, `segment_glyphs`, `merge_stacked` |
| `manual_features.dart` | `feat_g12` |
| `rf_classifier.dart` | `RandomForestClassifier.predict` (sklearn) |
| `manual_lexicon.dart` | `lexical_correct.py` |
| `word_lines.dart` | rekonstruksi spasi + pemecahan kata |
| `manual_ocr_engine.dart` | `ocr_pipeline.py` (versi manual) |

## Aset

| Aset | Isi | Ukuran |
|---|---|---|
| `assets/ocr_char_rf.bin` | RF 100 pohon, 70 kelas, 831.562 node | 11,6 MB |
| `assets/ocr_manual_lexicon.json` | 554 kata (kata umum struk + katalog produk) | 4,4 KB |

Model biner dibuat dari `ML/Percobaan OCR Manual/tools/model_rf/rf_100_full.json`
(model produksi, dilatih pada 100% data). JSON aslinya 20,8 MB; biner 11,6 MB
dan dibaca sebagai `ByteData`, jadi tidak ada `jsonDecode` besar saat aplikasi
mulai.

Regenerasi aset:

```bash
cd ML/manual_ocr_export
python export_manual_assets.py
```

## Paritas dengan Python

| Uji | Cakupan | Hasil |
|---|---|---|
| `test/ocr_manual_test.dart` | fitur 172 dim pada 64 glyph asli | selisih < 1e-4 |
| `test/ocr_manual_test.dart` | prediksi RF pada 64 glyph | 64/64 identik |
| `test/ocr_manual_test.dart` | teks akhir pada 3 citra struk | identik |

Hal yang harus ditiru persis agar paritas tercapai:

1. **Resize PIL** — koefisien fixed-point 22 bit, akumulasi integer,
   `+ (1 << 21)` sebelum digeser, dan hasil antara dipotong ke uint8.
   Interpolasi bilinear biasa berbeda 1 LSB dan cukup untuk menggeser fitur.
2. **Upscale pada grayscale** — citra diperbesar sebelum binarisasi, sesuai
   `load_image()`, bukan setelahnya.
3. **Hapus garis pada citra penuh** — per baris, deret digit padat
   (`081584471843`) terlihat seperti garis vertikal dan ikut terhapus.
4. **Connected components 4-arah** — `scipy.ndimage.label` bawaan memakai
   struktur cross, bukan 8-arah.
5. **Merge titik i/j** — overlap horizontal ≥50% lebar yang lebih kecil dan
   jarak vertikal ≤25% tinggi gabungan.
6. **Crop dari mask biner** — bukan dari grayscale; fitur dilatih pada
   okupansi 0/1.
7. **Kamus terurut** — `lexical_correct` berhenti pada kandidat pertama
   berjarak 1. Versi Python mengiterasi `set` (urutan arbitrer) sehingga
   token seperti `OAT` bisa menjadi `OBT` atau `CAT` pada proses berbeda.
   Port ini mengurutkan kamus agar hasilnya deterministik.
