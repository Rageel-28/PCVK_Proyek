# OCR Gabungan (Hybrid) — branch `rijalmix`

Pipeline gabungan: **pra-proses, ekstraksi fitur, dan klasifikasi** dari
pipeline manual; **segmentasi, rekonstruksi spasi, dan kamus** dari pipeline
klasik yang sudah ada. Seluruh tahap berjalan di perangkat, tanpa library OCR
siap pakai dan tanpa deep learning.

## Alur

```
foto struk
  │
  ├─ grayscale (luma Rec.601)                      [manual]
  ├─ batas sisi terpanjang 2600 px                 [manual]
  ├─ upscale adaptif: tinggi baris → 60 px          [manual]
  ├─ Otsu global                                    [manual]
  ├─ hapus garis tabel (≥90% tinta, citra penuh)    [manual]
  ├─ segmentasi baris & karakter                    [klasik]
  │    proyeksi + Otsu 1D untuk foto,
  │    connected components + closing 2x2 +
  │    split bbox lebar di lembah proyeksi
  ├─ normalisasi 64x48 (stretch, dari mask biner)   [manual]
  ├─ fitur manual 172 dim                           [manual]
  ├─ Random Forest 100 pohon (70 kelas)             [manual]
  ├─ rekonstruksi spasi (celah > 2x median)         [klasik]
  └─ koreksi Levenshtein ke 3.641 kata              [klasik]
```

## Pembagian komponen

| Tahap | Sumber | Alasan |
|---|---|---|
| Pra-proses (Otsu, upscale 60 px, hapus garis) | manual | Otsu unggul pada struk dua-tone; hapus garis tabel mencegah deret digit dianggap garis |
| Segmentasi | klasik | closing menyambung stroke terputus dan split lembah memisahkan huruf menempel |
| Fitur + classifier | manual | fitur 172 dim + Random Forest; akurasi glyph 90,94% (foto) |
| Rekonstruksi + kamus | klasik | Levenshtein ke 3.641 kata (katalog produk) |

## Berkas

| Berkas | Fungsi |
|---|---|
| `manual_config.dart` | konstanta pipeline manual |
| `pil_resize.dart` | `PIL.Image.resize` (BILINEAR & LANCZOS), fixed-point 22 bit |
| `manual_preprocess.dart` | Otsu, hapus garis tabel, upscale adaptif |
| `manual_features.dart` | fitur 172 dimensi (grid 12x12 + kontur 24 + bentuk 4) |
| `rf_classifier.dart` | Random Forest 100 pohon dari biner |
| `word_lines.dart` | rekonstruksi spasi + pemecahan kata |
| `hybrid_ocr_engine.dart` | orkestrasi gabungan |

Modul klasik yang dipakai ulang: `../classical/segmenter.dart`,
`../classical/lexicon_corrector.dart`, `../classical/gray_image.dart`.

## Aset

| Aset | Isi | Ukuran |
|---|---|---|
| `assets/ocr_char_rf.bin` | RF 100 pohon, 70 kelas, 831.562 node | 11,6 MB |
| `assets/ocr_lexicon.json` | 3.641 kata (sudah ada sebelumnya) | 35 KB |

## Paritas dengan Python

| Uji | Cakupan | Hasil |
|---|---|---|
| `test/ocr_hybrid_test.dart` | fitur 172 dim pada 64 glyph asli | selisih < 1e-4 |
| `test/ocr_hybrid_test.dart` | prediksi RF pada 64 glyph | identik |
| `test/ocr_hybrid_test.dart` | teks akhir pada 3 citra struk | identik |

Teks acuan gabungan dibuat oleh `ML/manual_ocr_export/export_hybrid_reference.py`
yang menjalankan kombinasi yang sama di Python.

## Catatan hasil

Segmentasi klasik memakai tinggi baris minimum 6 px (bukan 18 px seperti
pipeline manual), sehingga **garis putus-putus pada struk ikut terbaca
sebagai baris** dan menghasilkan baris sampah (`----4----4---`). Ini
konsekuensi langsung dari memakai segmentasi klasik. Bila tidak diinginkan,
naikkan `HybridOcrEngine.minLineHeight` (misalnya ke 18).
