# Ekspor Aset OCR Manual

Alat untuk menyiapkan aset yang dipakai `smart_receipt/lib/ocr/manual/`.

## Yang dihasilkan

| Keluaran | Isi |
|---|---|
| `smart_receipt/assets/ocr_char_rf.bin` | Random Forest 100 pohon (70 kelas, 172 fitur) dalam biner ringkas |
| `smart_receipt/assets/ocr_manual_lexicon.json` | 554 kata: kata umum struk + nama/alias produk |
| `smart_receipt/test/fixtures/manual_rf_golden.json` | 64 glyph asli + fitur 172 dim + kelas acuan Python |
| `smart_receipt/test/fixtures/manual_line_*.png` | 3 citra struk (PNG grayscale, tanpa kompresi lossy) |
| `smart_receipt/test/fixtures/manual_lines.json` | Teks acuan pipeline **manual** untuk ketiga citra |
| `smart_receipt/test/fixtures/hybrid_lines.json` | Teks acuan pipeline **gabungan** untuk ketiga citra |

`manual_lines.json` dipakai branch `rijalori`; `hybrid_lines.json` dipakai
branch `rijalmix`. Keduanya memakai citra fixture yang sama.

## Pemakaian

```bash
cd ML/manual_ocr_export
python export_manual_assets.py        # model biner, leksikon, golden vector
python export_hybrid_reference.py     # teks acuan pipeline gabungan
```

Prasyarat: `ML/Percobaan OCR Manual/Dataset Citra/` dan
`ML/Percobaan OCR Manual/tools/annotations/glyph_dataset.csv` tersedia
(keduanya tidak di-commit).

## Format model biner

Little-endian, tanpa padding:

```
magic        "RF01"
u32          nTrees
u16          nClasses
u16          nFeatures
nClasses x   (u8 panjang + byte UTF-8 label)
per pohon    u32 nodeCount, lalu nodeCount x 14 byte:
             u8 feature (255 = daun)
             f32 threshold
             i32 left   (relatif terhadap awal pohon, -1 di daun)
             i32 right
             u8 value   (indeks kelas pada daun)
```

JSON dari scikit-learn 20,8 MB dan butuh `jsonDecode` besar; biner 11,6 MB
dan dibaca langsung sebagai `ByteData`.

## Catatan penting

1. **Daun sklearn ditandai `feature = -2`** (`TREE_UNDEFINED`), bukan `-1`.
   Deteksi daun harus memakai nilai negatif; memakai `== -1` membuat
   traversal berputar tanpa henti.
2. **Indeks anak relatif per pohon.** Saat pohon digabung ke array global,
   indeks harus digeser dengan basis pohon, jika tidak prediksi menjadi acak.
3. **Kosakata diurutkan sebelum koreksi.** `lexical_correct.correct_line`
   berhenti pada kandidat pertama berjarak edit 1; urutan `set` Python
   arbitrer, sehingga hasilnya tidak reproduksibel. Mengurutkan kosakata
   membuat referensi dan port Dart memberi hasil sama.

## Skrip pendamping

| Berkas | Fungsi |
|---|---|
| `export_manual_assets.py` | Model biner, leksikon, golden vector, fixture citra |
| `export_hybrid_reference.py` | Teks acuan pipeline gabungan (branch `rijalmix`) |
| `pil_resize_ref.py` | Membuktikan implementasi resize identik dengan PIL (selisih 0) |
| `debug_reference.py` | Mencetak kotak glyph & karakter referensi (debug paritas) |
