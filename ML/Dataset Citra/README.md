# Dataset Struk Indomaret / Alfamart

Kumpulan foto dan render struk belanja Indonesia yang dipakai untuk
evaluasi dan pelatihan pipeline OCR klasik (`ML/ocr_classical`).

## Isi folder

| Pola nama | Jumlah | Sifat | Resolusi |
|---|---:|---|---|
| `receipt-*.jpg` | 500 | render bersih dari template (tegak, kontras tinggi) | 528 × 756–1008 |
| `CamScanner-*.jpg` | 122 | **foto asli** (kertas kusut, bayangan, QR) | 691–2974 × 3000 |

Folder ini **hanya berisi piksel**. Anotasi bounding box baris ada di
arsip ZIP di direktori induk:

- `ML/Receipt Indomaret.v1i.coco.zip` — 500 gambar, 2.156 bbox
- `ML/Alfamart Indomaret Struk.v1i.coco.zip` — 122 gambar, 630 bbox

Anotasi menandai **baris item** saja (`product_item`,
`product_item_discount`, `product_item_voucher`), bukan seluruh baris
struk. Tidak ada transkripsi teks di dataset; transkripsi manual untuk
9 struk ada di `ML/ocr_classical/transcripts_real.json`.

> **Penting.** Gambar di dalam ZIP diekspor dengan hash `.rf.<hash>` yang
> berbeda dari folder ini, dan ekspor `Receipt Indomaret` memakai
> *Resize 640×640 (Stretch)*. Karena itu `ML/ocr_classical/real_data.py`
> mencocokkan berkas lewat *stem* sebelum `.rf.` dan memetakan koordinat
> bbox kembali ke ukuran piksel asli di folder ini. Jangan hapus ZIP-nya.

## Sumber dan lisensi

Kedua dataset berasal dari workspace Roboflow `jupar`:

| Dataset | URL | Versi |
|---|---|---|
| Receipt Indomaret | https://universe.roboflow.com/jupar/receipt-indomaret-zdpdk | v1 |
| Alfamart Indomaret Struk | https://universe.roboflow.com/jupar/alfamart-indomaret-struk-hjhso | v1 |

Lisensi: **CC BY 4.0**. Atribusi wajib disertakan bila data ini dipakai
ulang. Berkas `README.dataset.txt` dan `README.roboflow.txt` di dalam
masing-masing ZIP memuat keterangan resmi dari Roboflow.

## Catatan pemakaian dalam laporan

500 dari 622 gambar adalah **render template**, bukan foto. Header dan
footer antar gambar identik (`KS TUBUN 0293493733`, `LAYANAN KONSUMEN`),
hanya baris itemnya berbeda. Karena itu:

- klaim ketahanan terhadap foto nyata **hanya** boleh bersandar pada 122
  gambar `CamScanner-*`;
- hasil evaluasi wajib dipisahkan antara `jenis=render` dan `jenis=foto`
  (lihat `ML/ocr_classical/HASIL_DATA_ASLI.md`).
