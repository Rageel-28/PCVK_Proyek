# 09 — Analitik & Prediksi Pengeluaran

Dua layanan murni-Dart mengolah struk yang tersimpan menjadi ringkasan
dan estimasi pengeluaran. Keduanya **tanpa ketergantungan Flutter**,
sehingga dapat diuji tanpa lingkungan widget.

## 1. Agregasi Pengeluaran (`spending_analytics.dart`)

Semua total dihitung dari **tanggal transaksi tercetak di struk**
(`Receipt.transactionDate`), **bukan** dari waktu baris disimpan
(`created_at`). Konsekuensinya, struk yang dimasukkan terlambat tetap
masuk ke bulan yang sebenarnya.

| Fungsi | Hasil |
|---|---|
| `monthStart(d)` | Tengah malam hari pertama bulan |
| `addMonths(d, n)` | Geser rentang kalender (dinormalisasi ke tanggal 1) |
| `inRange(date, from, to)` | Uji setengah-terbuka `[from, to)` |
| `totalBetween(receipts, from, to)` | Total belanja dalam rentang |
| `countBetween(receipts, from, to)` | Jumlah struk dalam rentang |
| `byCategory(receipts, names, from, to)` | Belanja per kategori |
| `monthlyTotals(receipts, months, now)` | Deret total per bulan (bulan kosong = 0) |
| `trimLeadingEmpty(series)` | Buang bulan-bulan kosong di awal |

Dua keputusan penting:

- **Rincian per kategori memakai total tingkat item**, bukan header struk,
  agar komponen-komponennya berjumlah tepat sama dengan total item
  meskipun struk juga memuat pajak atau pembulatan.
- `monthlyTotals` mengisi bulan tanpa struk dengan **0** supaya sumbu-x
  grafik tetap rata.
- `trimLeadingEmpty` membuang bulan kosong **di awal riwayat**; jika
  tidak, bulan sebelum struk pertama pengguna akan menjadi titik data
  nol dan menarik garis regresi turun secara artifisial.

## 2. Prediksi Pengeluaran (`spending_forecaster.dart`)

`SpendingForecaster.forecast(series)` mencocokkan **regresi linear**
`y = a + b·x` dengan metode kuadrat terkecil biasa (ordinary least
squares) atas deret total bulanan.

### Perhitungan

```
slope     = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²
intercept = ȳ − slope·x̄
predicted = max(0, intercept + slope·n)     // bulan setelah observasi terakhir
```

- `x` = indeks bulan (0, 1, 2, …), `y` = total belanja bulan tersebut.
- Prediksi dibatasi ≥ 0.

### Metrik

| Metrik | Rumus | Catatan |
|---|---|---|
| `mae` | Σ\|y − ŷ\| / n | kesalahan absolut rata-rata |
| `rmse` | √(Σ(y − ŷ)² / n) | kesalahan kuadratik rata-rata |
| `rSquared` | 1 − SS_res / SS_tot | 1 = cocok sempurna; ditangani khusus jika variansi total = 0 |
| `slope` | b | perubahan rupiah per bulan (positif = belanja naik) |
| `changeFromLastMonth` | (prediksi − bulan terakhir)/bulan terakhir | null bila bulan terakhir 0 |

Metrik dihitung **in-sample** (dievaluasi pada bulan yang sama tempat
model dilatih). Dengan hanya beberapa titik data, ini pilihan yang jujur,
dan UI melabelinya sebagai **estimasi**.

### Aturan Data Tidak Cukup

`SpendingForecaster.minMonths = 3` — pemeriksaannya `series.length < 3`,
yaitu **panjang deret** yang diberikan, bukan jumlah bulan yang benar-benar
berisi belanja. Karena itu pemanggil wajib menjalankan
`SpendingAnalytics.trimLeadingEmpty()` terlebih dahulu (membuang bulan kosong
di awal riwayat); tanpa itu, bulan nol di awal ikut terhitung sebagai sampel.

Bila tidak cukup, hasilnya `hasEnoughData = false` dan UI menampilkan kartu
"data belum cukup" (`_InsufficientDataCard`) alih-alih mengarang angka
(aturan bisnis BR-08 pada SKPL).

**Asal deret:** UI mengambil `monthlyTotals(receipts, months: 6)` (default
6 bulan, disesuaikan pilihan periode), lalu `trimLeadingEmpty`, baru
`forecast`. Contoh di §4 memakai deret yang sudah dipangkas.

### Label Bulan

`label(DateTime)` dan `nextLabel(DateTime)` menghasilkan singkatan
Indonesia: `Jan Feb Mar Apr Mei Jun Jul Agu Sep Okt Nov Des`.

## 3. Alur di UI

`AnalysisScreen` (layar 9) memilih periode (bulan ini, 3/6/12 bulan,
tahun), menampilkan ringkasan, distribusi kategori, grafik tren
(`TrendLineChart`), dan tautan ke `SpendingPredictionScreen`.

`SpendingPredictionScreen` (layar 10) memanggil `SpendingForecaster`,
menampilkan nilai prediksi, MAE/RMSE/R², dan kartu insight.

## 4. Contoh

Misalkan total bulanan (Agu–Okt): `[400.000, 450.000, 520.000]`.

- `x = [0, 1, 2]`, `y = [400e3, 450e3, 520e3]`
- `slope = 60.000`, `intercept ≈ 396.667`
- `predicted` (Nov) `≈ 396.667 + 60.000·3 = 576.667`
- `rSquared ≈ 0,991` (data hampir linear)
- `changeFromLastMonth = (576.667−520.000)/520.000 ≈ +10,9%`
