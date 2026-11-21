# 07 — Build & Rilis

## 1. Prasyarat

| Perangkat | Versi / Catatan |
|---|---|
| Flutter | SDK Dart `^3.13.2`, channel stable |
| Python (untuk `ML/`) | 3.x + `ML/requirements.txt` |
| Android | minSdk 23, izin `CAMERA` |

Pemasangan toolchain tidak dibahas di sini; ikuti dokumentasi resmi:

- Flutter: <https://docs.flutter.dev/get-started/install> (termasuk
  `flutter doctor` untuk memverifikasi Android SDK / Xcode).
- Python: pasang `requirements.txt` setelah Python 3 tersedia.

```bash
python3 -m pip install -r ML/requirements.txt
```

## 2. Dependensi Flutter

| Paket | Peran |
|---|---|
| `sqflite` / `sqflite_common_ffi` | SQLite lokal (uji di desktop memakai FFI) |
| `image_picker` | Kamera/galeri |
| `image` | Dekode & manipulasi citra (hanya di lapisan aplikasi, bukan pipeline OCR) |
| `provider` | State management |
| `shared_preferences` | Persistensi tema |
| `path_provider` | Direktori aplikasi |
| `fl_chart`, `intl`, `uuid` | Dideklarasikan; grafik & format saat ini diimplementasikan manual |
| `flutter_lints` (dev) | Lint rules |

**Tidak ada** paket OCR atau deep learning.

### Catatan SQLite di iOS

`package:sqlite3` (dependensi transitif `sqflite_common_ffi`) mengunduh
biner SQLite dari GitHub saat build hook. Hook itu juga berjalan di
build Xcode iOS, di mana unduhan gagal tanpa jaringan — karena itu
`pubspec.yaml` menetapkan:

```yaml
hooks:
  user_defines:
    sqlite3:
      source:
        ios: system
        default: sqlite3
```

iOS memakai SQLite bawaan sistem; platform lain memakai biner bundled.

## 3. Build & Jalankan

```bash
cd smart_receipt
flutter pub get
flutter run                  # debug di perangkat/emulator
flutter build apk --release  # Android
flutter build ios --release  # iOS (butuh Xcode)
```

## 4. CI: Build iOS IPA

`.github/workflows/build_ios.yml`:

- **Trigger:** push ke `main` (path `smart_receipt/**` atau workflow itu
  sendiri), tag `v*`, dan `workflow_dispatch`.
- **Runner:** `macos-latest`, Flutter stable.
- **Langkah:** `flutter pub get` → `flutter build ios --release
  --no-codesign` → zip `Payload/` → `smart_receipt.ipa`.
- **Artefak:** `smart_receipt-ios-ipa` (retensi 14 hari).
- **Tag `v*`:** IPA diterbitkan ke GitHub Releases.

## 5. Melatih Ulang Pipeline `ML/`

Semua perintah Python **dijalankan dari dalam folder `ML/`**:

```bash
python3 -m pip install -r requirements.txt

python3 -m ocr.render_dataset --n-lines 6000   # → ocr/data/chars.npz (~60 MB)
python3 -m ocr.train_char                      # → ocr/models/char_clf.joblib|npz
python3 -m ocr.export_char                     # → char_clf.json + golden_vectors.json
python3 -m ocr.export_lm                       # → lm.json, lexicon.json, templates.json

# Evaluasi
python3 -m ocr.evaluate                        # fixture terkunci receipts_v2
python3 -m ocr.synth_receipts --count 16 --seed 777
python3 -m ocr.evaluate --fixtures ocr/output/synth_receipts

# Uji
python3 -m pytest tests/ -q
```

Salin hasil ekspor (`ML/ocr/models/*.json`) ke `smart_receipt/assets/ocr/`,
lalu jalankan `flutter test` untuk memastikan paritas masih lulus. Ukuran model yang
dikirim ke perangkat ~1,3 MB.

## 6. Struktur Aset

| Path | Status |
|---|---|
| `smart_receipt/assets/ocr/` | **Dipakai** — model OCR |
| `smart_receipt/assets/model_kategori_produk.json` | **Dipakai** — model klasifikasi |
| `smart_receipt/assets/text_normalization_id.json` | **Dipakai** — praproses teks |
| `smart_receipt/assets/eval_dataset_dummy.json` | **Dipakai** — evaluasi model |
| `smart_receipt/assets/naive_bayes.json` | Relik PBL lama — tidak dibaca kode |
| `smart_receipt/Asset/model_kategori_produk.joblib` | Artefak scikit-learn asli — tidak dibundel |

## 7. Catatan Penilaian/Kuliah

Dokumen kuliah pendukung ada di `Document/`:

- `Proposal_PBL_SmartReceipt_Analyzer.docx`
- `SKPL_SmartReceipt_Analyzer.docx`
- `Laporan_Audit_Kelompok_1.docx`

Metodologi dan hasil evaluasi terperinci juga tersedia di
[`ML/README.md`](../ML/README.md).
