"""Ekspor aset OCR manual (Random Forest 172 dim) untuk aplikasi Flutter.

Menghasilkan tiga hal di `smart_receipt/`:

  1. `assets/ocr_char_rf.bin`
     Model Random Forest 100 pohon dari
     `ML/Percobaan OCR Manual/tools/model_rf/rf_100_full.json` (model
     produksi, dilatih pada 100% data) dalam format biner ringkas.
     JSON aslinya 20,8 MB; biner 11,6 MB dan tidak perlu di-parse saat
     aplikasi mulai (cukup dibaca sebagai ByteData).

  2. `assets/ocr_manual_lexicon.json`
     Kosakata untuk koreksi leksikal: kata umum struk dari
     `tools/lexical_correct.py` ditambah nama/alias produk dari
     `ML/katalog_produk_indonesia_v1.json`, persis seperti
     `load_vocabulary()`.

  3. `test/fixtures/manual_rf_golden.json` + `manual_line_*.png`
     Vektor uji paritas: potongan glyph beserta fitur 172 dim dan kelas
     hasil prediksi Python, serta tiga citra struk dengan teks acuan dari
     pipeline referensi. Dipakai `test/ocr_manual_test.dart`.

Pemakaian:
    python export_manual_assets.py
"""
from __future__ import annotations

import csv
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "ML" / "Percobaan OCR Manual" / "tools"
APP = ROOT / "smart_receipt"
DATA = ROOT / "ML" / "Percobaan OCR Manual" / "Dataset Citra"

sys.path.insert(0, str(TOOLS))
import annotate_lines as A  # noqa: E402
import lexical_correct as LC  # noqa: E402
from experiment_augment import trim  # noqa: E402
from experiment_fast import contours, grid  # noqa: E402

MODEL_JSON = TOOLS / "model_rf" / "rf_100_full.json"
CATALOG_JSON = ROOT / "ML" / "katalog_produk_indonesia_v1.json"
GLYPH_CSV = TOOLS / "annotations" / "glyph_dataset.csv"
SPLIT_CSV = TOOLS / "annotations" / "split.csv"

CHAR_SIZE = (64, 48)   # (lebar, tinggi) — PIL resize memakai urutan ini
MAGIC = b"RF01"


# ---------------------------------------------------------------------------
# Fitur 172 dimensi — identik dengan `feat_g12` di rebuild_feature_cache.py
# ---------------------------------------------------------------------------
def feat_g12(mask: np.ndarray) -> np.ndarray:
    g = trim(mask)
    arr = np.array(
        Image.fromarray((g * 255).astype(np.uint8)).resize(CHAR_SIZE, Image.BILINEAR)
    ).astype(np.float32) / 255.0
    h, w = arr.shape
    tv = np.abs(np.diff((arr > 0.5).astype(int), axis=0)).sum(axis=0)
    shape = [min(w / max(h, 1), 2.0) / 2.0, float(arr.mean()),
             float(tv.mean()), float((tv > 0).sum()) / max(w, 1)]
    return np.array(grid(arr, 12, 12) + contours(arr, 12) + shape, np.float32)


# ---------------------------------------------------------------------------
# Inferensi Random Forest dari JSON (untuk membuat label acuan)
# ---------------------------------------------------------------------------
class Forest:
    def __init__(self, data: dict) -> None:
        self.classes: list[str] = list(data["classes"])
        self.trees = data["trees"]
        self.n_features = int(data["n_features"])

    def predict(self, x: np.ndarray) -> str:
        votes = [0] * len(self.classes)
        for tree in self.trees:
            feature = tree["feature"]
            threshold = tree["threshold"]
            left = tree["left"]
            right = tree["right"]
            value = tree["value"]
            node = 0
            # sklearn menandai daun dengan feature = -2 (TREE_UNDEFINED) dan
            # children = -1, jadi daun dideteksi dari nilai negatif.
            while feature[node] >= 0:
                if x[feature[node]] <= threshold[node]:
                    node = left[node]
                else:
                    node = right[node]
            votes[int(value[node])] += 1
        return self.classes[int(np.argmax(votes))]


# ---------------------------------------------------------------------------
# 1. Model biner
# ---------------------------------------------------------------------------
def export_binary_model(forest_json: dict, path: Path) -> None:
    """Tulis model ke biner little-endian.

    Layout:
        magic "RF01"
        u32 nTrees | u16 nClasses | u16 nFeatures
        nClasses x (u8 panjang + byte ASCII)
        per pohon: u32 nodeCount, lalu nodeCount x 14 byte:
            u8 feature (255 = daun) | f32 threshold | i32 left | i32 right | u8 value
    """
    classes = [str(c) for c in forest_json["classes"]]
    trees = forest_json["trees"]

    out = bytearray()
    out += MAGIC
    out += struct.pack("<IHH", len(trees), len(classes), int(forest_json["n_features"]))
    for c in classes:
        raw = c.encode("utf-8")
        if len(raw) > 255:
            raise ValueError("label kelas terlalu panjang: %r" % c)
        out += bytes([len(raw)]) + raw

    nodes = 0
    for tree in trees:
        feature = tree["feature"]
        threshold = tree["threshold"]
        left = tree["left"]
        right = tree["right"]
        value = tree["value"]
        out += struct.pack("<I", len(feature))
        for i in range(len(feature)):
            is_leaf = feature[i] < 0      # -2 di sklearn, apa pun tandanya
            if is_leaf:
                # sklearn: left == right == -1 pada daun; nilai = indeks kelas.
                out += struct.pack(
                    "<BfiiB",
                    255,
                    0.0,
                    -1,
                    -1,
                    int(value[i]) & 0xFF,
                )
            else:
                out += struct.pack(
                    "<BfiiB",
                    int(feature[i]) & 0xFF,
                    float(threshold[i]),
                    int(left[i]),
                    int(right[i]),
                    0,
                )
            nodes += 1

    path.write_bytes(bytes(out))
    print("model  : %s (%.1f MB, %d pohon, %d node, %d kelas)"
          % (path.relative_to(ROOT), len(out) / 1e6, len(trees), nodes, len(classes)))


# ---------------------------------------------------------------------------
# 2. Leksikon
# ---------------------------------------------------------------------------
def export_lexicon(path: Path) -> set[str]:
    """Kosakata seperti `lexical_correct.load_vocabulary()`."""
    import re

    vocab: set[str] = set(LC.COMMON_RECEIPT_WORDS)
    if CATALOG_JSON.exists():
        data = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))
        for product in data.get("products", []):
            for source in [product.get("name") or ""] + list(product.get("aliases", [])):
                for word in re.split(r"[^A-Za-z0-9]+", source.upper()):
                    if len(word) >= 3:
                        vocab.add(word)

    path.write_text(
        json.dumps({"words": sorted(vocab)}, separators=(",", ":")),
        encoding="utf-8",
    )
    print("lexicon: %s (%d kata, %.1f KB)"
          % (path.relative_to(ROOT), len(vocab), path.stat().st_size / 1024))
    return vocab


# ---------------------------------------------------------------------------
# 3a. Golden glyph: fitur + kelas
# ---------------------------------------------------------------------------
def export_glyph_golden(forest: Forest, path: Path,
                        per_domain: int = 4, per_image: int = 8) -> None:
    """Ambil glyph dari sedikit citra saja agar ekspor cepat (muat 1x per citra)."""
    rows = list(csv.DictReader(GLYPH_CSV.open("r", encoding="utf-8")))

    rows_by_image: dict[str, list[dict]] = {}
    for r in rows:
        rows_by_image.setdefault(r["image"], []).append(r)

    # Pilih citra domain lama & baru (utamakan yang ada di split test).
    split: dict[str, str] = {}
    if SPLIT_CSV.exists():
        for r in csv.DictReader(SPLIT_CSV.open("r", encoding="utf-8")):
            split[r["image"]] = r["split"]

    def is_baru(name: str) -> bool:
        return name.startswith("receipt")

    def pick(domain_baru: bool) -> list[str]:
        candidates = [n for n in rows_by_image if is_baru(n) == domain_baru]
        candidates.sort(key=lambda n: (split.get(n) != "test", n))
        return candidates[:per_domain]

    chosen = pick(True) + pick(False)

    samples = []
    for name in chosen:
        src = DATA / name
        if not src.exists():
            continue
        mask = A.prepare_mask(A.load_image(src))

        picked: list[dict] = []
        seen: Counter[str] = Counter()
        for r in rows_by_image[name]:
            if len(picked) >= per_image:
                break
            # Batasi 2 glyph per kelas agar sampel tersebar.
            if seen[r["label"]] >= 2:
                continue
            picked.append(r)
            seen[r["label"]] += 1

        for r in picked:
            top, left = int(r["top"]), int(r["left"])
            h, w = int(r["height"]), int(r["width"])
            sub = mask[top:top + h, left:left + w]
            if sub.size == 0 or not sub.any():
                continue

            feats = feat_g12(sub)
            samples.append({
                "image": name,
                "label": r["label"],
                "pred": forest.predict(feats),
                "width": int(w),
                "height": int(h),
                "mask": "\n".join("".join("1" if v else "0" for v in row)
                                  for row in sub),
                "features": [round(float(v), 6) for v in feats],
            })

    payload = {
        "n_features": int(forest.n_features),
        "char_size": list(CHAR_SIZE),
        "classes": forest.classes,
        "samples": samples,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    agree = sum(1 for s in samples if s["label"] == s["pred"])
    print("golden : %s (%d glyph dari %d citra, model setuju label %d/%d, %.0f KB)"
          % (path.relative_to(ROOT), len(samples), len(chosen), agree, len(samples),
             path.stat().st_size / 1024))


# ---------------------------------------------------------------------------
# 3b. Referensi pipeline lengkap (satu spesifikasi dengan port Dart)
# ---------------------------------------------------------------------------
def join_with_spaces(boxes: list[tuple[int, int, int, int]], chars: list[str]) -> str:
    """Sisipkan spasi: celah > 2x median celah pada baris itu."""
    if not boxes:
        return ""
    if len(boxes) == 1:
        return chars[0]

    gaps = [boxes[i][1] - (boxes[i - 1][1] + boxes[i - 1][3])
            for i in range(1, len(boxes))]
    median_gap = float(np.median(gaps))
    threshold = max(2.0, median_gap * 2.0)

    out = [chars[0]]
    for i in range(1, len(boxes)):
        gap = boxes[i][1] - (boxes[i - 1][1] + boxes[i - 1][3])
        if gap > threshold:
            out.append(" ")
        out.append(chars[i])
    return "".join(out)


def manual_read(path: Path, forest: Forest, vocab: set[str]) -> dict:
    """Jalankan pipeline manual pada satu berkas; kembalikan teks per baris.

    Kosakata diurutkan sebelum dipakai mengoreksi. `lexical_correct`
    mengiterasi himpunan kata dan berhenti pada kandidat pertama yang
    berjarak 1; urutan iterasi `set` Python arbitrer, sehingga token
    seperti `OAT` bisa menjadi `OBT` atau `CAT` pada proses berbeda.
    Mengurutkannya membuat hasil reproduksibel dan sama dengan port Dart.
    """
    ordered_vocab = sorted(vocab)
    image = A.load_image(path)
    mask = A.prepare_mask(image)

    lines_out = []
    for ln in A.detect_lines(mask, path.name):
        band = mask[ln.top:ln.bottom + 1, ln.left:ln.right + 1]
        raw_boxes = A.segment_glyphs(band)
        if not raw_boxes:
            continue

        # Buang bintik noise: tinggi < 10% median baris (tanda baca tetap).
        med_h = float(np.median([b[2] for b in raw_boxes]))
        min_h = max(3.0, med_h * 0.10)
        kept = [b for b in raw_boxes if b[2] >= min_h and b[3] >= 2]
        if not kept:
            continue

        chars = []
        for (gy, gx, gh, gw) in kept:
            sub = mask[ln.top + gy:ln.top + gy + gh, ln.left + gx:ln.left + gx + gw]
            chars.append(forest.predict(feat_g12(sub)))

        text = join_with_spaces([(0, b[1], b[2], b[3]) for b in kept], chars)
        corrected, _changes = LC.correct_line(text, ordered_vocab)
        lines_out.append({"text": corrected, "chars": len(chars)})

    return {
        "size": [int(image.size[0]), int(image.size[1])],
        "lines": lines_out,
        "text": "\n".join(l["text"] for l in lines_out),
    }


def export_line_fixtures(forest: Forest, vocab: set[str],
                         out_dir: Path, manifest: Path, count: int = 3) -> None:
    """Simpan beberapa citra struk (PNG grayscale) + teks acuan referensi."""
    split: dict[str, str] = {}
    if SPLIT_CSV.exists():
        for r in csv.DictReader(SPLIT_CSV.open("r", encoding="utf-8")):
            split[r["image"]] = r["split"]

    files = sorted(p for p in DATA.glob("*.jpg"))
    if not files:
        print("  (lewati fixture: Dataset Citra tidak ditemukan)")
        return

    # Sebar pilihan: satu domain baru, satu domain lama, satu acak.
    baru = [p for p in files if p.name.startswith("receipt")]
    lama = [p for p in files if not p.name.startswith("receipt")]
    chosen = (baru[:2] + lama[:1]) if baru and lama else files[:count]

    entries = []
    for i, src in enumerate(chosen[:count]):
        image = Image.open(src).convert("L")
        if max(image.size) > 1400:
            scale = 1400 / max(image.size)
            image = image.resize(
                (max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale))),
                Image.LANCZOS,
            )
        name = "manual_line_%d.png" % i
        target = out_dir / name
        image.save(target, optimize=True)

        result = manual_read(target, forest, vocab)
        entries.append({
            "file": name,
            "source": src.name,
            "domain": "baru" if src.name.startswith("receipt") else "lama",
            "text": result["text"],
            "lines": result["lines"],
        })
        print("  fixture %s <- %s (%d baris, %d byte)"
              % (name, src.name, len(result["lines"]), target.stat().st_size))

    manifest.write_text(json.dumps({"images": entries}, indent=1), encoding="utf-8")
    print("manifest: %s" % manifest.relative_to(ROOT))


# ---------------------------------------------------------------------------
def main() -> None:
    if not MODEL_JSON.exists():
        raise SystemExit("model tidak ditemukan: %s" % MODEL_JSON)

    assets = APP / "assets"
    fixtures = APP / "test" / "fixtures"
    assets.mkdir(parents=True, exist_ok=True)
    fixtures.mkdir(parents=True, exist_ok=True)

    model_json = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    export_binary_model(model_json, assets / "ocr_char_rf.bin")

    vocab = export_lexicon(assets / "ocr_manual_lexicon.json")

    forest = Forest(model_json)
    export_glyph_golden(forest, fixtures / "manual_rf_golden.json")
    export_line_fixtures(forest, vocab, fixtures, fixtures / "manual_lines.json")


if __name__ == "__main__":
    main()
