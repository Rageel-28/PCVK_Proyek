"""Alat pelabelan glyph per BARIS untuk data struk thermal.

Kenapa per baris, bukan per huruf
---------------------------------
Melabeli ±250 glyph satu per satu terlalu lambat. Satu baris struk sudah
bermakna ("JAVANA TEH MLATI 350"), jadi cukup ketik teks baris SEKALI lalu
alat ini memetakan tiap karakter ke glyph hasil segmentasi.

Aturan pemetaan (penting)
-------------------------
Jumlah glyph hasil segmentasi HARUS sama dengan jumlah karakter yang
diketik (spasi diabaikan). Bila tidak sama, baris itu DILEWATI dan
ditandai "MISMATCH" — bukan ditebak. Menebak akan menggeser seluruh label
satu posisi dan merusak data latih.

Pemakaian
---------
    python label_lines.py                # mulai / lanjut
    python label_lines.py --stats        # progres
    python label_lines.py --list         # baris yang belum dilabeli
    python label_lines.py --export       # bangun ulang glyph_labels.csv

Kontrol keyboard
----------------
    Enter          simpan & lanjut ke baris berikutnya
    Tab            lihat glyph ber-ASCII (untuk memastikan segmentasi)
    Ctrl+Z         batalkan label terakhir
    Esc            keluar (progres tersimpan otomatis)

Keluaran
--------
    labels/lines_labeled.csv   image, line_index, text          (sumber kebenaran)
    labels/glyph_labels.csv    image, line_index, glyph_index,
                               label, top, left, height, width  (untuk latih)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OCR = ROOT / "ML" / "ocr_classical"
LABELS = HERE / "labels"
LINES_CSV = LABELS / "lines_labeled.csv"
GLYPHS_CSV = LABELS / "glyph_labels.csv"
CANDIDATES = HERE / "labeling_candidates.csv"

sys.path.insert(0, str(OCR))
sys.path.insert(0, str(HERE))
import dataset_source as DS  # noqa: E402
import deskew as DK  # noqa: E402
import paper as PPR  # noqa: E402
import preprocess as PP  # noqa: E402
import segment as SG  # noqa: E402

# Tinggi glyph yang dituju setelah pembesaran. Diukur dari citra yang
# segmentasinya paling stabil: pada ~20 px, baris terpisah bersih dan glyph
# masih cukup besar untuk dibaca mata saat dilabeli.
TARGET_GLYPH_H = 20.0

# Batas pembesaran agar citra sangat kecil tidak meledak memorinya.
MAX_SCALE = 3.0


def median_glyph_height(ink: np.ndarray) -> float:
    """Median tinggi komponen tinta yang ukurannya wajar untuk huruf."""
    num, _lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    heights = []
    for i in range(1, num):
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= 12 and 3 <= h <= 200:
            heights.append(h)
    return float(np.median(heights)) if heights else 0.0
MIN_LINE_H = 14

# Ukuran cadangan canvas bila winfo_* masih 0/1 sebelum layout selesai.
FALLBACK_W = 1180
FALLBACK_H = 420

# Batas ukuran pratinjau baris aktif. Tanpa batas, satu baris raksasa bisa
# mendorong canvas sampai mengecil.
MAX_PREVIEW_W = 1150
MAX_PREVIEW_H = 150

# Sudut deskew yang diterapkan per berkas, diisi oleh [load_mask] dan
# ditampilkan di GUI supaya pengguna tahu citra itu dimiringkan.
state_angles: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Sumber citra (dataset v10, hanya struk Indonesia)
# ---------------------------------------------------------------------------
def build_index(splits: tuple[str, ...] = ("train", "valid", "test")) -> dict[str, Path]:
    """Peta nama berkas -> path lengkap untuk citra Indonesia.

    Nama berkas di v10 sudah unik (Roboflow menambahkan hash), jadi satu
    kamus cukup untuk ketiga split. Struk Indonesia tersebar di train,
    valid, dan test, sehingga indeks dibangun menyeluruh.
    """
    index: dict[str, Path] = {}
    for split in splits:
        for p in DS.indonesian_files(split):
            index[p.name] = p
    return index


def resolve(name: str, index: dict[str, Path]) -> Path | None:
    return index.get(name)


# ---------------------------------------------------------------------------
# Pengolahan citra
# ---------------------------------------------------------------------------
def isolate_paper(img: np.ndarray) -> np.ndarray:
    """Potong ke komponen TERANG terbesar (kertas struk).

    Lebih tahan daripada `crop_paper` pada foto HP: struk adalah bidang
    terang terbesar, sedangkan latar (kain/meja) gelap. `crop_paper` punya
    syarat rasio aspek & fill yang ketat sehingga sering menolak foto yang
    sebenarnya bisa diisolasi. Fungsi ini tidak menolak — ia selalu memilih
    komponen terang terbesar, lalu potong ke situ.
    """
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    # Ambang tinggi: hanya kertas yang benar-benar terang yang lolos. Ambang
    # Otsu biasa ikut meloloskan meja kayu terang, sehingga teksturnya menjadi
    # "tinta" dan mencemari segmentasi.
    thr = max(int(np.percentile(blur, 70)), int(np.percentile(blur, 99) * 0.6))
    _, bright = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                              np.ones((9, 9), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              np.ones((15, 15), np.uint8))
    num, _lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    if num <= 1:
        return img

    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])

    # Tolak bila area terang ternyata sangat kecil (bukan kertas) atau
    # bentuknya tidak menyerupai struk (terlalu lebar/pendek).
    area_frac = (w * h) / max(img.shape[0] * img.shape[1], 1)
    if area_frac < 0.06:
        return img
    if w > 3.5 * max(h, 1):
        return img
    m = 4
    x0 = max(0, x - m)
    y0 = max(0, y - m)
    x1 = min(img.shape[1], x + w + m)
    y1 = min(img.shape[0], y + h + m)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return img
    return img[y0:y1, x0:x1]


def crop_to_content(gray: np.ndarray, ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Potong ke area bertinta saja (buang tepi kosong sisa rotasi)."""
    rows = np.nonzero(ink.sum(axis=1) > 0)[0]
    cols = np.nonzero(ink.sum(axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return gray, ink
    t, b = int(rows[0]), int(rows[-1])
    l, r = int(cols[0]), int(cols[-1])
    if b - t < 20 or r - l < 20:
        return gray, ink
    return gray[t:b + 1, l:r + 1], ink[t:b + 1, l:r + 1]

    if right <= left or bottom <= top:
        return gray, ink

    # Beri margin kecil agar glyph di tepi tidak terpotong.
    m = 3
    left = max(0, left - m)
    top = max(0, top - m)
    right = min(ink.shape[1] - 1, right + m)
    bottom = min(ink.shape[0] - 1, bottom + m)

    return (gray[top:bottom + 1, left:right + 1],
            ink[top:bottom + 1, left:right + 1])


def load_mask(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Kembalikan (gray, ink, scale) siap segmentasi.

    Termasuk **koreksi kemiringan**: struk yang difoto miring atau tertekuk
    membuat proyeksi horizontal gagal memisahkan baris, sehingga satu baris
    bisa tercampur dengan baris lain dan segmentasi glyph kacau. Deskew
    memakai variansi profil proyeksi (sudut yang membuat puncak/lembah
    paling tajam), rentang diperlebar ke +-25 derajat karena foto HP sering
    lebih miring daripada scan.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    # Isolasi kertas (komponen terang terbesar) lebih dulu, lalu crop_paper
    # sebagai penyempurna. Urutan ini penting: pada foto dengan latar gelap,
    # crop_paper sering menolak memotong sehingga latar ikut jadi tinta dan
    # seluruh struk melebur menjadi satu pita.
    img = isolate_paper(img)
    img, _ = PPR.crop_paper(img, enabled=True)

    # Skala dipilih dari TINGGI GLYPH terukur, bukan dari ukuran citra.
    # Ukuran citra tidak menentukan besar huruf (struk kecil bisa difoto
    # penuh frame), sedangkan tinggi glyph langsung menentukan apakah baris
    # bisa dipisah. Satu kali binarisasi cukup untuk mengukurnya.
    ink0 = PP.prepare(img, "sauvola")
    med_h = median_glyph_height(ink0)
    scale = 1.0
    if med_h > 0:
        scale = TARGET_GLYPH_H / med_h
        scale = max(1.0, min(MAX_SCALE, scale))
        scale = round(scale * 2) / 2          # 0.5 langkah agar deterministik
    if scale > 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    ink = PP.prepare(img, "sauvola")
    img, ink = crop_to_content(img, ink)

    # Koreksi kemiringan: struk miring/tertekuk membuat proyeksi horizontal
    # gagal memisahkan baris. Rentang diperlebar ke +-25 derajat karena foto
    # HP sering lebih miring daripada scan.
    angle = DK.estimate_skew(ink, max_angle=25.0)
    if abs(angle) >= 0.5:
        rotated_gray, rotated_ink, used = DK.auto_deskew_pair(
            img, ink, max_angle=25.0)
        # Terapkan hanya bila hasilnya tidak lebih buruk.
        if len(usable_lines(rotated_ink)) >= len(usable_lines(ink)):
            img, ink = rotated_gray, rotated_ink
            state_angles[path.name] = round(float(used), 1)
        else:
            state_angles[path.name] = 0.0
    else:
        state_angles[path.name] = 0.0

    return img, ink, scale


def split_tall_bands(
    profile: np.ndarray, bands: list[tuple[int, int]], med: float
) -> list[tuple[int, int]]:
    """Pecah pita yang lebih tinggi dari 2x median di lembah profil.

    Beberapa baris bisa menyatu menjadi satu pita (kertas tertekuk, jarak
    baris rapat, atau noise antar baris). Membuang pita itu berarti
    kehilangan semua baris di dalamnya; memecahnya di titik dengan tinta
    paling sedikit jauh lebih berguna.
    """
    out: list[tuple[int, int]] = []
    for a, b in bands:
        height = b - a + 1
        if med <= 0 or height <= 2.0 * med:
            out.append((a, b))
            continue

        parts = max(2, min(int(round(height / med)), 10))
        cuts: list[int] = []
        for k in range(1, parts):
            ideal = a + int(round(height * k / parts))
            margin = max(2, int(med * 0.45))
            lo = max(a + 1, ideal - margin)
            hi = min(b, ideal + margin)
            if hi <= lo:
                cuts.append(ideal)
                continue
            cuts.append(lo + int(np.argmin(profile[lo:hi + 1])))

        bounds = [a] + cuts + [b]
        for p, q in zip(bounds[:-1], bounds[1:]):
            if q - p + 1 >= MIN_LINE_H:
                out.append((p, q))
    return out


def usable_lines(ink: np.ndarray) -> list[tuple[int, int, int]]:
    """Baris yang layak dilabeli: (top, bottom, n_glyph).

    Menyaring dengan ukuran RELATIF terhadap median baris pada citra itu,
    bukan ambang tetap. Alasannya: tinggi baris bergantung pada skala foto,
    sehingga ambang mutlak selalu salah untuk sebagian citra.
    """
    bands = SG.segment_lines(ink)
    if not bands:
        return []

    heights = [b - a + 1 for a, b in bands]
    med = float(np.median(heights))
    if med <= 0:
        med = float(MIN_LINE_H)

    profile = (ink > 0).sum(axis=1).astype(np.float32)
    bands = split_tall_bands(profile, bands, med)

    out = []
    for lb in bands:
        height = lb[1] - lb[0] + 1
        if height < max(MIN_LINE_H, 0.55 * med):
            continue
        if height > 2.0 * med:
            continue
        n = len(SG.segment_characters(ink, lb))
        if n < 5 or n > 90:
            continue
        out.append((lb[0], lb[1], n))
    return out


# ---------------------------------------------------------------------------
# Penyimpanan
# ---------------------------------------------------------------------------
def load_labels() -> dict[tuple[str, int], str]:
    if not LINES_CSV.exists():
        return {}
    result = {}
    with LINES_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            result[(row["image"], int(row["line_index"]))] = row["text"]
    return result


def save_labels(labels: dict[tuple[str, int], str]) -> None:
    LABELS.mkdir(parents=True, exist_ok=True)
    with LINES_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "line_index", "text"])
        for (img, idx), text in sorted(labels.items()):
            w.writerow([img, idx, text])


def export_glyphs(labels: dict[tuple[str, int], str]) -> tuple[int, int]:
    """Petakan teks baris ke glyph; tulis glyph_labels.csv."""
    rows, ok, mismatch = [], 0, 0
    by_image: dict[str, list[tuple[int, str]]] = {}
    for (img, idx), text in labels.items():
        by_image.setdefault(img, []).append((idx, text))

    index = build_index()
    for img_name, items in sorted(by_image.items()):
        path = resolve(img_name, index)
        if path is None:
            continue
        try:
            gray, ink, _ = load_mask(path)
        except Exception:  # noqa: BLE001
            continue
        lines = usable_lines(ink)

        for idx, text in sorted(items):
            if idx >= len(lines):
                continue
            top, bottom, _ = lines[idx]
            boxes = SG.segment_characters(ink, (top, bottom))
            expected = [c for c in text if not c.isspace()]
            if len(boxes) != len(expected):
                mismatch += 1
                continue
            for gi, (box, ch) in enumerate(zip(boxes, expected)):
                l, t, r, b = box
                rows.append([img_name, idx, gi, ch, t, l, b - t + 1, r - l + 1])
            ok += 1

    with GLYPHS_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "line_index", "glyph_index", "label",
                    "top", "left", "height", "width"])
        w.writerows(rows)
    return ok, mismatch


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def run_gui(start: int = 0) -> None:
    import tkinter as tk
    from tkinter import ttk

    LABELS.mkdir(parents=True, exist_ok=True)
    labels = load_labels()

    index = build_index()
    if not index:
        raise SystemExit(f"tidak ada citra struk Indonesia di {DS.DATASET}")
    print(f"sumber: {DS.DATASET.name} - {len(index)} citra struk Indonesia")

    # Urutkan citra: skor tertinggi lebih dulu (dari CSV kandidat).
    if CANDIDATES.exists():
        with CANDIDATES.open("r", encoding="utf-8") as fh:
            names = [r["file"] for r in csv.DictReader(fh)]
    else:
        names = sorted(index)
    names = [n for n in names if n in index]
    if not names:
        print("tidak ada citra yang bisa diproses")
        return

    # Citra dimuat SATU PER SATU, bukan sekaligus. Memuat 71 citra di muka
    # butuh ~10 menit (binarisasi Sauvola mahal), sedangkan pengguna hanya
    # butuh satu citra pada satu waktu.
    state: dict[str, object] = {
        "pos": max(0, min(start, len(names) - 1)),
        "name": None,
        "gray": None,
        "ink": None,
        "lines": [],
        "line_i": 0,
    }
    undo: list[tuple[str, int]] = []
    image_skips: set[str] = set()

    root = tk.Tk()
    root.title("Pelabelan Baris Struk Thermal")
    root.geometry("1280x900")

    header = ttk.Frame(root, padding=(10, 8))
    header.pack(fill="x")
    progress = ttk.Label(header, text="", font=("Segoe UI", 10, "bold"))
    progress.pack(side="left")
    ttk.Label(
        header,
        foreground="#555",
        text="   Ketik teks baris yang disorot (spasi diabaikan). "
             "Enter = simpan & lanjut.",
    ).pack(side="left")

    # Canvas dengan tinggi TETAP supaya pratinjau tidak bisa mendorongnya.
    canvas = tk.Canvas(root, background="#f0f0f0", highlightthickness=0,
                       height=380)
    canvas.pack(fill="x", padx=10, pady=(4, 0))

    line_frame = ttk.LabelFrame(root, text="Baris aktif (diperbesar)", padding=6)
    line_frame.pack(fill="x", padx=10)
    line_label = ttk.Label(line_frame, text="")
    line_label.pack()

    entry_row = ttk.Frame(root, padding=(10, 8))
    entry_row.pack(fill="x")
    entry = ttk.Entry(entry_row, font=("Consolas", 18))
    entry.pack(side="left", fill="x", expand=True)
    info = ttk.Label(root, text="", foreground="#b45309", wraplength=1240)
    info.pack(fill="x", padx=10)
    info2 = ttk.Label(root, text="", foreground="#555", wraplength=1240)
    info2.pack(fill="x", padx=10)

    photos: dict[str, object] = {"photo": None, "line": None}

    def load_current() -> bool:
        """Muat citra pada posisi sekarang; False bila tidak ada yang bisa."""
        while int(state["pos"]) < len(names):
            name = names[int(state["pos"])]
            if name in image_skips:
                state["pos"] = int(state["pos"]) + 1
                continue
            try:
                gray, ink, scale = load_mask(index[name])
            except Exception as exc:  # noqa: BLE001
                print(f"  gagal memuat {name[:40]}: {exc}")
                state["pos"] = int(state["pos"]) + 1
                continue
            lines = usable_lines(ink)
            if not lines:
                print(f"  tidak ada baris layak pada {name[:40]}")
                state["pos"] = int(state["pos"]) + 1
                continue
            state["name"] = name
            state["gray"] = gray
            state["ink"] = ink
            state["lines"] = lines
            state["line_i"] = 0
            print(f"  {name[:46]}  skala={scale:.2f}  baris={len(lines)}  "
                  f"sudut={state_angles.get(name, 0.0)}")
            return True
        return False

    def next_line() -> None:
        state["line_i"] = int(state["line_i"]) + 1
        render()

    def render() -> None:
        lines = state["lines"]
        if state["name"] is None or int(state["line_i"]) >= len(lines):  # type: ignore[arg-type]
            if not load_current():
                progress.config(
                    text="Selesai - semua citra dalam antrean sudah diproses")
                save_labels(labels)
                ok, mm = export_glyphs(labels)
                info.config(text=f"Selesai. Baris dilabeli: {len(labels)} | "
                                 f"ekspor {ok} baris cocok, {mm} mismatch")
                return

        name = str(state["name"])
        gray = state["gray"]
        ink = state["ink"]
        i = int(state["line_i"])
        lines = state["lines"]
        top, bottom, n = lines[i]  # type: ignore[index]
        angle = state_angles.get(name, 0.0)

        progress.config(
            text=f"citra {int(state['pos']) + 1}/{len(names)}  |  "
                 f"baris {i + 1}/{len(lines)}  |  tersimpan {len(labels)}  |  "
                 f"{name[:22]}"
        )
        info2.config(text=f"berkas: {name}   sudut: {angle:+.1f} derajat")

        warnings = []
        if abs(angle) >= 3.0:
            warnings.append(f"citra dimiringkan {angle:.1f} derajat")
        if (bottom - top + 1) > 90:
            warnings.append("baris sangat tinggi - mungkin 2 baris menyatu")
        if n > 70:
            warnings.append(f"{n} glyph - segmentasi mungkin kacau")
        msg = f"glyph terdeteksi: {n}   |   baris y={top}-{bottom}"
        if warnings:
            msg += "   |   PERIKSA: " + "; ".join(warnings)
        info.config(text=msg)

        # Ukuran canvas diambil setelah layout selesai. Sebelum itu
        # winfo_width() bisa bernilai 1 sehingga skala gambar menjadi nol.
        root.update_idletasks()
        win_w = max(canvas.winfo_width(), FALLBACK_W)
        win_h = max(canvas.winfo_height(), FALLBACK_H)
        scale = min(win_w / gray.shape[1], win_h / gray.shape[0])
        disp = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
        canvas.delete("all")
        photos["photo"] = ImageTk.PhotoImage(Image.fromarray(disp))
        canvas.create_image(0, 0, anchor="nw", image=photos["photo"])
        canvas.create_rectangle(0, int(top * scale), disp.shape[1],
                                int(bottom * scale), outline="#e5484d",
                                width=2)

        # Pratinjau baris, tinggi DIBATASI agar tidak mendorong canvas.
        cols = np.nonzero((ink[top:bottom + 1, :] > 0).sum(axis=0) > 0)[0]
        if len(cols):
            x0, x1 = int(cols[0]), int(cols[-1])
            crop = gray[top:bottom + 1, x0:x1 + 1]
            if crop.size:
                ch, cw = crop.shape[:2]
                factor = min(MAX_PREVIEW_W / max(cw, 1),
                             MAX_PREVIEW_H / max(ch, 1), 8.0)
                factor = max(factor, 1.0)
                big = cv2.resize(crop, None, fx=factor, fy=factor,
                                 interpolation=cv2.INTER_CUBIC)
                photos["line"] = ImageTk.PhotoImage(Image.fromarray(big))
                line_label.config(image=photos["line"])

        entry.delete(0, "end")
        entry.focus_set()

    def commit(text: str) -> None:
        name = state["name"]
        if name is None:
            return
        clean = "".join(text.split())
        if not clean:
            return
        i = int(state["line_i"])
        n = state["lines"][i][2]  # type: ignore[index]
        if len(clean) != n:
            info.config(
                text=f"PERINGATAN: kamu mengetik {len(clean)} karakter, "
                     f"segmentasi mendeteksi {n} glyph. Baris disimpan dan "
                     f"akan ditandai MISMATCH saat ekspor."
            )
        labels[(str(name), i)] = clean
        save_labels(labels)
        undo.append((str(name), i))
        next_line()

    def skip_image() -> None:
        name = state["name"]
        if name is not None:
            image_skips.add(str(name))
            print(f"  dilewati (kualitas buruk): {name[:40]}")
        state["pos"] = int(state["pos"]) + 1
        state["name"] = None
        state["lines"] = []
        state["line_i"] = 0
        render()

    def prev_image() -> None:
        state["pos"] = max(0, int(state["pos"]) - 1)
        state["name"] = None
        state["lines"] = []
        state["line_i"] = 0
        render()

    def do_undo() -> None:
        if not undo:
            return
        name, idx = undo.pop()
        labels.pop((name, idx), None)
        save_labels(labels)
        if name in names:
            state["pos"] = names.index(name)
            state["name"] = None
            state["lines"] = []
            state["line_i"] = idx
            render()

    def show_ascii() -> None:
        name = state["name"]
        if name is None:
            return
        ink = state["ink"]
        i = int(state["line_i"])
        top, bottom, _ = state["lines"][i]  # type: ignore[index]
        boxes = SG.segment_characters(ink, (top, bottom))
        print(f"\n=== {name} baris {i}: {len(boxes)} glyph ===")
        for gi, (l, t, r, b) in enumerate(boxes):
            sub = (ink[t:b + 1, l:r + 1] > 0)
            art = "".join("".join("#" if v else "." for v in row)
                          for row in sub)
            print(f"  [{gi}] {r - l + 1}x{b - t + 1}: {art[:60]}")

    btns = ttk.Frame(entry_row)
    btns.pack(side="left", padx=(8, 0))
    ttk.Button(btns, text="Simpan",
               command=lambda: commit(entry.get())).pack(side="left")
    ttk.Button(btns, text="Lewati baris",
               command=next_line).pack(side="left", padx=4)
    ttk.Button(btns, text="Lewati citra",
               command=skip_image).pack(side="left")
    ttk.Button(btns, text="< citra", command=prev_image,
               width=8).pack(side="left", padx=4)
    ttk.Button(btns, text="Undo", command=do_undo).pack(side="left")
    ttk.Button(btns, text="ASCII", command=show_ascii).pack(side="left", padx=4)

    entry.bind("<Return>", lambda _e: commit(entry.get()))
    root.bind("<Control-z>", lambda _e: do_undo())
    root.bind("<Tab>", lambda _e: (show_ascii(), "break")[1])

    def on_close() -> None:
        save_labels(labels)
        ok, mm = export_glyphs(labels)
        print(f"progres disimpan: {len(labels)} baris; "
              f"ekspor {ok} ok, {mm} mismatch")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Escape>", lambda _e: on_close())
    root.after(80, render)
    root.mainloop()


# ---------------------------------------------------------------------------
def print_stats() -> None:
    labels = load_labels()
    counts = DS.count_by_split()
    print(f"dataset            : {DS.DATASET.name}")
    for split, (n_id, n_all) in counts.items():
        print(f"  {split:15s}: {n_id:4d} Indonesia / {n_all:5d} total")
    total_id = sum(n for n, _ in counts.values())
    print(f"citra Indonesia    : {total_id}")
    print(f"baris sudah dilabeli: {len(labels)}")
    print(f"file                : {LINES_CSV}")
    if GLYPHS_CSV.exists():
        with GLYPHS_CSV.open("r", encoding="utf-8") as fh:
            n = sum(1 for _ in fh) - 1
        print(f"glyph berlabel      : {n}  ({GLYPHS_CSV})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pelabelan baris struk thermal")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    if args.stats:
        print_stats()
        return
    if args.export:
        labels = load_labels()
        ok, mm = export_glyphs(labels)
        print(f"ekspor: {ok} baris cocok, {mm} baris mismatch -> {GLYPHS_CSV}")
        return
    run_gui(args.start)


if __name__ == "__main__":
    main()
