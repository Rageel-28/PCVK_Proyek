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

MAX_DIM = 1400
MIN_LINE_H = 14

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
    img, _ = PPR.crop_paper(img, enabled=True)

    scale = 1.0
    longest = max(img.shape)
    if longest > MAX_DIM:
        scale = MAX_DIM / longest
    elif longest < 1000:
        scale = 1000 / longest
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    ink = PP.prepare(img, "sauvola")

    # Koreksi kemiringan: struk miring/tertekuk membuat proyeksi horizontal
    # gagal memisahkan baris. Rentang diperlebar ke +-25 derajat karena foto
    # HP sering lebih miring daripada scan.
    angle = DK.estimate_skew(ink, max_angle=25.0)
    if abs(angle) >= 0.5:
        img, ink, _used = DK.auto_deskew_pair(img, ink, max_angle=25.0)
        state_angles[path.name] = round(float(_used), 1)
    else:
        state_angles[path.name] = 0.0

    return img, ink, scale


def usable_lines(ink: np.ndarray) -> list[tuple[int, int, int]]:
    """Baris yang layak dilabeli: (top, bottom, n_glyph)."""
    out = []
    for lb in SG.segment_lines(ink):
        if lb[1] - lb[0] + 1 < MIN_LINE_H:
            continue
        n = len(SG.segment_characters(ink, lb))
        if n >= 4:
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
    from tkinter import messagebox, ttk

    LABELS.mkdir(parents=True, exist_ok=True)
    labels = load_labels()

    # Indeks citra struk Indonesia dari dataset v10 (train+valid+test).
    index = build_index()
    if not index:
        raise SystemExit(f"tidak ada citra struk Indonesia di {DS.DATASET}")
    print(f"sumber: {DS.DATASET.name} — {len(index)} citra struk Indonesia")

    # Urutkan citra: yang skornya tinggi lebih dulu (dari CSV kandidat).
    if CANDIDATES.exists():
        with CANDIDATES.open("r", encoding="utf-8") as fh:
            names = [r["file"] for r in csv.DictReader(fh)]
    else:
        names = sorted(index)
    # Hanya berkas yang benar-benar ada di indeks Indonesia.
    names = [n for n in names if n in index]

    # Bangun antrean baris yang belum dilabeli.
    print("menyiapkan antrean...", flush=True)
    queue: list[tuple[str, int, tuple[int, int, int]]] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, name in enumerate(names):
        try:
            gray, ink, _ = load_mask(index[name])
        except Exception:  # noqa: BLE001
            continue
        cache[name] = (gray, ink)
        for idx, (top, bottom, n) in enumerate(usable_lines(ink)):
            if (name, idx) not in labels:
                queue.append((name, idx, (top, bottom, n)))
        if i % 40 == 0:
            print(f"  {i}/{len(names)} citra, {len(queue)} baris menunggu",
                  flush=True)
        if len(queue) > 4000:
            break

    if not queue:
        print("semua baris sudah dilabeli")
        return
    print(f"antrean: {len(queue)} baris dari {len(cache)} citra")

    cursor = min(start, len(queue) - 1)
    undo: list[tuple[str, int]] = []

    # Berkas yang ditolak karena kualitas buruk (miring/tertekuk/buram).
    image_skips: set[str] = set()

    root = tk.Tk()
    root.title("Pelabelan Baris Struk Thermal")
    root.geometry("1280x860")

    header = ttk.Frame(root, padding=(10, 8))
    header.pack(fill="x")
    progress = ttk.Label(header, text="", font=("Segoe UI", 10, "bold"))
    progress.pack(side="left")
    ttk.Label(header, foreground="#555",
              text="  Ketik teks baris yang disorot (spasi diabaikan). "
                   "Enter = simpan & lanjut.").pack(side="left")

    canvas = tk.Canvas(root, background="#f0f0f0", highlightthickness=0)
    canvas.pack(fill="both", expand=True, padx=10)

    line_frame = ttk.LabelFrame(root, text="Baris aktif (diperbesar)", padding=6)
    line_frame.pack(fill="x", padx=10)
    line_label = ttk.Label(line_frame, text="")
    line_label.pack()

    entry_row = ttk.Frame(root, padding=(10, 8))
    entry_row.pack(fill="x")
    entry = ttk.Entry(entry_row, font=("Consolas", 18))
    entry.pack(side="left", fill="x", expand=True)
    info = ttk.Label(root, text="", foreground="#b45309")
    info.pack(fill="x", padx=10)

    state = {"photo": None, "line_photo": None}

    def current():
        return queue[cursor]

    def render() -> None:
        if cursor >= len(queue):
            progress.config(text="Selesai — semua baris dalam antrean sudah dilabeli")
            save_labels(labels)
            ok, mm = export_glyphs(labels)
            messagebox.showinfo("Selesai",
                                f"Baris dilabeli: {len(labels)}\n"
                                f"Ekspor glyph: {ok} baris cocok, {mm} mismatch")
            root.destroy()
            return

        name, idx, (top, bottom, n) = current()
        gray, ink = cache[name]
        angle = state_angles.get(name, 0.0)
        progress.config(text=f"Baris {cursor + 1}/{len(queue)}  |  "
                             f"tersimpan {len(labels)}  |  {name[:26]}")

        # Peringatan kualitas: kemiringan besar dan/atau baris terlalu tinggi
        # (ciri segmentasi kacau) ditandai agar pengguna waspada. Baris yang
        # tingginya jauh di atas normal biasanya gabungan dua baris.
        warnings = []
        if abs(angle) >= 3.0:
            warnings.append(f"citra dimiringkan {angle:.1f} deg")
        if (bottom - top + 1) > 90:
            warnings.append("baris ini sangat tinggi - mungkin 2 baris menyatu")
        if n > 70:
            warnings.append(f"{n} glyph dalam satu baris - mungkin kacau")
        msg = f"glyph terdeteksi: {n}  |  baris y={top}-{bottom}"
        if warnings:
            msg += "   |   PERIKSA: " + "; ".join(warnings)
        info.config(text=msg)

        win_w = canvas.winfo_width() or 1240
        win_h = canvas.winfo_height() or 520
        scale = min(win_w / gray.shape[1], win_h / gray.shape[0], 1.0)
        disp = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
        canvas.delete("all")
        state["photo"] = ImageTk.PhotoImage(Image.fromarray(disp))
        canvas.create_image(0, 0, anchor="nw", image=state["photo"])
        canvas.create_rectangle(0, int(top * scale), disp.shape[1],
                                int(bottom * scale), outline="#e5484d", width=2)

        # Perbesar baris aktif untuk pembacaan
        cols = np.nonzero((ink[top:bottom + 1, :] > 0).sum(axis=0) > 0)[0]
        if len(cols):
            x0, x1 = int(cols[0]), int(cols[-1])
            crop = gray[top:bottom + 1, x0:x1 + 1]
            if crop.size:
                factor = min(6.0, 1100 / max(crop.shape[1], 1))
                big = cv2.resize(crop, None, fx=factor, fy=factor,
                                 interpolation=cv2.INTER_CUBIC)
                state["line_photo"] = ImageTk.PhotoImage(Image.fromarray(big))
                line_label.config(image=state["line_photo"])

        entry.delete(0, "end")
        entry.focus_set()

    def commit(text: str) -> None:
        nonlocal cursor
        clean = "".join(text.split())
        if not clean:
            return
        name, idx, (_, _, n) = current()
        if len(clean) != n:
            info.config(
                text=f"PERINGATAN: kamu mengetik {len(clean)} karakter, "
                     f"segmentasi mendeteksi {n} glyph. Baris tetap disimpan "
                     f"dan akan ditandai MISMATCH saat ekspor."
            )
        labels[(name, idx)] = clean
        save_labels(labels)
        undo.append((name, idx))
        cursor += 1
        render()

    def skip() -> None:
        nonlocal cursor
        cursor = min(cursor + 1, len(queue))
        render()

    def skip_image() -> None:
        """Lewati SELURUH citra ini (kualitas buruk / terlalu miring).

        Menolak citra adalah keputusan yang sah: label yang salah lebih
        merusak data latih daripada kekurangan beberapa citra.
        """
        nonlocal cursor
        if cursor >= len(queue):
            return
        name = queue[cursor][0]
        skipped = sum(1 for (n, _, _) in queue if n == name)
        for i, (n, _, _) in enumerate(queue):
            if n == name:
                cursor = max(cursor, i)
        # Maju ke baris pertama citra berikutnya.
        i = cursor
        while i < len(queue) and queue[i][0] == name:
            i += 1
        cursor = min(i, len(queue))
        image_skips.add(name)
        info.config(text=f"citra dilewati: {name[:40]} ({skipped} baris)")
        render()

    def do_undo() -> None:
        nonlocal cursor
        if not undo:
            return
        key = undo.pop()
        labels.pop(key, None)
        save_labels(labels)
        for i, (n, idx, _) in enumerate(queue):
            if (n, idx) == key:
                cursor = i
                break
        render()

    def show_ascii() -> None:
        """Tampilkan baris sebagai ASCII untuk memastikan segmentasi."""
        name, idx, (top, bottom, n) = current()
        _, ink = cache[name]
        boxes = SG.segment_characters(ink, (top, bottom))
        print(f"\n=== {name} baris {idx}: {len(boxes)} glyph ===")
        for gi, (l, t, r, b) in enumerate(boxes):
            sub = (ink[t:b + 1, l:r + 1] > 0)
            chars = "".join("".join("#" if v else "." for v in row) for row in sub)
            print(f"  [{gi}] {r - l + 1}x{b - t + 1}: "
                  f"{chars[:56]}")

    btns = ttk.Frame(entry_row)
    btns.pack(side="left", padx=(8, 0))
    ttk.Button(btns, text="Simpan", command=lambda: commit(entry.get())).pack(side="left")
    ttk.Button(btns, text="Lewati baris", command=skip).pack(side="left", padx=4)
    ttk.Button(btns, text="Lewati citra", command=skip_image).pack(side="left")
    ttk.Button(btns, text="Undo", command=do_undo).pack(side="left", padx=4)
    ttk.Button(btns, text="ASCII", command=show_ascii).pack(side="left")

    entry.bind("<Return>", lambda _e: commit(entry.get()))
    root.bind("<Control-z>", lambda _e: do_undo())
    root.bind("<Tab>", lambda _e: (show_ascii(), "break")[1])

    def on_close() -> None:
        save_labels(labels)
        ok, mm = export_glyphs(labels)
        print(f"progres disimpan: {len(labels)} baris; ekspor {ok} ok, {mm} mismatch")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Escape>", lambda _e: on_close())
    root.after(60, render)
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
