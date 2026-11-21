"""Segmentasi glyph (karakter) di dalam satu kata.

Metode klasik:

1. **Connected components** pada kata → kandidat glyph.
2. **Gabung** komponen yang vertikalnya bertumpuk dan celah horizontalnya
   sangat kecil (badan huruf + titik pada ``i``/``j``/``:``).
3. **Pecah** komponen yang terlalu lebar dibanding median lebar glyph, di
   lembah profil vertikalnya. Ini menangani huruf yang saling menyentuh
   (``rn`` yang tampak seperti ``m``).

Hasil akhirnya berupa ``Glyph`` dengan kotak tight di koordinat citra penuh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import ocr.config as C


@dataclass(frozen=True)
class Glyph:
    """Kotak satu glyph di koordinat citra penuh."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.top : self.bottom, self.left : self.right]


def _components(band: np.ndarray) -> list[Glyph]:
    """Connected components 8-konektivitas pada pita biner."""
    import cv2

    mask = (np.asarray(band) == C.INK).astype(np.uint8)
    if not mask.any():
        return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    out: list[Glyph] = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 2:
            continue
        out.append(
            Glyph(left=int(x), top=int(y), right=int(x + w), bottom=int(y + h))
        )
    return out


def _vertical_overlap(a: Glyph, b: Glyph) -> float:
    """Rasio tumpang-tindih vertikal terhadap tinggi yang lebih pendek."""
    top = max(a.top, b.top)
    bottom = min(a.bottom, b.bottom)
    overlap = max(0, bottom - top)
    shortest = min(a.height, b.height)
    return overlap / shortest if shortest else 0.0


def _merge_marks(
    glyphs: list[Glyph],
    gap_limit: int,
    median_width: float,
    median_height: float = 0.0,
) -> list[Glyph]:
    """Gabung komponen yang sebenarnya satu glyph.

    Contoh nyata: ``%`` terdiri dari dua lingkaran dan satu garis miring;
    kalau tidak digabung, satu karakter jadi tiga dan decoder menghasilkan
    tiga huruf palsu.

    Syaratnya sengaja sangat ketat — komponen harus benar-benar **bersentuhan**
    (celah <= 0). Pada teks kecil, celah antar huruf bisa hanya 1-2 piksel;
    kalau ambangnya longgar, huruf-huruf bertetangga ikut melebur dan
    segmentasi hancur. Huruf yang benar-benar menyentuh ditangani tahap
    pemecahan, bukan tahap penggabungan.
    """
    if not glyphs:
        return []
    ordered = sorted(glyphs, key=lambda g: g.left)
    limit = median_width * 1.15 if median_width > 0 else float("inf")

    merged: list[Glyph] = [ordered[0]]
    for glyph in ordered[1:]:
        prev = merged[-1]
        gap = glyph.left - prev.right
        union_width = max(prev.right, glyph.right) - min(prev.left, glyph.left)

        # Komponen yang **seluruhnya berada di dalam** rentang-x tetangganya
        # tidak mungkin huruf lain: itu bintik di dalam satu huruf (lubang
        # ``0`` yang terpisah, titik ``i``). Menggabungkannya tidak melebarkan
        # kotak, jadi batas lebar tidak relevan. Tanpa pengecualian ini,
        # ``57.000`` terbaca ``57.0.0.0.`` karena setiap nol 1.2x lebih lebar
        # dari median sehingga gabungannya ditolak batas lebar.
        contained = glyph.left >= prev.left and glyph.right <= prev.right
        touching = (
            gap <= 0
            and _vertical_overlap(prev, glyph) > 0.25
            and (contained or union_width <= limit)
        )
        if touching or _stacked_mark(prev, glyph, median_height):
            merged[-1] = Glyph(
                left=min(prev.left, glyph.left),
                top=min(prev.top, glyph.top),
                right=max(prev.right, glyph.right),
                bottom=max(prev.bottom, glyph.bottom),
            )
        else:
            merged.append(glyph)
    return merged


def _stacked_mark(a: Glyph, b: Glyph, median_height: float) -> bool:
    """True bila satu komponen menumpuk tepat di atas/bawah komponen lain.

    Dua kasus nyata butuh aturan ini:

    - Titik pada ``i``/``j`` dan dua titik pada ``:``. Titiknya terpisah dari
      badan huruf, jadi tidak pernah bersentuhan (celah > 0) dan tidak punya
      tumpang-tindih vertikal. Tanpa aturan ini, ``:`` menjadi dua glyph dan
      satu huruf ``i`` menjadi dua huruf.
    - Huruf yang terbelah baris tipis akibat resampling (``2`` pecah menjadi
      bagian atas dan bawah). Gejalanya sama: dua komponen bertumpuk dengan
      sumbu-x hampir sama.

    Syaratnya ketat supaya dua huruf berbeda tidak ikut melebur: tumpang
    tindih horizontal harus tinggi, jarak vertikalnya kecil, dan gabungannya
    tidak boleh jauh lebih tinggi dari satu huruf.
    """
    if median_height <= 0:
        return False

    left = max(a.left, b.left)
    right = min(a.right, b.right)
    overlap = right - left
    if overlap <= 0:
        return False
    smaller = min(a.width, b.width)
    if smaller <= 0 or overlap / smaller < 0.6:
        return False

    vertical_gap = max(b.top - a.bottom, a.top - b.bottom)
    if vertical_gap > max(1.0, 0.12 * median_height):
        return False

    # Gabungan tidak boleh jauh lebih tinggi dari komponen tertinggi. Batas
    # ini diukur terhadap komponen itu sendiri, bukan median: pada kata yang
    # hanya berisi satu huruf (``2`` pada ``2 X 9.800``) yang pecah dua,
    # median-nya justru setengah tinggi huruf sehingga batas median akan
    # menolak penggabungan yang benar.
    union_height = max(a.bottom, b.bottom) - min(a.top, b.top)
    tallest = max(a.height, b.height)
    return union_height <= 1.6 * tallest + vertical_gap


def split_wide_glyph(
    band: np.ndarray,
    glyph: Glyph,
    reference_width: float = 0.0,
    valley_ratio: float = 0.25,
) -> list[Glyph]:
    """Pecah glyph lebar di lembah profil vertikalnya.

    Dipanggil hanya untuk glyph yang jauh lebih lebar dari median. Titik
    potong dipilih di kolom dengan tinta paling sedikit.

    ``valley_ratio`` mengatur seberapa dalam lembah harus terlihat. Untuk
    komponen selebar 1-2 huruf dipakai nilai ketat (0.25) supaya huruf lebar
    seperti ``W``/``M`` tidak ikut terbelah di tengkuknya. Untuk komponen
    selebar 3 huruf atau lebih, lembah antar huruf pada huruf tebal bisa
    dangkal, jadi ambangnya dilonggarkan oleh pemanggil.
    """
    mask = (np.asarray(band) == C.INK).astype(np.uint8)
    if glyph.width < 6 or glyph.height < 4:
        return [glyph]

    crop = mask[glyph.top : glyph.bottom, glyph.left : glyph.right]
    if crop.size == 0:
        return [glyph]
    profile = crop.sum(axis=0).astype(np.float64)
    if profile.size < 6:
        return [glyph]

    margin = max(1, profile.size // 5)
    inner = profile[margin : profile.size - margin]
    if inner.size == 0:
        return [glyph]
    cut = int(np.argmin(inner)) + margin

    has_valley = (
        profile[cut] <= valley_ratio * float(profile.max())
        and cut >= 0.15 * profile.size
        and (profile.size - cut) >= 0.15 * profile.size
    )

    if not has_valley:
        # Tidak ada lembah: huruf tebal yang saling bersentuhan (mis. ``TUNAI``).
        # Pada font monospace lebar tiap huruf sama, jadi komponen bisa dibagi
        # rata — tapi hanya kalau lebarnya memang kelipatan bulat dari lebar
        # acuan. Tanpa syarat itu, pembagian rata menghasilkan potongan yang
        # tidak sejajar huruf dan classifier membaca sampah.
        return _split_evenly(glyph, reference_width)

    left_part = Glyph(
        left=glyph.left, top=glyph.top, right=glyph.left + cut, bottom=glyph.bottom
    )
    right_part = Glyph(
        left=glyph.left + cut, top=glyph.top, right=glyph.right, bottom=glyph.bottom
    )
    parts = []
    for part in (left_part, right_part):
        tight = _tighten(mask, part)
        if tight is not None:
            parts.append(tight)
    return parts if len(parts) == 2 else [glyph]


def _split_evenly(glyph: Glyph, reference_width: float) -> list[Glyph]:
    """Bagi glyph lebar menjadi bagian selebar acuan.

    Dipakai hanya ketika tidak ada lembah sama sekali — ciri huruf tebal yang
    menyatu. Syaratnya lebar harus mendekati kelipatan bulat dari lebar acuan;
    kalau tidak, potongannya tidak sejajar huruf dan hasilnya sampah.
    """
    if reference_width <= 0:
        return [glyph]

    ratio = glyph.width / reference_width
    count = int(round(ratio))
    if count < 2:
        return [glyph]
    if abs(ratio - count) > 0.3:
        return [glyph]

    step = glyph.width / count
    parts: list[Glyph] = []
    for i in range(count):
        left = glyph.left + int(round(i * step))
        right = glyph.left + int(round((i + 1) * step))
        if right - left < 2:
            return [glyph]
        parts.append(
            Glyph(left=left, top=glyph.top, right=right, bottom=glyph.bottom)
        )
    return parts


def _tighten(mask: np.ndarray, glyph: Glyph) -> Glyph | None:
    """Perketat kotak ke bbox tinta sebenarnya di dalamnya."""
    crop = mask[glyph.top : glyph.bottom, glyph.left : glyph.right]
    if crop.size == 0 or not crop.any():
        return None
    rows = np.flatnonzero(crop.any(axis=1))
    cols = np.flatnonzero(crop.any(axis=0))
    return Glyph(
        left=glyph.left + int(cols[0]),
        top=glyph.top + int(rows[0]),
        right=glyph.left + int(cols[-1]) + 1,
        bottom=glyph.top + int(rows[-1]) + 1,
    )


def _split_to_width(
    band: np.ndarray,
    glyph: Glyph,
    reference_width: float,
    split_ratio: float,
    depth: int = 0,
) -> list[Glyph]:
    """Pecah glyph lebar berulang sampai semua bagian selebar glyph biasa.

    Pemecahan harus **rekursif**: huruf tebal pada baris ``TUNAI``/``KEMBALI``
    saling bersentuhan sehingga lima huruf menjadi satu komponen selebar
    lima huruf. Memecah sekali hanya menghasilkan dua bagian yang masih
    terlalu lebar, dan decoder hanya melihat dua huruf.
    """
    if reference_width <= 0 or depth >= 6:
        return [glyph]
    if glyph.width <= reference_width * split_ratio:
        return [glyph]

    # Komponen yang jauh lebih lebar pasti berisi beberapa huruf, sehingga
    # lembah antar huruf boleh dangkal. Komponen yang hanya sedikit lebar
    # bisa jadi satu huruf lebar (``W``), jadi lembahnya harus dalam.
    multiple = glyph.width / reference_width
    valley_ratio = 0.55 if multiple >= 2.5 else 0.25

    parts = split_wide_glyph(band, glyph, reference_width, valley_ratio)
    if len(parts) <= 1:
        return [glyph]

    out: list[Glyph] = []
    for part in parts:
        out.extend(
            _split_to_width(band, part, reference_width, split_ratio, depth + 1)
        )
    return out


def segment_glyphs(
    band: np.ndarray,
    band_top: int = 0,
    band_left: int = 0,
    split_ratio: float | None = None,
    reference_width: float = 0.0,
) -> list[Glyph]:
    """Segmentasi glyph dalam satu pita kata/baris.

    Args:
        band: citra biner (255 = tinta).
        band_top: offset vertikal pita di citra penuh.
        band_left: offset horizontal pita di citra penuh.
        split_ratio: pengali lebar referensi untuk memicu pemecahan glyph.
        reference_width: lebar glyph acuan. Sebaiknya diambil dari **seluruh
            baris**, bukan dari kata ini saja: pada kata tebal, semua glyph
            menyatu sehingga median lebar kata itu menyesatkan.

    Returns:
        Daftar ``Glyph`` terurut kiri ke kanan, koordinat citra penuh.
    """
    if split_ratio is None:
        split_ratio = C.GLYPH_SPLIT_RATIO

    arr = np.asarray(band)
    if arr.ndim != 2 or arr.size == 0 or not (arr == C.INK).any():
        return []

    components = _components(arr)
    if not components:
        return []

    heights = [g.height for g in components]
    widths = [g.width for g in components]
    median_height = float(np.median(heights))
    median_width = float(np.median(widths))

    gap_limit = max(1, int(round(median_height * 0.12)))
    components = _merge_marks(
        components, gap_limit, median_width, median_height
    )

    # Referensi lebar untuk memicu pemecahan. Lebar dari pemanggil (biasanya
    # seluruh baris) jauh lebih andal daripada median kata ini saja.
    if reference_width <= 0:
        if len(components) >= 3:
            reference_width = median_width
        else:
            reference_width = 0.6 * median_height

    glyphs: list[Glyph] = []
    for glyph in components:
        glyphs.extend(
            _split_to_width(arr, glyph, reference_width, split_ratio)
        )

    glyphs.sort(key=lambda g: g.left)
    return [
        Glyph(
            left=g.left + band_left,
            top=g.top + band_top,
            right=g.right + band_left,
            bottom=g.bottom + band_top,
        )
        for g in glyphs
    ]
