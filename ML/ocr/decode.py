"""Decoder: beam search dengan model bahasa + koreksi leksikon.

Tiga sumber informasi digabung di sini:

1. **Classifier** — seberapa yakin glyph ini karakter tertentu.
2. **Model bahasa** — seberapa wajar rangkaian karakter itu di struk.
3. **Leksikon** — apakah kata ini mirip kata yang memang dikenal.

Bobot ``LM_WEIGHT`` menentukan seberapa besar konteks boleh membantah
classifier. Nilai 0 berarti kembali ke argmax murni; nilai besar membuat
decoder terlalu percaya diri pada teks yang sering muncul di korpus.
"""

from __future__ import annotations

import math
import re

import numpy as np

import ocr.config as C
from ocr.lexicon import Lexicon, weighted_levenshtein
from ocr.lm import BOS, CharLM

_EPS = 1e-12

# Spasi sebelum tanda baca hampir selalu salah pada struk: segmentasi kata
# memecah '.' menjadi kata sendiri karena huruf titik sangat sempit.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,:])")
# Spasi sesudah tanda baca diikuti angka juga salah, **tetapi hanya kalau
# tanda baca itu mengikuti angka** — itu pemisah ribuan/desimal
# ("8. 700" -> "8.700") — **dan** diikuti tepat tiga digit. Pada
# "NO. 0009" titik bukan pemisah angka (diikuti empat digit), jadi spasinya
# dipertahankan.
_SPACE_AFTER_PUNCT_BEFORE_DIGIT = re.compile(r"(?<=\d)([.,:])\s+(?=\d{3}(?!\d))")


def normalize_spacing(text: str) -> str:
    """Rapikan spasi di sekitar tanda baca angka.

    Segmentasi kata memisahkan ``.`` dan ``:`` menjadi kata tersendiri karena
    lebarnya hanya beberapa piksel, sehingga muncul ``8 . 700``. Aturan di
    sini mengembalikannya menjadi ``8.700`` tanpa menyentuh spasi antar kata
    biasa.
    """
    if not text:
        return text
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    out = _SPACE_AFTER_PUNCT_BEFORE_DIGIT.sub(r"\1", out)
    return out

# Rentetan garis pemisah pada struk (``------``) sering menempel pada baris
# teks: ``------ LAYANAN KONSUMEN ------``. Garis seperti itu bukan isi struk,
# jadi dibuang; baris yang seluruhnya garis menjadi kosong dan diabaikan.
_RULE_RUN = re.compile(r"[-_=]{4,}")

def strip_rule_artifacts(text: str) -> str:
    """Buang hiasan garis pemisah, sisakan teksnya saja."""
    if not text:
        return text
    cleaned = _RULE_RUN.sub(" ", text)
    cleaned = " ".join(cleaned.split())
    if not any(ch.isalnum() for ch in cleaned):
        return ""
    return cleaned


_NUMBER_PUNCT = set(".,:/-%")


# Huruf yang secara bentuk bisa tertukar dengan angka. Dipakai untuk
# menghitung "berapa banyak angka yang seharusnya ada" di sebuah token.
_DIGIT_CONFUSABLE = frozenset("OoQDIliZzASGTBbgq")

# Bentuk harga dengan pemisah ribuan yang masih memuat ``O``/``o`` sebagai
# pengganti ``0``, mis. ``1.OOO`` (``1.000``). Dipakai supaya token harga
# satu digit pun tetap dikenali sebagai angka.
_PRICE_SHAPE_RE = re.compile(r"^\d+[.,][0-9Oo]{3}$")

# Bentuk harga yang lebih longgar: semua karakter boleh huruf yang mirip
# angka. Dipakai sebagai penjaga agar ``L.OO0`` (``1.000``) tidak "diperbaiki"
# menjadi ``L.OOO`` oleh aturan kata.
_LOOSE_PRICE_RE = re.compile(
    r"^[\dOoIlSsZzBbGgTt]+[.,][\dOoIlSsZzBbGgTt]{3}$"
)


def _digit_count(token: str) -> int:
    return sum(1 for ch in token if ch.isdigit())


def _digit_slots(token: str) -> int:
    """Perkiraan jumlah posisi angka, termasuk huruf yang mirip angka.

    ``47.2OO`` punya lima posisi angka (4, 7, 2, O, O), sama dengan
    ``47.200``. Perbandingan ini yang mencegah ``95.4O0`` dikoreksi menjadi
    ``5.400`` — jumlah posisi angkanya berbeda, jadi nilainya berubah.
    """
    return sum(1 for ch in token if ch.isdigit() or ch in _DIGIT_CONFUSABLE)


def _is_numeric_token(token: str) -> bool:
    """True bila token hanya berisi digit dan tanda baca angka."""
    if not token:
        return False
    if _digit_count(token) == 0:
        return False
    return all(ch.isdigit() or ch in _NUMBER_PUNCT for ch in token)


def _is_numeric_ish(token: str) -> bool:
    """True bila token pada dasarnya angka tetapi memuat huruf.

    Bentuk seperti ``95.4O0`` muncul karena ``0`` dan ``O`` tidak bisa
    dibedakan dari bentuk. Syaratnya minimal dua digit asli: satu digit saja
    terlalu mudah muncul di kata biasa (``TOP84``), sedangkan dua digit
    hampir selalu harga, tanggal, atau nomor.
    """
    if not token:
        return False
    digits = _digit_count(token)
    if digits < 2:
        # Satu digit pun tetap jelas angka bila bentuknya harga berpemisah
        # ribuan, mis. ``1.OOO`` yang sebenarnya ``1.000``. Tanpa aturan ini
        # baris seperti ``DISKON 1.000`` gagal terkoreksi.
        return bool(_PRICE_SHAPE_RE.match(token))
    return digits / len(token) >= 0.3


def beam_decode(
    probs: np.ndarray,
    lm: CharLM,
    beam_width: int | None = None,
    lm_weight: float | None = None,
    candidate_width: int | None = None,
    prefix: str = "",
) -> str:
    """Cari rangkaian karakter dengan skor gabungan tertinggi.

    Args:
        probs: ``(n_glyph, n_kelas)`` probabilitas dari classifier. Satu
            baris di sini adalah satu **kata**; spasi berasal dari segmentasi
            kata, bukan dari classifier (charset tidak memuat spasi).
        lm: model bahasa karakter.
        beam_width: jumlah hipotesis yang disimpan per langkah.
        lm_weight: bobot log-prob model bahasa.
        candidate_width: jumlah kandidat karakter per glyph yang diperiksa.
        prefix: teks yang sudah terbaca sebelumnya, dipakai sebagai konteks
            model bahasa supaya kata ini tidak dinilai terpisah dari
            kalimatnya.

    Returns:
        Teks hasil decode (tanpa spasi buatan).
    """
    if beam_width is None:
        beam_width = C.BEAM_WIDTH
    if lm_weight is None:
        lm_weight = C.LM_WEIGHT
    if candidate_width is None:
        candidate_width = max(beam_width, 4)

    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return ""

    context_len = max(1, lm.order - 1)
    start_context = (BOS * context_len + prefix)[-context_len:]
    beams: list[tuple[float, str, str]] = [(0.0, "", start_context)]

    for position in range(arr.shape[0]):
        row = arr[position]
        # Ambil kandidat teratas untuk glyph ini saja.
        top = np.argsort(row)[::-1][:candidate_width]

        # Kalau classifier sangat yakin, model bahasa tidak boleh membantah.
        # Tanpa penjaga ini, LM yang dilatih pada korpus sintetis bisa
        # mengubah digit yang terbaca jelas (``95.400`` -> ``15.400``)
        # hanya karena rangkaian itu lebih sering muncul di korpus. Peran LM
        # adalah memutus ambiguitas bentuk (0/O, 1/I), bukan menimpa bukti.
        if float(row[top[0]]) >= C.LM_TRUST_THRESHOLD:
            top = top[:1]

        candidates: list[tuple[float, str, str]] = []
        for score, text, context in beams:
            for class_index in top:
                char = C.IDX_TO_CHAR[int(class_index)]
                acoustic = math.log(float(row[class_index]) + _EPS)
                language = lm.next_char_logprobs(context, char)
                total = score + acoustic + lm_weight * language
                candidates.append(
                    (total, text + char, (context + char)[-context_len:])
                )

        candidates.sort(key=lambda item: item[0], reverse=True)
        beams = candidates[:beam_width]

    return beams[0][1]


# Huruf yang secara bentuk bisa tertukar dengan angka. Dipakai **hanya** di
# dalam token angka: ``O`` pada ``47.2O0`` pasti nol, tetapi ``O`` pada
# ``OVO`` adalah huruf.
_DIGIT_CONFUSION = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "i": "1", "|": "1",
    "Z": "2", "z": "2",
    "A": "4",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "T": "7",
    "B": "8",
    "g": "9", "q": "9",
}

_PRICE_RE = re.compile(r"^\d{1,3}(?:\.\d{3})+(?:,\d{2})?$")
_PLAIN_DIGITS_RE = re.compile(r"^\d{3,}$")


def normalize_numeric_token(token: str) -> str:
    """Perbaiki huruf yang sebenarnya angka di dalam token angka.

    ``0`` dan ``O`` pada font tanpa garis miring bentuknya identik, begitu
    juga ``1``/``I`` dan ``5``/``S``. Di dalam token angka konteksnya jelas:
    itu pasti angka. Pemetaan hanya dilakukan kalau hasilnya benar-benar
    berbentuk angka — harga (``47.2O0`` -> ``47.200``) atau bilangan bulat
    panjang (``2O25`` -> ``2025``).

    Penjaga bentuk itu penting: ``12S`` pada ``SARI ROTI TAWAR 12S`` adalah
    ukuran kemasan, bukan harga. Tanpa penjaga, ``12S`` menjadi ``125``.
    """
    if not (_is_numeric_token(token) or _is_numeric_ish(token)):
        return token
    # Ukuran kemasan bukan angka walaupun sebagian besar karakternya digit.
    # Tanpa penjaga ini, ``12S`` (ukuran, bukan harga) menjadi ``125``.
    if is_packaging_token(token):
        return token
    mapped = "".join(_DIGIT_CONFUSION.get(ch, ch) for ch in token)
    if mapped == token:
        return token
    if _PRICE_RE.match(mapped) or _PLAIN_DIGITS_RE.match(mapped):
        return mapped
    return token


# Satuan kemasan yang lazim di struk minimarket. Daftar ini tertutup, jadi
# bisa dipakai untuk mengoreksi huruf yang salah baca pada bagian satuan.
_UNIT_TOKENS = (
    "G", "KG", "GR", "MG", "ML", "L", "S", "PCS", "BTL", "SCH", "PCH",
    "PACK", "BOX", "BKS", "RLS", "TOP", "X",
)

_PACKAGING_RE = re.compile(r"^(\d[\dOoIlSsZzBbGgTt]*)([A-Za-z]+)$")

# Satuan satu huruf dan digit yang bentuknya menyerupainya. Dipakai saat
# decoder membaca satuan sebagai digit sehingga token menjadi angka semua:
# ``250G`` -> ``2506``.
_DIGIT_TO_UNIT_LETTER = {"6": ("G",), "1": ("L",), "5": ("S",)}

# Satuan yang salah baca dua huruf sekaligus pada cetakan thermal.
# ``M``/``N`` dan ``I``/``L`` bentuknya hampir sama, jadi ``200ML`` menjadi
# ``200NIL`` dan ``200M1``.
_UNIT_ALIASES = {
    "NIL": "ML",
    "N1L": "ML",
    "M1L": "ML",
    "N1": "ML",
    "NI": "ML",
    "NII": "ML",
    "MI": "ML",
    "NILIL": "ML",
    "NL": "ML",
    "MIL": "ML",
    "K6": "KG",
    "KS": "KG",
}


def normalize_packaging_token(token: str, known=None) -> str:
    """Perbaiki ukuran kemasan seperti ``250G`` atau ``200ML``.

    Ukuran kemasan punya pola tetap: angka diikuti satuan. Bagian angka
    dinormalkan seperti harga (``20O`` -> ``200``), dan bagian satuan dicocokkan
    ke daftar satuan tertutup — ``1006`` -> ``100G`` karena ``6`` dan ``G``
    hanya berbeda satu piksel pada font ini.

    Tanpa aturan ini, ``NUGGET FIESTA 250G`` terbaca ``2506``: nama barang
    jadi salah walaupun harganya benar.
    """
    match = _PACKAGING_RE.match(token)
    if not match:
        # Tidak ada huruf sama sekali: satuannya terbaca sebagai digit
        # (``250G`` -> ``2506``), kadang bercampur huruf yang mirip angka
        # (``25O6``). Normalkan dulu ke bentuk angka, lalu coba ganti digit
        # terakhir menjadi satuan — **hanya** kalau hasilnya token yang
        # dikenal leksikon. Tanpa bukti itu, angka apa pun yang kebetulan
        # berakhiran 6 akan berubah menjadi satuan.
        if known is None:
            return token
        as_digits = "".join(_DIGIT_CONFUSION.get(ch, ch) for ch in token)
        if not (as_digits.isdigit() and len(as_digits) >= 3):
            return token
        for letter in _DIGIT_TO_UNIT_LETTER.get(as_digits[-1], ()):
            candidate = as_digits[:-1] + letter
            if known(candidate):
                return candidate
        return token

    digits, suffix = match.group(1), match.group(2)
    fixed_digits = "".join(_DIGIT_CONFUSION.get(ch, ch) for ch in digits)
    upper = suffix.upper()

    if upper in _UNIT_TOKENS:
        return fixed_digits + upper

    # Cari satuan yang berjarak edit paling banyak 1 dari yang terbaca.
    best: tuple[float, str] | None = None
    for unit in _UNIT_TOKENS:
        if abs(len(unit) - len(upper)) > 1:
            continue
        distance = weighted_levenshtein(upper, unit)
        if distance <= 1.0 and (best is None or distance < best[0]):
            best = (distance, unit)

    if best is not None:
        return fixed_digits + best[1]

    # Satuan tiga huruf yang salah baca dua huruf sekaligus. Pada cetakan
    # thermal ``M`` dan ``N`` hampir sama, begitu juga ``I`` dan ``L``,
    # sehingga ``200ML`` terbaca ``200NIL``. Jarak editnya 2, jadi tidak
    # tertangkap aturan di atas; peta tertutup ini menanganinya tanpa
    # membuka pintu bagi kata lain.
    alias = _UNIT_ALIASES.get(upper)
    if alias is not None:
        return fixed_digits + alias
    return token


def is_packaging_token(token: str, known=None) -> bool:
    """True bila token berbentuk ukuran kemasan (angka + satuan).

    Dipakai untuk **menghentikan** pemrosesan lebih lanjut: ``250G`` bukan
    angka, walaupun tiga dari empat karakternya digit. Tanpa penjaga ini,
    normalisasi angka mengubahnya menjadi ``2506``.
    """
    if not token or len(token) < 3:
        return False

    match = _PACKAGING_RE.match(token)
    if match:
        suffix = match.group(2).upper()
        if suffix in _UNIT_TOKENS:
            return True
        for unit in _UNIT_TOKENS:
            if abs(len(unit) - len(suffix)) <= 1 and weighted_levenshtein(
                suffix, unit
            ) <= 1.0:
                return True
        return False

    if known is None:
        return False
    as_digits = "".join(_DIGIT_CONFUSION.get(ch, ch) for ch in token)
    if not (as_digits.isdigit() and len(as_digits) >= 3):
        return False
    return any(
        known(as_digits[:-1] + letter)
        for letter in _DIGIT_TO_UNIT_LETTER.get(as_digits[-1], ())
    )


_AMOUNT_CONTEXT = frozenset({
    "DISKON", "TOTAL", "TUNAI", "KEMBALI", "HEMAT", "BAYAR", "JUAL",
    "HARGA", "SUBTOTAL", "PPN", "PPNBM", "BELANJA", "SISA", "DPP", "ITEM",
    "POTONGAN", "CASH", "DEBIT", "QRIS", "KARTU", "TAGIHAN",
    "PEMBULATAN",
})

# Di dalam kolom nominal, huruf yang tersisa hampir pasti angka yang salah
# baca. ``G`` dipetakan ke ``0`` (bukan ``6`` seperti di ``_DIGIT_CONFUSION``)
# karena harga bulat berakhiran nol jauh lebih sering daripada berakhiran 6:
# ``DISKON 50G`` adalah ``DISKON 500``.
_AMOUNT_LETTER = {
    "O": "0", "o": "0", "Q": "0", "D": "0", "G": "0", "g": "0",
    "I": "1", "l": "1", "L": "1", "|": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "B": "8",
}

def _amount_value(token: str) -> str | None:
    """Baca ``token`` sebagai nominal bila bentuknya memungkinkan.

    Dipakai hanya ketika token sebelumnya adalah kata kunci nominal, jadi
    konteksnya sudah jelas angka. Mengembalikan ``None`` bila hasilnya bukan
    bentuk harga/nominal yang sah, supaya token aneh tidak ikut diubah.
    """
    chars = list(token)
    # Huruf satuan di akhir token hampir pasti digit yang salah baca, dan
    # ``G`` pada kolom nominal lebih sering ``0`` daripada ``6``
    # (``DISKON 50G`` = ``DISKON 500``). Untuk posisi lain tetap pakai
    # ``_DIGIT_CONFUSION``.
    if chars and chars[-1] in _AMOUNT_LETTER:
        chars[-1] = _AMOUNT_LETTER[chars[-1]]
    mapped = "".join(
        _DIGIT_CONFUSION.get(ch) or _AMOUNT_LETTER.get(ch) or ch for ch in chars
    )
    mapped = mapped.strip(".,")
    if _PRICE_RE.match(mapped) or _PLAIN_DIGITS_RE.match(mapped):
        return mapped
    return None

# Kebalikan `_DIGIT_CONFUSION`: dipakai saat token jelas sebuah **kata**.
# ``0`` dan ``O`` bentuknya identik pada font tanpa garis miring, jadi
# classifier tidak punya dasar memilih. Konteks kata yang memutuskan:
# ``SIDOM0R0``, ``K0NSUMEN``, ``R0TI``, ``PR0MO`` semuanya kata.
_LETTER_CONFUSION = {
    "0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B",
}

def normalize_alpha_token(token: str) -> str:
    """Ganti angka yang sebenarnya huruf di dalam sebuah kata.

    Hanya berlaku bila token mayoritas huruf dan tidak berbentuk harga atau
    ukuran kemasan. Tanpa penjaga itu, ``100G`` akan menjadi ``IOOG``.
    """
    if not token:
        return token
    letters = sum(1 for ch in token if ch.isalpha())
    if letters < 2:
        return token
    if letters / len(token) < 0.6:
        return token
    if is_packaging_token(token):
        return token
    # Bentuk harga (``L.OO0``) bukan kata; biarkan aturan angka yang menangani.
    if _LOOSE_PRICE_RE.match(token):
        return token
    mapped = "".join(_LETTER_CONFUSION.get(ch, ch) for ch in token)
    return mapped

def correct_words(
    text: str,
    lexicon: Lexicon,
    max_distance: float | None = None,
    use_lexicon: bool = True,
) -> str:
    """Perbaiki kata yang salah eja memakai leksikon.

    Tiga jenis token diperlakukan berbeda, dan pembedaan itu penting:

    - **Kata pendek** (1-2 huruf) tidak disentuh: terlalu mudah berubah
      menjadi kata lain.
    - **Token angka** (``47.200``) dan **token hampir angka** (``95.4O0``)
      tidak pernah dikoreksi leksikon. Dulu token hampir angka masih
      dikoreksi ke angka lain di leksikon, dan hasilnya bencana: ``95.400``
      menjadi ``15.400`` karena ``15.400`` juga ada di korpus. Untuk angka,
      satu-satunya koreksi yang sah adalah memetakan huruf yang bentuknya
      sama dengan angka (``O`` -> ``0``), bukan mengganti nilainya.
    - **Kata** dikoreksi ke entri leksikon terdekat dalam batas jarak.
    """
    if max_distance is None:
        max_distance = C.LEXICON_MAX_DISTANCE

    pieces = text.split(" ")
    out: list[str] = []
    previous = ""
    for token in pieces:
        if not token or len(token) < 3:
            out.append(token)
            previous = token.strip(".,:-/%&").upper()
            continue
        if previous in _AMOUNT_CONTEXT:
            # Kolom nominal: percayai konteks lebih dari bentuk glyph, karena
            # ``500`` dan ``50G`` sama-sama masuk akal tanpa konteks.
            amount = _amount_value(token)
            if amount is not None:
                out.append(amount)
                previous = amount
                continue
        if is_packaging_token(token, lexicon.contains):
            out.append(normalize_packaging_token(token, lexicon.contains))
            previous = token.strip(".,:-/%&").upper()
            continue
        if _is_numeric_token(token) or _is_numeric_ish(token):
            out.append(normalize_numeric_token(token))
            previous = token.strip(".,:-/%&").upper()
            continue
        if lexicon.contains(token):
            out.append(token)
            previous = token.strip(".,:-/%&").upper()
            continue

        # Kata yang memuat angka pengganti huruf (``SIDOM0R0``) diperbaiki
        # sebelum pencarian leksikon, supaya jaraknya ke entri benar nol.
        alpha_fixed = normalize_alpha_token(token)
        if alpha_fixed != token:
            out.append(alpha_fixed)
            previous = alpha_fixed.strip(".,:-/%&").upper()
            continue

        # Batas jarak tumbuh bersama panjang kata. Untuk kata 5 huruf hanya
        # 1 huruf boleh salah; tanpa batas ini ``TUNAI`` berubah menjadi
        # ``TAWAR`` (jarak 2) hanya karena keduanya ada di leksikon.
        if not use_lexicon:
            out.append(token)
            previous = token.strip(".,:-/%&").upper()
            continue

        limit = max(1.0, min(max_distance, len(token) * C.LEXICON_MAX_RATIO))
        replacement, _ = lexicon.closest(token, limit)
        out.append(replacement if replacement is not None else token)
    return " ".join(out)


def decode_line(
    probs: np.ndarray,
    lm: CharLM,
    lexicon: Lexicon | None = None,
    beam_width: int | None = None,
    lm_weight: float | None = None,
    prefix: str = "",
) -> tuple[str, float]:
    """Decode satu kata dan kembalikan ``(teks, keyakinan)``.

    Keyakinan dihitung dari rata-rata probabilitas maksimum per glyph —
    mengukur seberapa tegas classifier, bukan seberapa wajar teksnya.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return "", 0.0

    text = beam_decode(
        arr, lm, beam_width=beam_width, lm_weight=lm_weight, prefix=prefix
    )
    confidence = float(np.mean(arr.max(axis=1)))

    if lexicon is not None:
        # Leksikon adalah **cadangan** untuk bacaan yang ragu. Kalau
        # classifier yakin pada setiap glyph, kata itu dibiarkan: dulu
        # ``INDOMARIE BISKUIT`` yang terbaca benar diubah menjadi
        # ``INDOMIE BISKUAT`` hanya karena bentuk kedua lebih sering muncul
        # di korpus. Normalisasi angka dan satuan tetap dijalankan.
        text = correct_words(
            text,
            lexicon,
            use_lexicon=confidence < C.LEXICON_TRUST_THRESHOLD,
        )
    return text, confidence
