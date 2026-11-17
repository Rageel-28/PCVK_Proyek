"""Koreksi kata hasil OCR memakai leksikon produk.

Prinsipnya sederhana namun sangat efektif: hasil OCR yang salah biasanya
masih berjarak edit kecil dari kata yang benar ("IND0M1E" -> "INDOMIE").
Dengan kamus 1.419 nama produk + kata umum struk, kesalahan karakter
bisa dipulihkan menjadi kata yang benar.

Levenshtein diimplementasikan manual agar bisa ditiru di Dart.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

import config as C


# Kata umum yang sering muncul di struk namun tidak ada di nama produk.
EXTRA_WORDS = {
    "TOTAL", "SUBTOTAL", "TUNAI", "KEMBALI", "KEMBALIAN", "DISKON", "PPN",
    "PAJAK", "ITEM", "QTY", "HARGA", "JUMLAH", "BAYAR", "CASH", "CHANGE",
    "GRAND", "TOTAL", "RP", "TGL", "KASIR", "STRUK", "NOTA", "TERIMA",
    "KASIH", "SELAMAT", "DATANG", "KEMBALI", "MINIMARKET", "INDOMARET",
    "ALFAMART", "ALFAMIDI", "INDOMARCO", "PRISMATAMA", "ALFARIA", "TRIJAYA",
    "SUMBER", "TRANSAKSI", "ANDA", "MEMBER", "NPWP", "DPP", "BKP", "JKP",
    "SHOPEEPAY", "GOPAY", "QRIS", "OVO", "DANA", "DEBIT", "KREDIT", "VOUCHER",
    "POTONGAN", "HEMAT", "LAYANAN", "KONSUMEN", "ALFA", "TOWER", "TANGERANG",
    "JAKARTA", "BANDUNG", "SURABAYA", "BANTEN", "INDONESIA", "RAYA", "JALAN",
    "DEWI", "SARTIKA", "KAV", "NO", "RT", "RW", "KEL", "KEC", "KOTA",
}

# Panjang minimum token yang boleh dikoreksi. Di bawah ini jumlah tetangga
# pada jarak edit 1 terlalu banyak, sehingga koreksi hampir selalu salah.
MIN_LEN = 5

# Daftar kata utuh struk & produk ritel Indonesia agar kosakata struk asli
# tercakup tanpa memasukkan potongan terpotong dari kolom `receipt_text`.
CURATED_RECEIPT_WORDS = {
    "TOTAL", "SUBTOTAL", "TUNAI", "KEMBALI", "KEMBALIAN", "BELANJA", "DISKON",
    "HARGA", "JUMLAH", "HEMAT", "LUNAS", "KASIR", "TERIMA", "KASIH", "BARANG",
    "SUDAH", "DIBELI", "TIDAK", "DIKEMBALIKAN", "MEMBER", "ALFAGIFT", "VOUCHER",
    "POTONGAN", "CASHBACK", "LAYANAN", "KONSUMEN", "SHOPEEPAY", "DEBIT", "KREDIT",
    "TANGGAL", "STRUK", "PAJAK", "TRANSFER", "EXPIRED", "POTENSI", "SELAMAT",
    "BERTAMBAH", "MENDAPATKAN", "KETENTUAN", "KEBIJAKAN", "PRIVASI", "PERSETUJUAN",
    "ALFAMART", "ALFAMIDI", "INDOMARET", "INDOMARCO", "PRISMATAMA", "ALFARIA",
    "TRIJAYA", "MINIMARKET", "INDONESIA",
    "CIMORY", "HAZELNUT", "FRESTEA", "CREAM", "KANZLER", "MINERALE", "MINERAL",
    "CHOCOLATE", "COKLAT", "COKELAT", "VANILLA", "STRAWBERRY", "ORIGINAL",
    "BLUEBERRY", "CARAMEL", "YOGURT", "INDOMIE", "INDOMILK", "ULTRAMILK",
    "PEPSODENT", "LIFEBUOY", "SUNSILK", "SUNLIGHT", "CHITATO", "PIATTOS",
    "NABATI", "POTABEE", "SILVERQUEEN", "CADBURY", "TANGO", "WAFER", "BISKUIT",
    "NESCAFE", "KAPAL", "TORABIKA", "GOODDAY", "POCARI", "SWEAT", "MIZONE",
    "PUCUK", "SOSRO", "GORENG", "KACANG", "SOSIS", "BAKSO", "NUGGET", "SARIROTI",
    "BRANDING", "EXPRESO", "FRISIAN", "GARDENIA", "PERMATA", "MAKMUR", "SOETOMO",
    "JATINANGOR", "SUMEDANG", "PROMOSI", "LENGKAP", "BERRY",
    "MARLBORO", "SAMPOERNA", "SURYA", "DJARUM", "DUNHILL", "GUDANG", "GARAM",
    "FILTER", "PANTENE", "CLEAR", "REJOICE", "SHAMPOO", "SABUN", "DETERJEN",
    "RINSO", "ATTACK", "MOLTO", "DOWNY", "VIXAL", "WIPOL", "BAYGON", "GLADE",
    "PASEO", "TESSA", "TISSUE", "FLORIDINA", "SPRITE", "FANTA", "PEPSI",
    "GREENFIELDS", "BRAND", "DANCOW", "BEBELAC", "ENFAGROW", "NUTRILON",
    "OVALTINE", "BANGO", "ROYCO", "MASAKO", "SAJIKU", "BIMOLI", "SANIA", "FILMA",
    "TROPICAL", "SUNCO", "FORTUNE", "GULAKU", "NEXTAR", "GARUDA", "QTELA",
    "KUSUKA", "FRENCH", "FRIES", "CHOCO", "MATCHA", "LATTE", "CAPPUCCINO",
    "AMERICANO", "ESPRESSO", "BROWN", "SUGAR", "BLACK", "WHITE", "CREAMY",
    "BUTTER", "CHEESE", "SINGLES", "SALMON", "BAWANG", "RENDANG", "PEDAS",
    "MANIS", "BOTOL", "GALON", "REFILL", "POUCH", "SACHET", "KALENG", "PLASTIK",
    "MERAH", "PUTIH", "HITAM", "HIJAU", "KUNING", "BESAR", "KECIL", "REGULER",
    "EXTRA", "SPESIAL", "SPECIAL", "SUPER", "PREMIUM", "FRESH", "NATURAL",
    "HEALTHY", "BEAUTY", "DAILY", "JUICE", "BURGER", "PIZZA", "PASTA",
    "SPAGHETTI", "RAMEN", "SNACK", "CHIPS", "ALMOND", "MOCHI", "CLASSIC",
    "PROCHIZ", "SEDAAP", "SARIMI", "BODREX", "ANTANGIN", "SALONPAS", "BETADINE",
    "PONDS", "WARDAH", "GARNIER", "EMINA", "MARINA", "NIVEA", "VASELINE",
    "CHARM", "LAURIER", "MAMYPOKO", "SWEETY", "HUGGIES", "ZWITSAL", "CUSSONS",
    "PIGEON", "KLIKINDOMARET", "GRATIS", "ONGKIR", "SAMPAI", "LEBIH", "MUDAH",
    "HEMAT", "SEDEKAH", "KEMANUSIAAN", "PEDULI", "PENDIDIKAN", "MANDIRI",
    "PREPAID", "NOMOR", "KARTU", "JENIS", "DUPLIKAT",
}

# Pasangan karakter yang sering tertukar secara visual pada cetakan thermal.
# Biaya substitusi diskalakan x20 agar 100% deterministik dengan Dart:
# cocok=0, pasangan mirip=9, substitusi biasa=20, sisip/hapus=23.
_CONFUSION_PAIRS = frozenset({
    frozenset({"0", "O"}), frozenset({"O", "C"}), frozenset({"O", "Q"}), frozenset({"O", "D"}),
    frozenset({"O", "."}), frozenset({"O", ","}), frozenset({"O", "G"}),
    frozenset({"1", "I"}), frozenset({"I", "L"}), frozenset({"I", "T"}), frozenset({"L", "T"}),
    frozenset({"I", "J"}), frozenset({"L", "E"}),
    frozenset({"M", "N"}), frozenset({"M", "A"}), frozenset({"M", "W"}), frozenset({"M", "H"}),
    frozenset({"M", "R"}), frozenset({"M", "I"}), frozenset({"M", "."}), frozenset({"M", ","}),
    frozenset({"N", "A"}), frozenset({"N", "H"}), frozenset({"N", "U"}),
    frozenset({"Z", "2"}), frozenset({"Z", "G"}), frozenset({"Z", "A"}), frozenset({"Z", "E"}),
    frozenset({"Z", "&"}), frozenset({"Z", "7"}),
    frozenset({"E", "@"}), frozenset({"E", "3"}), frozenset({"E", "F"}), frozenset({"E", "B"}),
    frozenset({"B", "8"}), frozenset({"B", "H"}), frozenset({"B", "R"}),
    frozenset({"D", "A"}), frozenset({"D", "S"}), frozenset({"D", "0"}),
    frozenset({"U", "V"}), frozenset({"U", "A"}), frozenset({"U", "4"}), frozenset({"U", ","}),
    frozenset({"Y", ","}), frozenset({"Y", "V"}), frozenset({"Y", "T"}),
    frozenset({"S", "5"}), frozenset({"G", "6"}), frozenset({"T", "7"}),
    frozenset({"A", "4"}), frozenset({"F", "P"}), frozenset({"K", "R"}),
})


def _sub_cost(a: str, b: str) -> int:
    if a == b:
        return 0
    return 9 if frozenset({a, b}) in _CONFUSION_PAIRS else 20


def weighted_levenshtein(a: str, b: str, max_cost: int) -> int:
    """Jarak edit berbobot kesamaan visual OCR (skala x20)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return max_cost + 1
    if la <= 6 and la != lb:
        return max_cost + 1

    prev = [i * 23 for i in range(lb + 1)]
    for i in range(1, la + 1):
        cur = [i * 23] + [0] * lb
        row_min = cur[0]
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = _sub_cost(ca, b[j - 1])
            d_del = prev[j] + 23
            d_ins = cur[j - 1] + 23
            d_sub = prev[j - 1] + cost
            v = d_del if d_del < d_ins else d_ins
            if d_sub < v:
                v = d_sub
            cur[j] = v
            if v < row_min:
                row_min = v
        if row_min > max_cost:
            return max_cost + 1
        prev = cur
    return prev[lb]


def levenshtein(a: str, b: str, max_dist: int | None = None) -> int:
    """Jarak edit dengan pemotongan dini bila melebihi ``max_dist``."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    if max_dist is not None and abs(la - lb) > max_dist:
        return max_dist + 1

    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        row_min = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if cur[j] < row_min:
                row_min = cur[j]
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[lb]


def _clean(word: str) -> str:
    return re.sub(r"[^A-Z0-9.&/+\-,:@()=]", "", word.upper())


class Lexicon:
    """Kamus kata dengan pencarian cepat berbasis panjang kata."""

    def __init__(self, words: set[str] | None = None) -> None:
        self.words: set[str] = set()
        self._by_len: dict[int, list[str]] = {}
        if words:
            self.add_many(words)

    def add_many(self, words) -> None:
        for w in words:
            w = _clean(w)
            if len(w) >= 2:
                self.words.add(w)
        self._rebuild()

    def _rebuild(self) -> None:
        self._by_len.clear()
        for w in self.words:
            self._by_len.setdefault(len(w), []).append(w)
        for bucket in self._by_len.values():
            bucket.sort()

    @classmethod
    def from_dataset(cls) -> "Lexicon":
        lex = cls()
        words: set[str] = {w for w in EXTRA_WORDS if len(w) >= MIN_LEN} | CURATED_RECEIPT_WORDS

        if C.DATASET_CSV.exists():
            df = pd.read_csv(C.DATASET_CSV)
            if "nama_produk" in df.columns:
                for value in df["nama_produk"].dropna().astype(str):
                    for tok in re.findall(r"[A-Z]+", value.upper()):
                        if len(tok) >= MIN_LEN:
                            words.add(tok)
        lex.add_many(words)
        return lex

    def candidates(self, word: str, max_dist: int) -> list[str]:
        """Kandidat dengan panjang berbeda maksimal ``max_dist``."""
        if not word:
            return []
        max_len_diff = 0 if len(word) <= 6 else min(1, max_dist)
        out: list[str] = []
        for length in range(max(MIN_LEN, len(word) - max_len_diff), len(word) + max_len_diff + 1):
            bucket = self._by_len.get(length)
            if bucket:
                out.extend(bucket)
        return out

    def correct(self, word: str, max_dist: int | None = None) -> tuple[str, int, bool]:
        """Kembalikan (kata terbaik, jarak, apakah dikoreksi).

        Koreksi konservatif memakai biaya edit berbobot kesamaan OCR:
        - token berangka, terlalu pendek (< 5), atau < 2 huruf dibiarkan,
        - kata <= 6 huruf tidak boleh berubah panjangnya (mencegah pemendekan
          atau penyusutan seperti `BACTERIAL` -> `BATERAI`), dan
        - pemenang harus unggul jelas (`selisih biaya >= 6`).
        """
        w = re.sub(r"[^A-Z0-9.,@&/-]", "", word.upper())
        if len(w) < MIN_LEN:
            return word, 0, False
        if any(ch.isdigit() for ch in w):
            return word, 0, False
        if sum(1 for ch in w if ch.isalpha()) < 2:
            return word, 0, False

        if w in self.words:
            return w, 0, False

        if max_dist is not None:
            limit = max_dist * 20
        elif len(w) <= 5:
            limit = 13
        elif len(w) <= 7:
            limit = 24
        elif len(w) <= 9:
            limit = 29
        else:
            limit = 33

        cands: list[tuple[int, str]] = []
        for cand in self.candidates(w, 1):
            if cand == w:
                continue
            cost = weighted_levenshtein(w, cand, limit)
            if cost <= limit:
                cands.append((cost, cand))

        if not cands:
            return word, 0, False
        cands.sort()
        if len(cands) == 1 or (cands[1][0] - cands[0][0] >= 6):
            return cands[0][1], (cands[0][0] + 19) // 20, True
        return word, 0, False

    def correct_text(self, text: str, max_dist: int | None = None) -> tuple[str, int]:
        """Koreksi per kata. Kembalikan (teks, jumlah kata terkoreksi)."""
        fixed = 0
        out: list[str] = []
        for tok in text.split():
            new, dist, changed = self.correct(tok, max_dist)
            if changed:
                fixed += 1
            out.append(new)
        return " ".join(out), fixed
