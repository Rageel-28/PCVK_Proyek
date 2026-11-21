"""Korpus baris struk sintetis untuk melatih classifier karakter.

Korpus dibangkitkan dari kosakata produk minimarket Indonesia dan format
baris struk Indomaret/Alfamart. Karena teksnya kita sendiri yang menyusun,
label karakter bisa didapat gratis saat render (lihat ``render_dataset``).

Aturan penting: setiap baris hanya boleh memakai karakter yang ada di
``config.CHARSET`` ditambah spasi. Kalau tidak, ada kelas yang tidak
pernah dilatih tetapi muncul saat inferensi.
"""

from __future__ import annotations

import random
import re

# ---------------------------------------------------------------------------
# Kosakata
# ---------------------------------------------------------------------------
STORES = (
    "INDOMARET",
    "ALFAMART",
    "ALFAMIDI",
    "INDOMARET",
    "ALFAMART",
)

TAGLINES = (
    "SERBA ADA SERBA MUDAH",
    "BELANJA LEBIH MUDAH",
    "SERBA LENGKAP SERBA MUDAH",
)

ADDRESSES = (
    "JL. RAYA KNO 12 KEL. SUKAPURA",
    "JL. GATOT SUBROTO 17 KEL. KARANG",
    "JL. SULTAN AGUNG 12 KEL. PANDAWA",
    "JL. AHMAD YANI 90 KEL. GUBENG",
    "JL. PANGERAN ANTASARI 5 KEL. TANJUNG",
    "JL. MERDEKA 45 KEL. SUMUR BANDUNG",
    "JL. DIPONEGORO 88 KEL. DAMAI",
)

CITIES = (
    "KEC. CILINCING JAKARTA UTARA",
    "KEC. SEMARANG TENGAH SEMARANG",
    "KEC. CIREBON UTARA CIREBON",
    "KEC. GUBENG SURABAYA",
    "KEC. MEDAN SELAYANG MEDAN",
)

STORE_LINES = (
    "STORE NO. 0001 021-8684074",
    "STORE NO. 0005 021-5777148",
    "STORE NO. 0006 021-2508064",
    "STORE NO. 0009 021-4390827",
    "STORE NO. 0010 021-3539587",
)

FOOTERS = (
    "TERIMA KASIH ATAS KUNJUNGAN ANDA",
    "BARANG YANG SUDAH DIBELI",
    "TIDAK DAPAT DITUKAR KEMBALI",
    "LAYANAN KONSUMEN",
    "BELANJA LEBIH MUDAH DI KLIKINDOMARET",
    "GRATIS ONGKIR 1 JAM SAMPAI",
)

# Nama produk: hanya A-Z, 0-9, spasi, dan simbol yang ada di CHARSET.
PRODUCTS = (
    "POCARI SWEAT 500ML",
    "POCARI SWEAT 350ML",
    "INDOMILK UHT PLAIN 250ML",
    "INDOMILK UHT COKLAT 250ML",
    "KAPAS KECIL TOP84",
    "MARLBORO MERAH 12",
    "MARLBORO PUTIH 12",
    "KACANG GARUDA 90G",
    "MINYAK GORENG SANIA 1L",
    "MINYAK GORENG BIMOLI 2L",
    "BIMOLI 200ML PCH",
    "SARI ROTI TAWAR 12S",
    "ROTI TAWAR SARI ROTI",
    "ROTI KRIM KEJU",
    "NISSIN MIE GORENG",
    "INDOMIE GORENG 85G",
    "INDOMIE AYAM BAWANG",
    "SARIMI SOTO 55G",
    "TELUR AYAM 1/2 KG",
    "TELUR AYAM 1 KG",
    "GULAKU GULA PREMIUM 1KG",
    "GULA PASIR 1 KG",
    "KOPI KAPAL API 65G",
    "KOPI ABC SUSU 10S",
    "NUTRISARI JERUK 1S",
    "FRESTEA BTL 350ML",
    "TEH PUCUK 350ML",
    "TROPICANA SLIM 200ML",
    "YAKULT PACK 5S",
    "CIMORY FRESH MILK 250ML",
    "NUGGET FIESTA 250G",
    "SOSIS SO GOOD 250G",
    "SHAMPO CLEAR SACHET",
    "SHAMPO PANTENE 170ML",
    "SABUN LIFEBUOY 85G",
    "SABUN LIFEBUOY 110G",
    "DETERJEN RINSO 800G",
    "SOFTENER DOWNY 700ML",
    "PEPSODENT 190G",
    "SENSODYNE 100G",
    "ANTANGIN JRG SACHET",
    "TOLAK ANGIN SACHET",
    "PARACETAMOL 500MG",
    "BODREX FLU BATUK",
    "MASKER MEDIS 3PLY",
    "TISSUE PASEO 250S",
    "TISSUE JUMBO 250S",
    "SUNLIGHT 750ML",
    "MAMA LEMON 400ML",
    "AQUA 600ML",
    "AQUA GALON 19L",
    "LE MINERALE 600ML",
    "CLEO 550ML",
    "COCA COLA 390ML",
    "SPRITE 390ML",
    "FANTA STROBERI 390ML",
    "POWERADE 500ML",
    "BUAVITA JAMBU 250ML",
    "PULPY ORANGE 350ML",
    "KACANG DUA KELINCI 80G",
    "CHITATO SAPI PANGGANG",
    "LAYS RUMPUT LAUT",
    "TARO NET 70G",
    "BISKUAT KEJU 300G",
    "OREO VANILLA 133G",
    "KHONG GUAN 650G",
    "SILVERQUEEN 65G",
    "CADBURY DAIRY MILK",
    "ES KRIM WALLS 700ML",
    "PUDING NUTRIJEL",
    "BERAS SETRA RAMOS 5KG",
    "BERAS PANDAN WANGI 5KG",
    "TEPUNG SEGITIGA 1KG",
    "KECAP ABC 520ML",
    "SAOS SAMBU 135ML",
    "GARAM REFINA 500G",
    "MICIN AJINOMOTO 250G",
    "MARGARIN BLUE BAND 200G",
    "KEJU CHEDDAR 170G",
    "SUSU BENDERA 545G",
    "SUSU DANCOW 400G",
    "DIAPERS MERRIES M34",
    "A&W ROOT BEER 330ML",
    "MIZONE 500ML",
    "ZINC LOZENGES 10S",
    "BELI 2+1 GRATIS",
)

# Baris ringkasan / pembayaran.
SUMMARY_LABELS = (
    "TOTAL",
    "SUBTOTAL",
    "DPP",
    "PPN 11%",
    "PPN 10%",
    "DISKON",
    "POTONGAN",
    "PEMBULATAN",
    "TUNAI",
    "DEBIT BCA",
    "DEBIT MANDIRI",
    "DEBIT BRI",
    "QRIS",
    "OVO",
    "DANA",
    "SHOPEEPAY",
    "GOPAY",
    "KEMBALI",
    "TOTAL ITEM",
    "TOTAL QTY",
    "ANDA HEMAT",
    "HARGA JUAL",
)

TRANSACTION_META = (
    "NO 76963184",
    "NO 80511481",
    "NO 11543456",
    "KSR R. PRATAMA",
    "KSR B. SANTOSO",
    "KSR D. ANGGR",
    "ITEM",
    "TOTAL",
)


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def format_price(value: int) -> str:
    """Format rupiah gaya Indonesia: ``8700`` -> ``8.700``."""
    return f"{value:,}".replace(",", ".")


def classify_line(line: str) -> str:
    """Golongkan satu baris korpus.

    Nilai: ``rule`` (garis putus-putus), ``qty`` (``3 X 19.800``),
    ``summary`` (TOTAL/DPP/PPN/dst), ``item`` (nama + harga), atau
    ``text`` (header, footer, metadata).
    """
    stripped = line.strip()
    if not stripped:
        return "text"
    if set(stripped) <= {"-"}:
        return "rule"
    if re.match(r"^\d+\s*X\s+\d", stripped):
        return "qty"
    first = stripped.split()[0]
    if any(stripped.startswith(label) for label in SUMMARY_LABELS):
        return "summary"
    if first in SUMMARY_LABELS:
        return "summary"
    last = stripped.split()[-1]
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})*", last) and re.search(r"[A-Z]", stripped):
        return "item"
    return "text"


def line_kinds(lines: list[str]) -> dict[str, int]:
    """Hitung jumlah baris per jenis."""
    counts = {"rule": 0, "qty": 0, "summary": 0, "item": 0, "text": 0}
    for line in lines:
        counts[classify_line(line)] += 1
    return counts


# ---------------------------------------------------------------------------
# Pembangkit korpus
# ---------------------------------------------------------------------------
def _price(rng: random.Random, low: int, high: int) -> int:
    """Harga kelipatan 100, seperti struk sungguhan."""
    return rng.randrange(low // 100, high // 100 + 1) * 100


def _rule(rng: random.Random) -> str:
    return "-" * rng.randrange(20, 61)


def _item_line(rng: random.Random) -> tuple[str, int]:
    name = rng.choice(PRODUCTS)
    price = _price(rng, 1000, 60000)
    return f"{name} {format_price(price)}", price


def _qty_line(rng: random.Random, unit: int, qty: int) -> str:
    return f"{qty} X {format_price(unit)}  {format_price(unit * qty)}"


def _summary_line(rng: random.Random, label: str, value: int) -> str:
    return f"{label} {format_price(value)}"


def _receipt_block(rng: random.Random) -> list[str]:
    """Satu struk sintetis: header, item, ringkasan, footer."""
    out: list[str] = []
    out.append(rng.choice(STORES))
    out.append(rng.choice(TAGLINES))
    out.append(rng.choice(ADDRESSES))
    out.append(rng.choice(CITIES))
    out.append(rng.choice(STORE_LINES))
    out.append(_rule(rng))
    out.append(rng.choice(TRANSACTION_META))
    out.append(_rule(rng))
    out.append("ITEM")
    out.append(_rule(rng))

    items_total = 0
    n_items = rng.randrange(2, 8)
    for _ in range(n_items):
        if rng.random() < 0.35:
            # Item dengan kuantitas: nama diikuti baris qty.
            name = rng.choice(PRODUCTS)
            unit = _price(rng, 1000, 30000)
            qty = rng.randrange(2, 6)
            out.append(name)
            out.append(_qty_line(rng, unit, qty))
            items_total += unit * qty
        else:
            line, price = _item_line(rng)
            out.append(line)
            items_total += price

    out.append(_rule(rng))
    out.append(_summary_line(rng, "TOTAL", items_total))

    if rng.random() < 0.4:
        discount = _price(rng, 100, min(5000, items_total // 10))
        out.append(_summary_line(rng, "DISKON", discount))
        items_total -= discount
        out.append(_summary_line(rng, "TOTAL", items_total))

    if rng.random() < 0.3:
        rounding = _price(rng, 100, 1000)
        out.append(_summary_line(rng, "PEMBULATAN", rounding))
        items_total += rounding

    dpp = int(items_total / 1.11)
    out.append(_summary_line(rng, "DPP", dpp))
    out.append(_summary_line(rng, "PPN 11%", items_total - dpp))
    out.append(_rule(rng))

    payment = rng.choice(
        ("TUNAI", "DEBIT BCA", "DEBIT MANDIRI", "QRIS", "OVO", "SHOPEEPAY")
    )
    if payment == "TUNAI":
        paid = items_total + _price(rng, 0, 10000)
        out.append(_summary_line(rng, "TUNAI", paid))
        out.append(_summary_line(rng, "KEMBALI", paid - items_total))
    else:
        out.append(_summary_line(rng, payment, items_total))

    out.append(_rule(rng))
    out.append(rng.choice(FOOTERS))
    out.append(rng.choice(FOOTERS))
    out.append(_rule(rng))
    return out


def build_corpus(n_lines: int = 15000, seed: int = 42) -> list[str]:
    """Bangkitkan ``n_lines`` baris struk sintetis yang deterministik."""
    rng = random.Random(seed)
    lines: list[str] = []
    while len(lines) < n_lines:
        lines.extend(_receipt_block(rng))
    return lines[:n_lines]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Bangkitkan korpus baris struk sintetis.")
    ap.add_argument("--n-lines", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    lines = build_corpus(args.n_lines, args.seed)
    kinds = line_kinds(lines)
    print(f"{len(lines)} baris, jenis: {kinds}")

    chars = {ch for line in lines for ch in line if ch != " "}
    print(f"karakter unik: {len(chars)} dari 42")

    if args.out:
        from pathlib import Path

        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"ditulis ke {args.out}")


if __name__ == "__main__":
    main()

def boilerplate_lines() -> list[str]:
    """Baris **teks tetap** struk: tagline, alamat, footer, label ringkasan.

    Dipakai sebagai templat pencocokan baris. Baris barang sengaja tidak
    dimasukkan: nama produk berubah-ubah dan tidak boleh "diperbaiki"
    menjadi produk lain hanya karena bentuknya mirip.
    """
    lines: list[str] = []
    for group in (
        STORES,
        TAGLINES,
        ADDRESSES,
        CITIES,
        STORE_LINES,
        FOOTERS,
        SUMMARY_LABELS,
        TRANSACTION_META,
    ):
        lines.extend(group)
    return lines
