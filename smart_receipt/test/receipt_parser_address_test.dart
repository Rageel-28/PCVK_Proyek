/// Regresi: baris alamat/kontak toko tidak boleh menjadi barang atau nama toko.
///
/// Struk minimarket Indonesia mencetak alamat lengkap tepat di bawah nama
/// toko, misalnya:
///
///     INDOMARET PESONA ANGGREK
///     KEBUMEN 0 0287386631
///     JL. PEMUDA. KEL.KEBUMEN. KEC.KEBUMEN. KAB.
///     KEBUMEN. 54330
///     TEH GLS LESS.SGR 350   2   3800   6.200
///
/// Kode pos lima digit di ujung baris alamat terbaca sebagai harga, sehingga
/// alamat muncul sebagai barang seharga 54.330. Baris alamat juga sering
/// terpilih sebagai nama toko karena nama merchant gagal dikenali OCR.
library;

import 'package:flutter_test/flutter_test.dart';
import 'package:smart_receipt/ocr/ocr_raw_text.dart';
import 'package:smart_receipt/ocr/receipt_parser.dart';

OcrRawResult _raw(List<String> lines) => OcrRawResult(
      lines: <OcrLine>[
        for (int i = 0; i < lines.length; i++)
          OcrLine(
            text: lines[i],
            left: 0,
            top: i * 20.0,
            right: 100,
            bottom: i * 20.0 + 18,
          ),
      ],
      fullText: lines.join('\n'),
    );

void main() {
  final ReceiptParser parser = ReceiptParser();

  test('baris alamat dengan kode pos bukan barang', () {
    final ParsedReceipt p = parser.parse(_raw(<String>[
      'INDOMARET PESONA ANGGREK',
      'KEBUMEN 0 0287386631',
      'JL. PEMUDA. KEL.KEBUMEN. KEC.KEBUMEN. KAB.',
      'KEBUMEN. 54330',
      'TEH GLS LESS.SGR 350 2 3800 6.200',
      'TOTAL 3.800',
    ]));

    expect(p.items, hasLength(1));
    expect(p.items.single.name.toUpperCase(), contains('TEH'));
    expect(p.total, 3800);
  });

  test('NPWP, telepon, dan RT/RW bukan barang', () {
    final ParsedReceipt p = parser.parse(_raw(<String>[
      'ALFAMART CINERE',
      'NPWP 01.234.567.8-901.000',
      'TELP 021-5551234',
      'RT. 04 RW. 02 KEL. CINERE',
      'SARI ROTI 4500',
      'TOTAL 4.500',
    ]));

    expect(p.items, hasLength(1));
    expect(p.items.single.name.toUpperCase(), contains('SARI'));
    expect(p.total, 4500);
  });

  test('nama toko dipilih dari baris merchant, bukan alamat', () {
    final ParsedReceipt p = parser.parse(_raw(<String>[
      'JL. PEMUDA NO 12',
      'ALFAMART CINERE',
      'KEC. CINERE KAB. DEPOK 16511',
      'SARI ROTI 4500',
      'TOTAL 4.500',
    ]));

    expect(p.store.toUpperCase(), contains('ALFAMART'));
    expect(p.store.toUpperCase(), isNot(contains('JL.')));
  });

  test('baris barang tidak pernah menjadi nama toko', () {
    final ParsedReceipt p = parser.parse(_raw(<String>[
      'KEBUMEN 0 0287386631',
      'JL. PEMUDA. KEL.KEBUMEN. KEC.KEBUMEN. KAB.',
      'KEBUMEN. 54330',
      'TEH GLS LESS.SGR 350 2 3800 6.200',
      'TOTAL 3.800',
    ]));

    expect(p.store.toUpperCase(), isNot(contains('TEH')));
    expect(p.store.toUpperCase(), isNot(contains('3800')));
    expect(p.store, isNot(contains('54330')));
  });
}
