/// Regresi: tahun struk tidak boleh jatuh di masa depan.
///
/// OCR sering membaca `19` sebagai `34` pada tanggal `19-05-2019`. Tanpa batas
/// atas, `_parseDateCandidate` memetakan dua digit tahun apa pun ke `2000 + yy`
/// sehingga tanggal struk menjadi 2034 — tahun yang belum terjadi saat struk
/// dipindai.
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

  ParsedReceipt parseWithDate(String line) => parser.parse(_raw(<String>[
        'TOKO MAJU',
        line,
        'INDOMIE GORENG 7,000',
        'TOTAL 7,000',
      ]));

  test('tanggal bertahun masa depan ditolak', () {
    final ParsedReceipt p = parseWithDate('02.09.34 - 06:46');
    expect(p.date, isNull);
  });

  test('tahun yang masih masuk akal tetap diterima', () {
    expect(
      parseWithDate('14.09.26 - 06:46').date,
      DateTime(2026, 9, 14, 6, 46),
    );
  });

  test('tanggal empat digit di masa depan juga ditolak', () {
    final ParsedReceipt p = parseWithDate('02/09/2034 19:14');
    expect(p.date, isNull);
  });
}
