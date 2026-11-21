import 'package:flutter_test/flutter_test.dart';
import 'package:smart_receipt/ocr/ocr_raw_text.dart';
import 'package:smart_receipt/ocr/receipt_parser.dart';

void main() {
  test('debug widget fixture', () {
    const List<String> deviceLines = <String>[
      'INDOMARET', 'Jl. Sudirman No. 12, Jakarta', 'NPWP: 01.234.567.8-901.000',
      '13/09/ 2026 19:14:32', 'No: 0123', 'Kasir: RINA', '7.000', '2 x 3.500',
      'INDOMIE GORENG', '4.000', '1 X 4.000', 'AQUA 60OML', '5.500',
      '1 x 5.500', 'SABUN LIFEBUOY', '2 X 12.000 24.000', 'TISSUE PASEO',
      '5.000', '1 x 5.000', 'TEH BOTOL SOSRO', '1 x 14.000 14.000',
      'ROTI TAWAR', '59.500', 'SUBTOTAL', '100.000', 'TUNAI', '40.500',
      'KEMBALI',
    ];
    final OcrRawResult raw = OcrRawResult(
      fullText: deviceLines.join('\n'),
      lines: <OcrLine>[
        for (int i = 0; i < deviceLines.length; i++)
          OcrLine(
            text: deviceLines[i],
            left: 0,
            top: i * 30.0,
            right: 100,
            bottom: i * 30.0 + 20,
          ),
      ],
    );
    final ParsedReceipt parsed = ReceiptParser().parse(raw);
    // ignore: avoid_print
    print('total=${parsed.total} (struk=${parsed.totalFromReceipt}) '
        'itemsTotal=${parsed.itemsTotal} source=${parsed.source}');
    for (final i in parsed.items) {
      // ignore: avoid_print
      print('  "${i.name}" ${i.qty} x ${i.price} = ${i.lineTotal}');
    }
    for (final r in parsed.rowDump) {
      // ignore: avoid_print
      print('  ROW: $r');
    }
  });
}
