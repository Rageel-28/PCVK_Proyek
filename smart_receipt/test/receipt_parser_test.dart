import 'package:flutter_test/flutter_test.dart';

import 'package:smart_receipt/ocr/ocr_raw_text.dart';
import 'package:smart_receipt/ocr/receipt_parser.dart';

/// Builds an [OcrRawResult] from plain lines, the way ML Kit hands them over.
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

/// Builds an [OcrRawResult] the way ML Kit returns a table: every cell is its
/// own `TextLine`, with a bounding box that reflects its column.
OcrRawResult _rawTable(List<List<String>> rows) {
  const double nameLeft = 20;
  const double nameRight = 280;
  const double qtyLeft = 300;
  const double qtyRight = 325;
  const double unitLeft = 340;
  const double unitRight = 405;
  const double totalLeft = 430;
  const double totalRight = 505;

  final List<OcrLine> lines = <OcrLine>[];
  final StringBuffer text = StringBuffer();
  double top = 0;

  void add(String value, double left, double right) {
    if (value.isEmpty) {
      return;
    }
    lines.add(OcrLine(
      text: value,
      left: left,
      top: top,
      right: right,
      bottom: top + 18,
    ));
    text.writeln(value);
  }

  for (final List<String> row in rows) {
    if (row.length == 1) {
      add(row[0], nameLeft, nameRight);
    } else if (row.length == 2) {
      // label + amount, e.g. "TOTAL :" + "33,900"
      add(row[0], nameLeft, nameRight);
      add(row[1], totalLeft, totalRight);
    } else {
      // name, qty, unit, total
      add(row[0], nameLeft, nameRight);
      add(row[1], qtyLeft, qtyRight);
      add(row[2], unitLeft, unitRight);
      add(row[3], totalLeft, totalRight);
    }
    top += 24;
  }

  return OcrRawResult(lines: lines, fullText: text.toString());
}

/// A real Indomaret receipt: dotted date, column layout without `x`, a
/// supplier header above the shop, and summary lines such as `HARGA JUAL`.
const List<String> _indomaretRows = <String>[
  'PT.INDING FOOD',
  'JL.LETJEN S.PARMAN-SUMEDANG',
  'KM.20 - JABAR SUM',
  'NPWP:22.131.107.6-446.001',
  'JATINANGOR KM.20-SUMEDA',
  '14.09.16- 06:46',
  '2.0. 31 914115/ALIA MU/01',
  'S/ROTI KRIM KEJU 72G 4 4500 18,000',
  'CIMORY MIX BERRY 22S 1 8500 8,500',
  'CAFELA ESPRESSO 200ML 1 3500 3,500',
  'FF LOW FAT VAN 225 1 5200 5,200',
  'PLASTIK SDG 1 1 1',
  'DISKON : (1)',
  'HARGA JUAL : 35,200',
  'DISKON FRISIAN FLAG : (1,300)',
  'TOTAL : 33,900',
  'TUNAI : 40,000',
  'KEMBALI : 6,100',
  'ANDA HEMAT : 1,300',
  'TERIMA KASIH. SELAMAT BELANJA KEMBALI',
  '======= LAYANAN KONSUMEN INDOMARET =======',
  'CALL 1500580 SMS : 0816 500 580',
  'EMAIL : KONTAK@INDOMARET.CO.ID',
  'WWW.INDOMARET.CO.ID/PROMO',
];

/// The same receipt when ML Kit splits every table cell onto its own line.
const List<String> _indomaretColumns = <String>[
  'PT.INDING FOOD',
  'JL.LETJEN S.PARMAN-SUMEDANG',
  'JATINANGOR KM.20-SUMEDA',
  '14.09.16- 06:46',
  '2.0. 31 914115/ALIA MU/01',
  '18,000',
  '4 4500',
  'S/ROTI KRIM KEJU 72G',
  '8,500',
  '1 8500',
  'CIMORY MIX BERRY 22S',
  '3,500',
  '1 3500',
  'CAFELA ESPRESSO 200ML',
  '5,200',
  '1 5200',
  'FF LOW FAT VAN 225',
  '1',
  '1 1',
  'PLASTIK SDG',
  'HARGA JUAL : 35,200',
  'TOTAL : 33,900',
  'TUNAI : 40,000',
  'KEMBALI : 6,100',
  '======= LAYANAN KONSUMEN INDOMARET =======',
];

const List<String> _expectedNames = <String>[
  'S/ROTI KRIM KEJU 72G',
  'CIMORY MIX BERRY 22S',
  'CAFELA ESPRESSO 200ML',
  'FF LOW FAT VAN 225',
  'PLASTIK SDG',
];

const List<int> _expectedQty = <int>[4, 1, 1, 1, 1];
const List<int> _expectedPrice = <int>[4500, 8500, 3500, 5200, 1];

void main() {
  group('ReceiptParser on a real Indomaret receipt', () {
    final ReceiptParser parser = ReceiptParser();

    for (final (String label, List<String> lines) in <(String, List<String>)>[
      ('single-line rows', _indomaretRows),
      ('column-split rows', _indomaretColumns),
    ]) {
      test('$label: extracts every item with its quantity and price', () {
        final ParsedReceipt p = parser.parse(_raw(lines));

        expect(p.items.map((ParsedItem i) => i.name).toList(), _expectedNames);
        expect(p.items.map((ParsedItem i) => i.qty).toList(), _expectedQty);
        expect(p.items.map((ParsedItem i) => i.price).toList(), _expectedPrice);
      });

      test('$label: uses the printed total, not the sum of columns', () {
        final ParsedReceipt p = parser.parse(_raw(lines));
        expect(p.total, 33900);
      });

      test('$label: reads the dotted date and colon time', () {
        final ParsedReceipt p = parser.parse(_raw(lines));
        expect(p.date, DateTime(2016, 9, 14, 6, 46));
      });
    }

    test('prefers the merchant name printed in the footer over the supplier',
        () {
      final ParsedReceipt p = parser.parse(_raw(_indomaretRows));
      expect(p.store, 'Indomaret');
    });

    test('does not turn summary or metadata lines into items', () {
      final ParsedReceipt p = parser.parse(_raw(_indomaretRows));
      final List<String> names =
          p.items.map((ParsedItem i) => i.name.toUpperCase()).toList();

      // Register/cashier metadata ("2.0. 31 914115/ALIA MU/01").
      expect(names.any((String n) => n.contains('914115')), isFalse);
      // Summary lines that used to be parsed as a 35,200 item.
      expect(names.any((String n) => n.contains('HARGA JUAL')), isFalse);
      expect(names.any((String n) => n.contains('DISKON')), isFalse);
      expect(names.any((String n) => n.contains('TOTAL')), isFalse);
    });
  });

  group('ReceiptParser with ML Kit table geometry', () {
    final ReceiptParser parser = ReceiptParser();

    /// The same Indomaret receipt as `_indomaretRows`, but every table cell is
    /// its own OCR line with a bounding box — which is what ML Kit really
    /// emits for this layout.
    final List<List<String>> tableRows = <List<String>>[
      <String>['PT.INDING FOOD'],
      <String>['JL.LETJEN S.PARMAN-SUMEDANG'],
      <String>['KM.20 - JABAR SUM'],
      <String>['NPWP:22.131.107.6-446.001'],
      <String>['JATINANGOR KM.20-SUMEDA'],
      <String>['14.09.16- 06:46'],
      <String>['2.0. 31 914115/ALIA MU/01'],
      <String>['S/ROTI KRIM KEJU 72G', '4', '4500', '18,000'],
      <String>['CIMORY MIX BERRY 22S', '1', '8500', '8,500'],
      <String>['CAFELA ESPRESSO 200ML', '1', '3500', '3,500'],
      <String>['FF LOW FAT VAN 225', '1', '5200', '5,200'],
      <String>['PLASTIK SDG', '1', '1', '1'],
      <String>['DISKON :', '(1)'],
      <String>['HARGA JUAL :', '35,200'],
      <String>['DISKON FRISIAN FLAG :', '(1,300)'],
      <String>['TOTAL :', '33,900'],
      <String>['TUNAI :', '40,000'],
      <String>['KEMBALI :', '6,100'],
      <String>['ANDA HEMAT :', '1,300'],
      <String>['TERIMA KASIH. SELAMAT BELANJA KEMBALI'],
      <String>['======= LAYANAN KONSUMEN INDOMARET ======='],
      <String>['CALL 1500580 SMS : 0816 500 580'],
      <String>['EMAIL : KONTAK@INDOMARET.CO.ID'],
      <String>['WWW.INDOMARET.CO.ID/PROMO'],
    ];

    test('assigns each price to the right item', () {
      final ParsedReceipt p = parser.parse(_rawTable(tableRows));

      expect(p.items.map((ParsedItem i) => i.name).toList(), _expectedNames);
      expect(p.items.map((ParsedItem i) => i.qty).toList(), _expectedQty);
      expect(p.items.map((ParsedItem i) => i.price).toList(), _expectedPrice);
      expect(
        p.items.map((ParsedItem i) => i.subtotal).toList(),
        <int>[18000, 8500, 3500, 5200, 1],
      );
    });

    test('reads the printed total, not the sum of the items', () {
      final ParsedReceipt p = parser.parse(_rawTable(tableRows));
      expect(p.total, 33900);
      expect(p.paymentMethod, 'Tunai');
    });

    test('reconciles item sum with the printed total through discounts', () {
      final ParsedReceipt p = parser.parse(_rawTable(tableRows));
      // 18,000 + 8,500 + 3,500 + 5,200 + 1 = 35,201
      expect(p.itemsTotal, 35201);
      // DISKON (1) + DISKON FRISIAN FLAG (1,300) = 1,301
      expect(p.discountTotal, 1301);
      expect(p.total + p.discountTotal, p.itemsTotal);
      expect(p.difference, -1301);
      expect(p.totalFromReceipt, isTrue);
      // The receipt adds up, so nothing should be flagged for the user.
      expect(p.warnings, isEmpty);
    });

    test('label and amount on separate lines still resolve', () {
      // Same receipt, but OCR put the total's value on its own line.
      final List<List<String>> split = <List<String>>[
        for (final List<String> row in tableRows)
          if (row.length == 2 && row[0].startsWith('TOTAL'))
            ...<List<String>>[
              <String>['TOTAL :'],
              <String>[row[1]],
            ]
          else
            row,
      ];
      final ParsedReceipt p = parser.parse(_rawTable(split));
      expect(p.total, 33900);
    });

    test('geometry that carries no layout falls back to text parsing', () {
      // Every line shares one box: no columns to group by.
      final ParsedReceipt p = parser.parse(_raw(_indomaretRows));
      expect(p.items.map((ParsedItem i) => i.name).toList(), _expectedNames);
      expect(p.total, 33900);
    });
  });

  group('ReceiptParser repairs OCR damage', () {
    final ReceiptParser parser = ReceiptParser();

    ParsedReceipt parseWith(List<String> lines) => parser.parse(_raw(lines));

    test('joins amounts split at the thousands separator', () {
      final ParsedReceipt p = parseWith(<String>[
        'INDOMARET',
        'S/ROTI KRIM KEJU 72G 4 4500 18, 000',
        'CIMORY MIX BERRY 22S 1 8500 8, 500',
        'PLASTIK SDG 1 1 1',
        'TOTAL : 27, 501',
      ]);

      expect(p.items.map((ParsedItem i) => i.name).toList(), <String>[
        'S/ROTI KRIM KEJU 72G',
        'CIMORY MIX BERRY 22S',
        'PLASTIK SDG',
      ]);
      expect(p.items.map((ParsedItem i) => i.subtotal).toList(),
          <int>[18000, 8500, 1]);
      expect(p.total, 27501);
    });

    test('joins a line total split across two OCR tokens', () {
      final ParsedReceipt p = parseWith(<String>[
        'INDOMARET',
        'CIMORY MIX BERRY 22S 1 8500 8, 500',
        'TOTAL : 8, 500',
      ]);

      expect(p.items.single.subtotal, 8500);
      expect(p.items.single.price, 8500);
      expect(p.total, 8500);
    });

    test('recognises a merchant name mangled by OCR', () {
      final ParsedReceipt p = parseWith(<String>[
        'Sndon Oomane',
        'JATINANGOR KM. 20-SUMEDA',
        'PLASTIK SDG 1 1 1',
        'CALL 1500580 SUMEN INDOMAREMBALI',
      ]);

      expect(p.store, 'Indomaret');
    });

    test('recovers amounts whose thousands dot was misread as a letter', () {
      // OCR reads the dot (.) in `8.700` as `O`/`M`/`+` and zeros as `O`.
      expect(parser.amountValue('8O700'), 8700);
      expect(parser.amountValue('3M5OO'), 3500);
      expect(parser.amountValue('16+0OO'), 16000);
      expect(parser.amountValue('34O8OO'), 34800);
      expect(parser.amountValue('7OOOO'), 7000);
      expect(parser.amountValue('67O3OO'), 67300);
      expect(parser.amountValue('6OO631'), 60631);
      expect(parser.amountValue('6O669'), 6669);
      // Clean amounts still parse (regression guard).
      expect(parser.amountValue('8.700'), 8700);
      expect(parser.amountValue('18,000'), 18000);
      expect(parser.amountValue('4500'), 4500);
    });

    test('does not turn a phone number into a mangled amount', () {
      // A 7-digit number with a stray letter is not a valid N.NNN amount.
      expect(parser.amountValue('4D95863'), isNull);
      // A clean 7-digit number is a plain integer, not NNN.NNN.
      expect(parser.amountValue('4095863'), 4095863);
    });

    test('parses a two-line Indomaret item with mangled dots', () {
      final ParsedReceipt p = parseWith(<String>[
        'INDOMARET',
        'STORE NO. 0001 (021) 4095863',
        'NO : 39617647 14/07/2023',
        'KSR : D. ANGGRAINI 15:05:33',
        'ITEM TOTAL',
        'POCARI SWEAT 500ML',
        '4 X 8O700 34O8OO',
        'ANTANGIN JRG SACHET',
        '2 X 3M5OO 7OOOO',
        'ROTI TAWAR SARI ROTI 16+0OO',
        'KOPI KAPAL API 65G 9O500',
        'TOTAL 67O3OO',
        'DPP 6OO631',
        'PPN 11% 6O669',
        'DEBIT BCA 67O3OO',
      ]);

      expect(
        p.items.map((ParsedItem i) => i.name).toList(),
        <String>[
          'POCARI SWEAT 500ML',
          'ANTANGIN JRG SACHET',
          'ROTI TAWAR SARI ROTI',
          'KOPI KAPAL API 65G',
        ],
      );
      expect(p.items.map((ParsedItem i) => i.qty).toList(), <int>[4, 2, 1, 1]);
      expect(p.items.map((ParsedItem i) => i.price).toList(),
          <int>[8700, 3500, 16000, 9500]);
      expect(p.items.map((ParsedItem i) => i.subtotal).toList(),
          <int>[34800, 7000, 16000, 9500]);
      expect(p.total, 67300);
      expect(p.totalFromReceipt, isTrue);
    });
  });

  group('ReceiptParser with column-block OCR order', () {
    final ReceiptParser parser = ReceiptParser();

    /// The exact failure mode seen on a real device: ML Kit returns the receipt
    /// **block by block**, so every product name arrives before any number.
    /// Only the bounding boxes connect `S/ROTI...` to `4`, `4500`, `18,000`.
    test('pairs names and numbers from geometry, not text order', () {
      const List<String> names = <String>[
        'S/ROTI KRIM KEJU 72G',
        'CIMORY MIX BERRY 225',
        'CAFELA EXPRESO 200ML',
        'FF LOW FAT VAN 225',
        'PLASTIK SDG',
      ];
      const List<String> qty = <String>['4', '1', '1', '1', '1'];
      const List<String> unit = <String>['4500', '8500', '3500', '5200', '1'];
      const List<String> total = <String>[
        '18, 000',
        '8,',
        '3, 500',
        '5, 200',
        '1',
      ];

      final List<OcrLine> lines = <OcrLine>[];
      void add(String text, double left, double right, int row) {
        lines.add(OcrLine(
          text: text,
          left: left,
          top: 100 + row * 30.0,
          right: right,
          bottom: 100 + row * 30.0 + 20,
        ));
      }

      // Header, then the name column, then each numeric column — the order ML
      // Kit actually emits on device.
      add('Sndon Oomane', 40, 400, -2);
      add('JATINANGOR KM. 20-SUMEDA', 40, 400, -1);
      for (int i = 0; i < names.length; i++) {
        add(names[i], 40, 400, i);
      }
      for (int i = 0; i < qty.length; i++) {
        add(qty[i], 420, 460, i);
      }
      for (int i = 0; i < unit.length; i++) {
        add(unit[i], 470, 560, i);
      }
      for (int i = 0; i < total.length; i++) {
        // "8," and "500" land on separate lines but the same row.
        if (total[i] == '8,') {
          add('8,', 570, 600, i);
          add('500', 605, 680, i);
        } else {
          add(total[i], 570, 680, i);
        }
      }
      add('DISKON :', 40, 300, names.length);
      add('TOTAL :', 40, 300, names.length + 1);
      add('33, 900', 570, 680, names.length + 1);
      add('TUNAI :', 40, 300, names.length + 2);
      add('40', 570, 680, names.length + 2);

      final ParsedReceipt p =
          parser.parse(OcrRawResult(lines: lines, fullText: ''));

      expect(p.items.map((ParsedItem i) => i.name).toList(), <String>[
        'S/ROTI KRIM KEJU 72G',
        'CIMORY MIX BERRY 225',
        'CAFELA EXPRESO 200ML',
        'FF LOW FAT VAN 225',
        'PLASTIK SDG',
      ]);
      expect(p.items.map((ParsedItem i) => i.qty).toList(),
          <int>[4, 1, 1, 1, 1]);
      expect(p.items.map((ParsedItem i) => i.price).toList(),
          <int>[4500, 8500, 3500, 5200, 1]);
      expect(p.items.map((ParsedItem i) => i.subtotal).toList(),
          <int>[18000, 8500, 3500, 5200, 1]);
    });
  });

  group('ReceiptParser on ragged ML Kit columns', () {
    final ReceiptParser parser = ReceiptParser();

    /// Reproduces the device output that broke the old parser: text blocks are
    /// emitted column by column, the quantity column has fewer lines than the
    /// names (ML Kit merged cells), and `8,500` is split across two boxes.
    test('attaches numbers by vertical proximity, not line count', () {
      final List<OcrLine> lines = <OcrLine>[];
      void add(String text, double left, double right, double top) {
        lines.add(OcrLine(
          text: text,
          left: left,
          top: top,
          right: right,
          bottom: top + 20,
        ));
      }

      const List<String> names = <String>[
        'S/ROTI KRIM KEJU 72G',
        'CIMORY MIX BERRY 225',
        'CAFELA EXPRESO 200ML',
        'FF LOW FAT VAN 225',
        'PLASTIK SDG',
      ];
      const List<String> units = <String>['4500', '8500', '3500', '5200', '1'];

      add('Sndon Omane', 40, 400, 40);
      add('JATINANGOR KM. 20-SUMEDA', 40, 400, 70);
      add('14. 09. 16- 06:46', 40, 400, 100);
      add('2.0. 31 914115/ALIA MU/01', 40, 400, 130);
      for (int i = 0; i < names.length; i++) {
        add(names[i], 40, 400, 160 + i * 30);
      }
      // Only two quantity cells survived.
      add('4', 420, 460, 160);
      add('1', 420, 460, 190);
      for (int i = 0; i < units.length; i++) {
        add(units[i], 470, 560, 160 + i * 30);
      }
      // Line totals; the second is split and slightly offset.
      add('18, 000', 570, 680, 160);
      add('8,', 570, 600, 190);
      add('500', 605, 680, 194);
      add('3, 500', 570, 680, 220);
      add('5, 200', 570, 680, 250);
      add('1', 570, 600, 280);
      // Summary block.
      add('DISKON :', 40, 300, 310);
      add('(1)', 570, 620, 310);
      add('HARGA JUAL :', 40, 300, 340);
      add('35, 200', 570, 680, 340);
      add('DISKON FRISIAN FLAG :', 40, 300, 370);
      add('(1, 300)', 570, 680, 370);
      add('TOTAL :', 40, 300, 400);
      add('33, 900', 570, 680, 400);
      add('TUNAI :', 40, 300, 430);
      add('40', 570, 600, 430);
      add('000', 605, 680, 434);
      add('KEMBALI :', 40, 300, 460);
      add('6, 100', 570, 680, 460);
      add('CALL 1500580 SUMEN INDOMAREMBALI', 40, 400, 490);

      final ParsedReceipt p =
          parser.parse(OcrRawResult(lines: lines, fullText: ''));

      expect(p.items.map((ParsedItem i) => i.name).toList(), names);
      expect(p.items.map((ParsedItem i) => i.qty).toList(),
          <int>[4, 1, 1, 1, 1]);
      expect(p.items.map((ParsedItem i) => i.price).toList(),
          <int>[4500, 8500, 3500, 5200, 1]);
      expect(p.items.map((ParsedItem i) => i.subtotal).toList(),
          <int>[18000, 8500, 3500, 5200, 1]);
      expect(p.total, 33900);
      expect(p.store, 'Indomaret');
    });
  });

  group('ReceiptParser date formats', () {
    final ReceiptParser parser = ReceiptParser();

    ParsedReceipt parseWithDate(String line) => parser.parse(_raw(<String>[
          'TOKO MAJU',
          line,
          'INDOMIE GORENG 7,000',
          'TOTAL 7,000',
        ]));

    test('accepts dd.mm.yy', () {
      expect(parseWithDate('14.09.16 - 06:46').date,
          DateTime(2016, 9, 14, 6, 46));
    });

    test('accepts dd/mm/yyyy', () {
      expect(parseWithDate('14/09/2026 19:14').date,
          DateTime(2026, 9, 14, 19, 14));
    });

    test('does not read a dotted amount as a date', () {
      final ParsedReceipt p = parser.parse(_raw(<String>[
        'TOKO MAJU',
        'INDOMIE GORENG 7.000',
        'TOTAL 7.000',
      ]));
      expect(p.date, isNull);
    });
  });
}
