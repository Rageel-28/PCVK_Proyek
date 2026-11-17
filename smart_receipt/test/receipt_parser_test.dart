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

  group('ReceiptParser real-OCR edge cases', () {
    final ReceiptParser parser = ReceiptParser();

    test('stitches 2-line items when 1x is read as IX and recovers Alfa,art', () {
      final ParsedReceipt p = parser.parse(_raw(<String>[
        'Alfa,art Jogja',
        'Cimory',
        'IX 2.000 2.000',
        'Frestea madu',
        'IX 8.000 8.000',
        'Total Rp 10.000',
      ]));
      expect(p.store, 'Alfamart');
      expect(p.total, 10000);
      expect(p.items.map((ParsedItem i) => i.name).toList(), <String>[
        'Cimory',
        'Frestea madu',
      ]);
      expect(p.items.map((ParsedItem i) => i.subtotal).toList(),
          <int>[2000, 8000]);
    });

    test('rejects DISC and OCR-corrupted footer labels (T.TAL LTO, TGMAI, KEMBAIIAD)', () {
      final ParsedReceipt p = parser.parse(_raw(<String>[
        'ALFAMART',
        'KP BRMDLNG (S) 1 500 500',
        'PIUK 350M 2 3.000 6.000',
        'DISC -1.000',
        'T.TAL LTO 6.500',
        'TOTAL 5.500',
        'TGMAI 10.000',
        'KEMBAIIAD 4.500',
      ]));
      expect(p.items.map((ParsedItem i) => i.name).toList(), <String>[
        'KP BRMDLNG (S)',
        'PIUK 350M',
      ]);
      expect(p.total, 5500);
      expect(p.discountTotal, 1000);
    });

    test('recovers qty=1 when 1x is read as 18 on equal prices and drops 18 5.OAD remnant', () {
      final ParsedReceipt p = parser.parse(_raw(<String>[
        'ALFAMART',
        '18 5.OAD 5.000',
        'KANZLER 18 10.000 10.000',
        'LE MINERANE IX 4.000 4.000',
        '7GTAL Rp 14.000',
      ]));
      expect(p.total, 14000);
      expect(p.items.map((ParsedItem i) => i.name).toList(), <String>[
        'KANZLER',
        'LE MINERANE',
      ]);
      expect(p.items.map((ParsedItem i) => i.qty).toList(), <int>[1, 1]);
    });

    test('stitches 2-line item for leading LX while preserving trailing LX size label', () {
      final ParsedReceipt p = parser.parse(_raw(<String>[
        'ALFAMART',
        'CIMORY',
        'LX 2.000 2.000',
        'PACK LX 10.000',
        'TOTAL 12.000',
      ]));
      expect(p.total, 12000);
      expect(p.items.map((ParsedItem i) => i.name).toList(), <String>[
        'CIMORY',
        'PACK LX',
      ]);
      expect(p.items.map((ParsedItem i) => i.qty).toList(), <int>[1, 1]);
      expect(p.items.map((ParsedItem i) => i.subtotal).toList(),
          <int>[2000, 10000]);
    });

    test('stitches stacked L8/T8 multipliers and recovers trailing G.ODO/E.OOO copies', () {
      final ParsedReceipt p = parser.parse(_raw(<String>[
        'ALFAMART',
        'CIMORY',
        'L8 2.000 2.000',
        'CIMORY HAZELNUT',
        '18 9.000 G.ODO',
        'XCE CREAB AICE',
        'T8 10.000 10.000',
        'KANZLER',
        '18 8.000 E.OOO',
        'TOTAL RP38.000',
      ]));
      expect(p.store, 'Alfamart');
      expect(p.total, 38000);
      expect(p.items.map((ParsedItem i) => i.name).toList(), <String>[
        'CIMORY',
        'CIMORY HAZELNUT',
        'XCE CREAB AICE',
        'KANZLER',
      ]);
      expect(p.items.map((ParsedItem i) => i.qty).toList(), <int>[1, 1, 1, 1]);
      expect(p.items.map((ParsedItem i) => i.subtotal).toList(),
          <int>[2000, 9000, 10000, 8000]);
    });

    test('recovers middle unit price on 2 3,MO 7.800 (Alfamart Soetomo word boxes)', () {
      final List<OcrLine> lines = <OcrLine>[
        const OcrLine(text: 'ALFAMART', left: 196, top: 217, right: 400, bottom: 268),
        const OcrLine(text: 'KP', left: 19, top: 798, right: 62, bottom: 845),
        const OcrLine(text: 'BRMDLNG', left: 84, top: 797, right: 251, bottom: 845),
        const OcrLine(text: '(S)', left: 276, top: 797, right: 334, bottom: 848),
        const OcrLine(text: '1', left: 465, top: 798, right: 478, bottom: 847),
        const OcrLine(text: '200', left: 591, top: 798, right: 654, bottom: 847),
        const OcrLine(text: '200', left: 803, top: 804, right: 867, bottom: 851),
        const OcrLine(text: 'PIUK', left: 18, top: 867, right: 124, bottom: 917),
        const OcrLine(text: '350N', left: 148, top: 867, right: 250, bottom: 915),
        const OcrLine(text: '2', left: 464, top: 867, right: 484, bottom: 915),
        const OcrLine(text: '3,MO', left: 548, top: 868, right: 653, bottom: 917),
        const OcrLine(text: '7.800', left: 760, top: 873, right: 866, bottom: 920),
        const OcrLine(text: 'DISC.', left: 16, top: 940, right: 117, bottom: 987),
        const OcrLine(text: '-200', left: 252, top: 937, right: 335, bottom: 985),
        const OcrLine(text: 'TOTAL', left: 13, top: 1223, right: 113, bottom: 1272),
        const OcrLine(text: 'BELANJA', left: 143, top: 1221, right: 291, bottom: 1271),
        const OcrLine(text: '7.800', left: 756, top: 1221, right: 862, bottom: 1272),
        const OcrLine(text: 'TUNAI', left: 12, top: 1294, right: 113, bottom: 1341),
        const OcrLine(text: '7.800', left: 756, top: 1291, right: 861, bottom: 1341),
      ];
      final ParsedReceipt p = parser.parse(OcrRawResult(lines: lines, fullText: ''));
      expect(p.store, 'Alfamart');
      expect(p.total, 7800);
      expect(p.discountTotal, 200);
      expect(p.items.map((ParsedItem i) => i.name).toList(), <String>[
        'KP BRMDLNG (S)',
        'PIUK 350N',
      ]);
      expect(p.items.map((ParsedItem i) => i.qty).toList(), <int>[1, 2]);
      expect(p.items.map((ParsedItem i) => i.price).toList(), <int>[200, 3900]);
      expect(p.items.map((ParsedItem i) => i.subtotal).toList(), <int>[200, 7800]);
    });

    test('clusters word-level anchors & parses 18, OOO and KENXBALL/ANDAHEVT on Indomaret', () {
      final List<OcrLine> lines = <OcrLine>[
        const OcrLine(text: 'S/ROTL', left: 121, top: 331, right: 224, bottom: 355),
        const OcrLine(text: 'KRIV', left: 273, top: 332, right: 350, bottom: 355),
        const OcrLine(text: 'KEIU', left: 388, top: 331, right: 458, bottom: 355),
        const OcrLine(text: '723', left: 495, top: 331, right: 553, bottom: 355),
        const OcrLine(text: '4', left: 686, top: 332, right: 703, bottom: 355),
        const OcrLine(text: '4500', left: 758, top: 331, right: 837, bottom: 355),
        const OcrLine(text: '18,', left: 894, top: 331, right: 941, bottom: 355),
        const OcrLine(text: 'OOO', left: 956, top: 331, right: 1017, bottom: 355),
        const OcrLine(text: 'CLVORY', left: 118, top: 373, right: 242, bottom: 397),
        const OcrLine(text: 'VLX', left: 268, top: 373, right: 324, bottom: 396),
        const OcrLine(text: 'BERRY', left: 350, top: 373, right: 445, bottom: 396),
        const OcrLine(text: '225', left: 470, top: 373, right: 523, bottom: 397),
        const OcrLine(text: '1', left: 688, top: 373, right: 702, bottom: 395),
        const OcrLine(text: '8500', left: 763, top: 373, right: 837, bottom: 397),
        const OcrLine(text: '8.500', left: 926, top: 373, right: 1018, bottom: 397),
        const OcrLine(text: 'CAFELA', left: 118, top: 417, right: 233, bottom: 440),
        const OcrLine(text: 'EXPRESS', left: 259, top: 417, right: 394, bottom: 440),
        const OcrLine(text: 'ZOOVL', left: 420, top: 417, right: 524, bottom: 440),
        const OcrLine(text: '1', left: 688, top: 415, right: 702, bottom: 438),
        const OcrLine(text: '3500', left: 763, top: 415, right: 837, bottom: 439),
        const OcrLine(text: '3.500', left: 926, top: 415, right: 1018, bottom: 440),
        const OcrLine(text: 'FF', left: 119, top: 459, right: 148, bottom: 481),
        const OcrLine(text: 'LOVV', left: 181, top: 459, right: 247, bottom: 481),
        const OcrLine(text: 'FAT', left: 280, top: 459, right: 328, bottom: 481),
        const OcrLine(text: 'VAN', left: 357, top: 459, right: 417, bottom: 481),
        const OcrLine(text: '225', left: 451, top: 459, right: 502, bottom: 481),
        const OcrLine(text: '1', left: 688, top: 460, right: 702, bottom: 483),
        const OcrLine(text: '5200', left: 763, top: 460, right: 837, bottom: 483),
        const OcrLine(text: '5.200', left: 926, top: 460, right: 1018, bottom: 483),
        const OcrLine(text: 'PLASTLKSDG', left: 119, top: 500, right: 310, bottom: 523),
        const OcrLine(text: '1', left: 688, top: 500, right: 702, bottom: 523),
        const OcrLine(text: '1', left: 818, top: 500, right: 832, bottom: 523),
        const OcrLine(text: '1', left: 1003, top: 500, right: 1017, bottom: 523),
        const OcrLine(text: 'DISKON', left: 688, top: 550, right: 797, bottom: 574),
        const OcrLine(text: ':', left: 833, top: 557, right: 836, bottom: 573),
        const OcrLine(text: '(', left: 962, top: 550, right: 968, bottom: 574),
        const OcrLine(text: '1', left: 983, top: 550, right: 996, bottom: 574),
        const OcrLine(text: ')', left: 1009, top: 550, right: 1016, bottom: 574),
        const OcrLine(text: 'HARGA', left: 538, top: 625, right: 657, bottom: 648),
        const OcrLine(text: 'IUAL', left: 698, top: 626, right: 779, bottom: 648),
        const OcrLine(text: ':', left: 835, top: 632, right: 838, bottom: 647),
        const OcrLine(text: '35.200', left: 888, top: 625, right: 1018, bottom: 648),
        const OcrLine(text: 'DISKON', left: 394, top: 665, right: 520, bottom: 686),
        const OcrLine(text: 'FRISIAN', left: 542, top: 665, right: 675, bottom: 686),
        const OcrLine(text: 'FLAM', left: 697, top: 665, right: 778, bottom: 686),
        const OcrLine(text: ':', left: 835, top: 669, right: 838, bottom: 684),
        const OcrLine(text: '1.300', left: 882, top: 665, right: 1017, bottom: 689),
        const OcrLine(text: 'TOTAL', left: 678, top: 736, right: 778, bottom: 759),
        const OcrLine(text: ':', left: 835, top: 740, right: 838, bottom: 756),
        const OcrLine(text: '33.900', left: 898, top: 736, right: 1018, bottom: 759),
        const OcrLine(text: 'TUNAI', left: 678, top: 778, right: 780, bottom: 801),
        const OcrLine(text: '40.000', left: 895, top: 778, right: 1017, bottom: 801),
        const OcrLine(text: 'KENXBALL', left: 638, top: 821, right: 780, bottom: 844),
        const OcrLine(text: '6.180', left: 918, top: 821, right: 1018, bottom: 844),
        const OcrLine(text: 'ANDAHEVT', left: 558, top: 865, right: 784, bottom: 889),
        const OcrLine(text: '1.380', left: 920, top: 864, right: 1018, bottom: 889),
        const OcrLine(text: 'LNDOVARET--==--', left: 651, top: 942, right: 991, bottom: 965),
      ];
      final ParsedReceipt p = parser.parse(OcrRawResult(lines: lines, fullText: ''));
      expect(p.total, 33900);
      expect(p.discountTotal, 1301);
      expect(p.items.length, 5);
      expect(p.items.map((ParsedItem i) => i.subtotal).toList(),
          <int>[18000, 8500, 3500, 5200, 1]);
      expect(p.warnings, isEmpty);
    });

    test('strips vertical border L/I margins and parses Alfamart Permata Gardenia', () {
      final List<OcrLine> lines = <OcrLine>[
        const OcrLine(text: 'I', left: 52, top: 317, right: 64, bottom: 451),
        const OcrLine(text: 'ALFAMART', left: 514, top: 317, right: 975, bottom: 451),
        const OcrLine(text: 'PERMATA', left: 1047, top: 317, right: 1459, bottom: 451),
        const OcrLine(text: 'GARDENLA', left: 1526, top: 317, right: 1998, bottom: 451),
        const OcrLine(text: 'L', left: 44, top: 2097, right: 57, bottom: 2227),
        const OcrLine(text: 'ALFA', left: 108, top: 2099, right: 341, bottom: 2226),
        const OcrLine(text: 'K.HJ', left: 405, top: 2097, right: 626, bottom: 2227),
        const OcrLine(text: '2506', left: 695, top: 2097, right: 926, bottom: 2227),
        const OcrLine(text: '1', left: 1350, top: 2097, right: 1380, bottom: 2227),
        const OcrLine(text: '9.600', left: 1583, top: 2097, right: 1874, bottom: 2227),
        const OcrLine(text: '9.600', left: 2180, top: 2097, right: 2468, bottom: 2227),
        const OcrLine(text: 'L', left: 2621, top: 2097, right: 2626, bottom: 2227),
        const OcrLine(text: 'L', left: 47, top: 2291, right: 58, bottom: 2422),
        const OcrLine(text: 'INDOMIE', left: 130, top: 2291, right: 513, bottom: 2419),
        const OcrLine(text: 'KOYA75G', left: 574, top: 2291, right: 985, bottom: 2422),
        const OcrLine(text: '1', left: 1349, top: 2291, right: 1379, bottom: 2422),
        const OcrLine(text: '2.600', left: 1583, top: 2291, right: 1873, bottom: 2422),
        const OcrLine(text: '2.600', left: 2180, top: 2291, right: 2467, bottom: 2422),
        const OcrLine(text: 'L', left: 2616, top: 2291, right: 2625, bottom: 2422),
        const OcrLine(text: 'I', left: 51, top: 2682, right: 64, bottom: 2810),
        const OcrLine(text: 'TOTAL', left: 113, top: 2682, right: 377, bottom: 2808),
        const OcrLine(text: 'ITEM', left: 479, top: 2682, right: 691, bottom: 2810),
        const OcrLine(text: '2', left: 988, top: 2683, right: 1041, bottom: 2810),
        const OcrLine(text: '12.200', left: 2120, top: 2682, right: 2465, bottom: 2810),
        const OcrLine(text: 'L', left: 2615, top: 2682, right: 2624, bottom: 2808),
        const OcrLine(text: 'L', left: 48, top: 2875, right: 60, bottom: 3001),
        const OcrLine(text: 'SHOPEEPAY', left: 113, top: 2875, right: 629, bottom: 3001),
        const OcrLine(text: '12.200', left: 2121, top: 2875, right: 2464, bottom: 3001),
        const OcrLine(text: 'L', left: 2614, top: 2875, right: 2623, bottom: 3001),
        const OcrLine(text: 'LL', left: 4, top: 4231, right: 39, bottom: 4363),
        const OcrLine(text: 'POLN', left: 575, top: 4231, right: 808, bottom: 4363),
        const OcrLine(text: 'MEMBER', left: 874, top: 4231, right: 1223, bottom: 4362),
        const OcrLine(text: 'ANDA', left: 1289, top: 4231, right: 1518, bottom: 4363),
        const OcrLine(text: '15075', left: 1591, top: 4231, right: 1875, bottom: 4363),
      ];
      final ParsedReceipt p = parser.parse(OcrRawResult(lines: lines, fullText: ''));
      expect(p.store, 'Alfamart');
      expect(p.paymentMethod, 'ShopeePay');
      expect(p.total, 12200);
      expect(p.items.map((ParsedItem i) => i.name).toList(), <String>[
        'ALFA K.HJ 2506',
        'INDOMIE KOYA75G',
      ]);
      expect(p.items.map((ParsedItem i) => i.subtotal).toList(), <int>[9600, 2600]);
      expect(p.warnings, isEmpty);
    });

    test('rejects 0-Rupiah noise rows on crumpled receipts', () {
      final ParsedReceipt p = parser.parse(_raw(<String>[
        'ALFAMIDI',
        'SOFTIES 1 15.900 15.900',
        '- UULOV UI R - M 1 4 1 N OINN 0',
        'IT- 0:.5M 0',
        'KEMBALIAN 0',
      ]));
      expect(p.items.length, 1);
      expect(p.items.single.subtotal, 15900);
    });
  });
}
