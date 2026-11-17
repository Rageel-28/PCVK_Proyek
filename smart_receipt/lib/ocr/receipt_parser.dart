import 'dart:math' as math;

import 'ocr_raw_text.dart';

/// Structured data extracted from the raw OCR text of an Indonesian receipt.
class ParsedReceipt {
  ParsedReceipt({
    required this.store,
    required this.date,
    required this.items,
    required this.total,
    required this.paymentMethod,
    int? itemsTotal,
    this.discountTotal = 0,
    this.totalFromReceipt = false,
    this.source = 'teks',
    List<String>? rowDump,
    List<String>? warnings,
  })  : itemsTotal = itemsTotal ??
            items.fold<int>(0, (int sum, ParsedItem i) => sum + i.subtotal),
        rowDump = rowDump ?? const <String>[],
        warnings = warnings ?? const <String>[];

  /// Merchant name.
  final String store;

  /// Transaction date, when one could be read.
  final DateTime? date;

  /// Purchased line items.
  final List<ParsedItem> items;

  /// Total printed on the receipt, or the sum of [items] when no total line was
  /// recognised.
  final int total;

  /// Payment method, e.g. `Tunai`, `QRIS`.
  final String paymentMethod;

  /// Sum of the line items as parsed.
  final int itemsTotal;

  /// Sum of receipt-level discounts (always positive).
  final int discountTotal;

  /// True when [total] came from a `TOTAL` line instead of being computed.
  final bool totalFromReceipt;

  /// Which reconstruction produced this result: `geometri` or `teks`.
  final String source;

  /// The reconstructed rows, one per line, for diagnostics in the UI.
  final List<String> rowDump;

  /// Non-fatal problems worth surfacing to the user.
  final List<String> warnings;

  /// `total - itemsTotal`. Zero when the items add up to the printed total;
  /// negative for discounts, positive when the parser missed something.
  int get difference => total - itemsTotal;

  bool get isEmpty => items.isEmpty && total == 0;
}

/// One purchased line item.
class ParsedItem {
  ParsedItem({
    required this.name,
    required this.qty,
    required this.price,
    int? lineTotal,
  }) : lineTotal = lineTotal ?? qty * price;

  final String name;

  /// Quantity printed on the receipt.
  final int qty;

  /// Unit price.
  final int price;

  /// Line total as printed. Kept separate from `qty * price` because receipts
  /// sometimes apply a per-line discount, and because a quantity that does not
  /// divide the total evenly would otherwise lose a rupiah.
  final int lineTotal;

  int get subtotal => lineTotal;
}

class _DateParts {
  const _DateParts({required this.year, required this.month, required this.day});

  final int year;
  final int month;
  final int day;
}

/// One OCR cell with its horizontal extent. Cells on the same visual row are
/// sorted by [left] so column order is preserved.
class _Cell {
  const _Cell({required this.text, required this.left, required this.right});

  final String text;
  final double left;
  final double right;
}

/// A visual row of the receipt.
class _Row {
  _Row(this.cells);

  final List<_Cell> cells;

  String get text =>
      cells.map((_Cell c) => c.text.trim()).where((String t) => t.isNotEmpty).join(' ');
}

/// One plausible reading of an item row's numbers.
class _ItemCandidate {
  const _ItemCandidate({
    required this.qty,
    required this.unit,
    required this.lineTotal,
  });

  final int qty;
  final int unit;
  final int lineTotal;

  int get subtotal => lineTotal;
}

/// A parsed item plus every plausible reading of its numbers, so the parser can
/// pick the one that makes the receipt add up.
class _ItemDraft {
  _ItemDraft(this.name, this.candidates);

  final String name;
  final List<_ItemCandidate> candidates;
  int index = 0;

  _ItemCandidate get selected => candidates[index];
}

/// Parses Indonesian receipt text into items, quantities, prices and totals.
///
/// The pipeline is deliberately staged so each step can be reasoned about and
/// tested in isolation:
///
///   1. **Rows** – rebuild visual rows from ML Kit's per-cell lines using the
///      bounding boxes. Without geometry, fall back to text lines and stitch
///      stacked cells back together.
///   2. **Classify** – label each row as item / total / payment / discount /
///      noise using keyword sets.
///   3. **Interpret** – turn each item row's trailing numbers into
///      `qty`, `unit price` and `line total`, keeping every plausible reading
///      (printers differ in column order).
///   4. **Reconcile** – if the receipt prints a `TOTAL`, choose the readings
///      whose sum matches `total + discounts`. This is what stops a price from
///      being attached to the wrong item.
class ReceiptParser {
  // ---------------------------------------------------------------------------
  // Vocabulary
  // ---------------------------------------------------------------------------

  /// Words that mark a line as a total/summary rather than a purchased item.
  static final RegExp _totalWords = RegExp(
    r'\b(grand\s*total|total\s*bayar|[t7][o0g.]t[a4][l1i]|jumlah|sub\s*total|subtotal|'
    r'tagihan|amount\s*due)\b',
    caseSensitive: false,
  );

  /// `SUB TOTAL` must never win over `TOTAL`.
  static final RegExp _subtotalWords = RegExp(
    r'\b(sub\s*total|subtotal)\b',
    caseSensitive: false,
  );

  static final RegExp _paymentWords = RegExp(
    r'\b(tunai|cash|debit|kredit|qris|gopay|go-?pay|ovo|dana|shopeepay|'
    r'linkaja|bca|mandiri|bri|bni|cimb|permata)\b',
    caseSensitive: false,
  );

  static final RegExp _discountWords = RegExp(
    r'\b(diskon|discount|disc|potongan|voucher|hemat)\b',
    caseSensitive: false,
  );

  /// Savings summaries that restate a discount already counted above, e.g.
  /// `ANDA HEMAT : 1,300`. Counting them would double the discount.
  static final RegExp _savingsSummary = RegExp(
    r'(anda|total)\s*h[ea][mv][a4]?t|andahe[mv][a4]?t|harga\s*[ji1l][uv][a4][l1i]',
    caseSensitive: false,
  );

  /// Lines that indicate pure non-monetary point/coupon counts.
  static final RegExp _pointOrCouponWords = RegExp(
    r'\b(kupon|poin|point)\b',
    caseSensitive: false,
  );

  /// Lines that indicate item/unit count rather than monetary total.
  /// E.g. `TOTAL ITEM : 2`, `TOTAL QTY : 3`, `TOTAL BARANG : 4`.
  static final RegExp _itemCountWords = RegExp(
    r'\b(item|items|qty|kuantitas|pcs|barang|kupon|poin|point)\b',
    caseSensitive: false,
  );

  /// Change/kembalian lines that might follow or contain the word 'total'.
  static final RegExp _changeWords = RegExp(
    r'\b(kembali|kembalian|change|kembaliannya)\b',
    caseSensitive: false,
  );

  /// Lines that carry receipt metadata rather than a purchased product.
  static final RegExp _noiseWords = RegExp(
    r'\b(kembali|kembalian|change|tunai|cash|debit|kredit|qris|gopay|go-?pay|'
    r'ovo|dana|shopeepay|linkaja|ppn|pajak|tax|diskon|discount|disc|potongan|hemat|'
    r'voucher|poin|point|saldo|sisa|bayar|pembayaran|[t7][o0g.]t[a4][l1i]|subtotal|'
    r'sub\s*total|jumlah|tagihan|grand\s*total|terima\s*kasih|terimakasih|'
    r'npwp|harga\s*[ji1l][uv][a4][l1i]|anda\s*h[ea][mv][a4]?t|andahe[mv][a4]?t|dpp|alfacare|kritik|saran|member|ponta|'
    r'selamat|bertambah|klikindomaret|ongkir|cust|care|alfagift|expired|'
    r'kasir|npk|struk|nota|invoice|kwitansi|tanggal|jam|waktu|telp|tel|'
    r'fax|phone|jl\.?|jalan|persero|layanan|konsumen|call|sms|email|www)\b',
    caseSensitive: false,
  );

  /// Bare numbers / phone numbers that should never be a store name.
  static final RegExp _nonStoreLine = RegExp(r'^[\d\s\-+().,:]+$');

  /// Merchant names printed as text somewhere on the receipt. The logo is an
  /// image, so OCR reads it as noise; the name almost always appears in a
  /// footer such as `LAYANAN KONSUMEN INDOMARET`.
  static const List<String> _knownMerchants = <String>[
    'indomaret',
    'alfamart',
    'alfamidi',
    'alfa express',
    'lawson',
    'circle k',
    'familymart',
    'hypermart',
    'transmart',
    'carrefour',
    'superindo',
    'indogrosir',
    'yogya',
    'lotte mart',
    'ranch market',
  ];

  /// Merchant stems that survive OCR errors. Checked in order after the exact
  /// names in [_knownMerchants] fail.
  static const Map<String, String> _merchantStems = <String, String>{
    'indomar': 'Indomaret',
    'alfam': 'Alfamart',
    'alfami': 'Alfamidi',
    'lawson': 'Lawson',
    'circle k': 'Circle K',
    'familymart': 'FamilyMart',
    'superindo': 'Superindo',
    'indogrosir': 'Indogrosir',
    'transmart': 'Transmart',
    'hypermart': 'Hypermart',
  };

  // ---------------------------------------------------------------------------
  // Numbers
  // ---------------------------------------------------------------------------

  /// A printed amount: `4500`, `18,000`, `18.000`, `Rp 4.500`, `(1,300)`.
  static final RegExp _amountShape =
      RegExp(r'^(?:\d{1,3}(?:[.,]\d{3})+|\d{1,7})$');

  /// Returns the integer value of [raw] if it is a printed amount, else `null`.
  ///
  /// Both `,` and `.` are accepted as thousands separators, matching how
  /// Indonesian receipts are printed. A parenthesised or trailing-minus value
  /// is returned negative.
  int? amountValue(String raw) {
    String t = raw.trim();
    if (t.isEmpty) {
      return null;
    }
    // Drop trailing punctuation left behind by OCR, e.g. "7.000,".
    t = t.replaceAll(RegExp(r'[,:;]+$'), '');
    t = t.replaceAll(RegExp(r'^rp\.?', caseSensitive: false), '').trim();

    bool negative = false;
    if (t.startsWith('(') && t.endsWith(')')) {
      negative = true;
      t = t.substring(1, t.length - 1).trim();
    }
    if (t.startsWith('-')) {
      negative = true;
      t = t.substring(1).trim();
    }
    if (t.endsWith('-')) {
      negative = true;
      t = t.substring(0, t.length - 1).trim();
    }

    if (!_amountShape.hasMatch(t)) {
      return null;
    }
    final int? value = int.tryParse(t.replaceAll(RegExp(r'[^0-9]'), ''));
    if (value == null) {
      return null;
    }
    return negative ? -value : value;
  }

  static bool _isMultiplier(String token) {
    final String t = token.toLowerCase();
    return t == 'x' || t == '*' || t == '@';
  }

  static final RegExp _gluedQtyToken =
      RegExp(r'^(\d{1,2}|[1lIL|])[xX*]$');
  static final RegExp _leadingOcrQtyToken =
      RegExp(r'^[LlTtIi|1][xX8*]$');
  static final RegExp _garbledThousandsToken =
      RegExp(r'^[A-Za-z0-9][.,][0OQDGoqd]{3}[,;:]?$');

  /// Extracts a quantity from glued tokens such as `1x`, `2x`, or OCR-misread
  /// `IX` / `lx` (or leading `LX`/`L8`/`T8` when [allowLeadingL] is true),
  /// returning `null` if [token] is not a glued qty multiplier.
  static int? _parseGluedMultiplier(
    String token, {
    bool allowLeadingL = false,
  }) {
    final String t = token.trim();
    if (_gluedQtyToken.hasMatch(t)) {
      final String prefix = t.substring(0, t.length - 1);
      if ('1lI|'.contains(prefix) || (allowLeadingL && prefix == 'L')) {
        return 1;
      }
      if (prefix == 'L') {
        return null;
      }
      return int.tryParse(prefix);
    }
    if (allowLeadingL && _leadingOcrQtyToken.hasMatch(t)) {
      return 1;
    }
    return null;
  }

  // ---------------------------------------------------------------------------
  // Entry point
  // ---------------------------------------------------------------------------

  ParsedReceipt parse(OcrRawResult raw) {
    // Both reconstructions are attempted and scored, because neither is
    // trustworthy on its own: geometry can be wrong if ML Kit returns useless
    // boxes, and text order is wrong for any real receipt. The score rewards a
    // printed total that matches the items, then item count.
    final List<OcrLine> cleanLines = _stripBorderMarginNoise(raw.lines);
    final List<ParsedReceipt> candidates = <ParsedReceipt>[];

    final List<_Row>? geometry = _rowsFromGeometry(cleanLines);
    if (geometry != null) {
      candidates.add(
        _parseRows(_mergeStackedCells(geometry), raw, 'geometri'),
      );
    }

    final List<_Row>? nearest = _rowsFromNearestName(cleanLines);
    if (nearest != null) {
      candidates.add(
        _parseRows(_mergeStackedCells(nearest), raw, 'kolom'),
      );
    }

    candidates.add(
      _parseRows(_mergeStackedCells(_rowsFromText(cleanLines)), raw, 'teks'),
    );

    candidates.sort((ParsedReceipt a, ParsedReceipt b) {
      final int byScore = scoreOf(b).compareTo(scoreOf(a));
      if (byScore != 0) {
        return byScore;
      }
      // Prefer geometry on a tie: it is the only reconstruction that respects
      // the receipt's table layout.
      return b.source == 'geometri' ? 1 : -1;
    });
    return candidates.first;
  }

  /// Ranks a parse result. The printed total is the strongest evidence that the
  /// rows were reconstructed correctly, so a receipt that adds up outranks one
  /// that merely found more lines.
  static int scoreOf(ParsedReceipt p) {
    int score = p.items.length * 10;
    if (p.totalFromReceipt) {
      score += 20;
      final int gap = (p.total + p.discountTotal - p.itemsTotal).abs();
      if (gap <= 1) {
        score += 100;
      } else if (gap <= 100) {
        score += 10;
      }
    }
    if (p.items.isEmpty) {
      score -= 50;
    }
    return score;
  }

  ParsedReceipt _parseRows(List<_Row> rows, OcrRawResult raw, String source) {
    final String store = _extractStore(rows);
    final DateTime? date = _extractDate(raw.fullText);
    final String payment = _extractPayment(rows);
    final int? printedTotal = _extractTotal(rows);
    final int discounts = _extractDiscounts(rows);

    final List<_ItemDraft> drafts = <_ItemDraft>[];
    for (int i = 0; i < rows.length; i++) {
      final _ItemDraft? draft = _readItem(rows[i], isLowerHalf: i >= rows.length ~/ 2);
      if (draft != null) {
        drafts.add(draft);
      }
    }

    // Reconciliation: choose the reading of each ambiguous row that makes the
    // items add up to the printed total (plus receipt-level discounts).
    final int? target = printedTotal == null ? null : printedTotal + discounts;
    _reconcile(drafts, target);

    final List<ParsedItem> items = <ParsedItem>[
      for (final _ItemDraft d in drafts)
        ParsedItem(
          name: d.name,
          qty: d.selected.qty,
          price: d.selected.unit,
          lineTotal: d.selected.lineTotal,
        ),
    ];

    final int itemsTotal =
        items.fold<int>(0, (int sum, ParsedItem i) => sum + i.subtotal);
    final List<String> warnings = <String>[];

    final int total;
    if (printedTotal != null) {
      total = printedTotal;
    } else {
      total = itemsTotal;
      if (items.isNotEmpty) {
        warnings.add('Baris TOTAL tidak terbaca; total dihitung dari item.');
      }
    }

    if (items.isEmpty) {
      warnings.add('Tidak ada item yang terbaca dari struk.');
    } else if (printedTotal != null) {
      final int gap = total + discounts - itemsTotal;
      if (gap.abs() > 1) {
        warnings.add(
          'Total struk dan jumlah item berbeda '
          '${gap.abs()} rupiah. Periksa harga item.',
        );
      }
    }

    final List<String> rowDump = <String>[
      for (final _Row row in rows)
        if (row.text.trim().isNotEmpty)
          '${row.text}   => ${_splitRow(row).numbers}',
    ];

    return ParsedReceipt(
      store: store,
      date: date,
      items: items,
      total: total,
      paymentMethod: payment,
      itemsTotal: itemsTotal,
      discountTotal: discounts,
      totalFromReceipt: printedTotal != null,
      source: source,
      rowDump: rowDump,
      warnings: warnings,
    );
  }

  // ---------------------------------------------------------------------------
  // Stage 1 — rebuild visual rows
  // ---------------------------------------------------------------------------

  /// Groups OCR cells into visual rows.
  ///
  /// ML Kit does **not** return a receipt in reading order: it emits text
  /// blocks, and on a receipt those blocks are the table's columns. The raw
  /// text therefore reads "all product names, then all quantities, then all
  /// prices", which is unusable on its own. The bounding boxes are the only
  /// link between a product and its numbers, so rows are rebuilt by clustering
  /// lines on their vertical centre and sorting each cluster left to right.
  List<_Row>? _rowsFromGeometry(List<OcrLine> lines) {
    if (lines.length < 3) {
      return null;
    }

    // Median line height is the natural unit of vertical tolerance: receipt
    // rows are about one line apart, and boxes on the same row rarely differ by
    // more than half a line.
    final List<double> heights = <double>[
      for (final OcrLine l in lines)
        if ((l.bottom - l.top).abs() > 0) (l.bottom - l.top).abs(),
    ]..sort();
    if (heights.isEmpty) {
      return null;
    }
    final double medianHeight = heights[heights.length ~/ 2];
    final double tolerance = math.max(medianHeight * 0.6, 1.0);

    final List<OcrLine> sorted = List<OcrLine>.of(lines)
      ..sort((OcrLine a, OcrLine b) {
        final int byY = a.centerY.compareTo(b.centerY);
        return byY != 0 ? byY : a.left.compareTo(b.left);
      });

    final List<List<OcrLine>> groups = <List<OcrLine>>[];
    for (final OcrLine line in sorted) {
      if (groups.isNotEmpty) {
        final List<OcrLine> current = groups.last;
        final double meanY = current
                .map((OcrLine l) => l.centerY)
                .reduce((double a, double b) => a + b) /
            current.length;
        if ((line.centerY - meanY).abs() <= tolerance) {
          current.add(line);
          continue;
        }
      }
      groups.add(<OcrLine>[line]);
    }

    return <_Row>[
      for (final List<OcrLine> group in groups)
        _Row(<_Cell>[
          for (final OcrLine line in (List<OcrLine>.of(group)
            ..sort((OcrLine a, OcrLine b) => a.left.compareTo(b.left))))
            _Cell(text: line.text, left: line.left, right: line.right),
        ]),
    ];
  }

  /// Drops repeating vertical border artifacts (e.g., thin dark edges along the
  /// left/right margin misread as `L`/`I`/`O`/`.` across multiple rows).
  List<OcrLine> _stripBorderMarginNoise(List<OcrLine> lines) {
    if (lines.length < 8) return lines;

    double maxRight = 0;
    final List<double> heights = <double>[];
    for (final OcrLine l in lines) {
      if (l.right > maxRight) maxRight = l.right;
      final double h = (l.bottom - l.top).abs();
      if (h > 0) heights.add(h);
    }
    if (maxRight < 100 || heights.isEmpty) return lines;
    heights.sort();
    final double medianH = heights[heights.length ~/ 2];
    final double rowTol = math.max(medianH * 0.6, 1.0);
    final RegExp marginGlyphRe = RegExp(r'^[lLIi|\-_.,:OD]+$');

    bool isCandidate(OcrLine l) {
      final String t = l.text.trim();
      if (t.isEmpty || t.length > 2) return false;
      if (!marginGlyphRe.hasMatch(t)) return false;
      return (l.right - l.left).abs() <= maxRight * 0.035;
    }

    final List<int> leftIdx = <int>[];
    final List<int> rightIdx = <int>[];
    for (int i = 0; i < lines.length; i++) {
      final OcrLine l = lines[i];
      if (!isCandidate(l)) continue;
      final bool nearLeft = l.left <= maxRight * 0.035;
      final bool nearRight = l.right >= maxRight * 0.965;
      if (!nearLeft && !nearRight) continue;

      bool hasPeerRight = false;
      bool hasPeerLeft = false;
      for (int j = 0; j < lines.length; j++) {
        if (i == j) continue;
        final OcrLine o = lines[j];
        if ((o.centerY - l.centerY).abs() > rowTol) continue;
        if (isCandidate(o) &&
            (o.left <= maxRight * 0.035 || o.right >= maxRight * 0.965)) {
          continue;
        }
        if (o.left > l.right) hasPeerRight = true;
        if (o.right < l.left) hasPeerLeft = true;
      }
      if (nearLeft && hasPeerRight) leftIdx.add(i);
      if (nearRight && hasPeerLeft) rightIdx.add(i);
    }

    final Set<int> drop = <int>{};
    if (leftIdx.length >= 3) drop.addAll(leftIdx);
    if (rightIdx.length >= 3) drop.addAll(rightIdx);
    if (drop.isEmpty) return lines;

    return <OcrLine>[
      for (int i = 0; i < lines.length; i++)
        if (!drop.contains(i)) lines[i],
    ];
  }

  /// Pairs each numeric cell with the text line it belongs to using vertical
  /// proximity, without assuming the two columns have the same number of lines.
  ///
  /// This is the safety net for real ML Kit output. The recognizer returns the
  /// receipt as text blocks (one per column) and it merges cells it thinks are
  /// adjacent, so the quantity column may have two lines while the name column
  /// has five. Row clustering then produces ragged rows; anchoring on the text
  /// lines and attaching the nearest numbers on the right is immune to that.
  List<_Row>? _rowsFromNearestName(List<OcrLine> lines) {
    if (lines.length < 4) {
      return null;
    }

    bool isNumericLine(OcrLine line) {
      final _RowParts parts = _splitRow(
        _Row(<_Cell>[
          _Cell(text: line.text, left: line.left, right: line.right),
        ]),
      );
      final String withoutMultipliers =
          line.text.replaceAll(RegExp(r'[xX*@]'), '');
      return parts.numbers.isNotEmpty && !_hasLetters(withoutMultipliers);
    }

    final List<OcrLine> anchors =
        lines.where((OcrLine l) => !isNumericLine(l)).toList();
    final List<OcrLine> numbers =
        lines.where(isNumericLine).toList();
    if (anchors.isEmpty || numbers.isEmpty) {
      return null;
    }

    // Only meaningful when numbers actually sit to the right of the text, i.e.
    // the OCR output really is laid out as columns.
    final bool columnLayout = numbers.any(
      (OcrLine n) => anchors.any((OcrLine a) => n.left > a.left + 1),
    );
    if (!columnLayout) {
      return null;
    }

    // Word-level engines (e.g. ClassicalOcrEngine) emit one OcrLine per word,
    // so non-numeric words on the same horizontal row must be merged into a
    // single row anchor before numeric cells are attached.
    final List<double> heights = <double>[
      for (final OcrLine l in lines)
        if ((l.bottom - l.top).abs() > 0) (l.bottom - l.top).abs(),
    ]..sort();
    final double medianHeight =
        heights.isEmpty ? 16.0 : heights[heights.length ~/ 2];
    final double rowTol = math.max(medianHeight * 0.55, 1.0);

    final List<OcrLine> sortedAnchors = List<OcrLine>.of(anchors)
      ..sort((OcrLine a, OcrLine b) {
        final int byY = a.centerY.compareTo(b.centerY);
        return byY != 0 ? byY : a.left.compareTo(b.left);
      });
    final List<List<OcrLine>> anchorClusters = <List<OcrLine>>[];
    for (final OcrLine a in sortedAnchors) {
      if (anchorClusters.isNotEmpty) {
        final List<OcrLine> cur = anchorClusters.last;
        final double meanY = cur
                .map((OcrLine l) => l.centerY)
                .reduce((double x, double y) => x + y) /
            cur.length;
        if ((a.centerY - meanY).abs() <= rowTol) {
          cur.add(a);
          continue;
        }
      }
      anchorClusters.add(<OcrLine>[a]);
    }
    final List<OcrLine> mergedAnchors = <OcrLine>[
      for (final List<OcrLine> grp in anchorClusters)
        (() {
          grp.sort((OcrLine a, OcrLine b) => a.left.compareTo(b.left));
          return OcrLine(
            text: grp.map((OcrLine g) => g.text).join(' '),
            left: grp.map((OcrLine g) => g.left).reduce(math.min),
            top: grp.map((OcrLine g) => g.top).reduce(math.min),
            right: grp.map((OcrLine g) => g.right).reduce(math.max),
            bottom: grp.map((OcrLine g) => g.bottom).reduce(math.max),
          );
        })(),
    ];

    final Map<OcrLine, List<OcrLine>> attached =
        <OcrLine, List<OcrLine>>{};
    for (final OcrLine number in numbers) {
      OcrLine? best;
      double bestDistance = double.infinity;
      for (final OcrLine anchor in mergedAnchors) {
        if (number.left <= anchor.left + 1) {
          continue;
        }
        final double distance = (number.centerY - anchor.centerY).abs();
        if (distance < bestDistance) {
          bestDistance = distance;
          best = anchor;
        }
      }
      if (best != null) {
        attached.putIfAbsent(best, () => <OcrLine>[]).add(number);
      }
    }

    return <_Row>[
      for (final OcrLine anchor in mergedAnchors)
        _Row(<_Cell>[
          _Cell(text: anchor.text, left: anchor.left, right: anchor.right),
          for (final OcrLine number in (attached[anchor] ?? <OcrLine>[])
            ..sort((OcrLine a, OcrLine b) => a.left.compareTo(b.left)))
            _Cell(text: number.text, left: number.left, right: number.right),
        ]),
    ];
  }

  List<_Row> _rowsFromText(List<OcrLine> lines) {
    return <_Row>[
      for (final OcrLine line in lines)
        _Row(<_Cell>[
          _Cell(text: line.text, left: line.left, right: line.right),
        ]),
    ];
  }

  /// Stitches rows that belong together but were emitted as separate lines.
  ///
  /// Two shapes are handled:
  ///
  /// ```
  /// 18,000          <- line total
  /// 4 4500          <- qty + unit
  /// S/ROTI KRIM...  <- name          (name below the numbers)
  /// ```
  ///
  /// and
  ///
  /// ```
  /// S/ROTI KRIM...  <- name
  /// 4 4500          <- qty + unit
  /// 18,000          <- line total    (numbers below the name)
  /// ```
  List<_Row> _mergeStackedCells(List<_Row> rows) {
    final int n = rows.length;
    final List<bool> isName = List<bool>.filled(n, false);
    final List<bool> isNumbers = List<bool>.filled(n, false);

    for (int i = 0; i < n; i++) {
      final _RowParts parts = _splitRow(rows[i]);
      // `2 x 3.500` or `IX 2.000 2.000` (`1x`) is a numbers-only row even
      // though it contains multiplier letters.
      isNumbers[i] = parts.numbers.isNotEmpty &&
          ((parts.name.isEmpty &&
                  (parts.hasMultiplier || parts.hasGarbledCopy)) ||
              !_hasLetters(rows[i].text.replaceAll(RegExp(r'[xX*@]'), '')));
      isName[i] = parts.numbers.isEmpty &&
          _hasLetters(rows[i].text) &&
          !_noiseWords.hasMatch(rows[i].text) &&
          !_isFuzzyNoiseOrSummary(rows[i].text, isLowerHalf: i >= n ~/ 2) &&
          _looksLikeProductName(rows[i].text);
    }

    // Decide which numeric rows belong to which product name before emitting
    // anything, so a numeric row is never consumed by the wrong item.
    final List<int> owner = List<int>.filled(n, -1);
    for (int i = 0; i < n; i++) {
      if (!isName[i]) {
        continue;
      }

      // Numbers printed above the name belong to it only when the previous
      // item has not already claimed them. This handles both column orders:
      //   numbers / name  (Indomaret)   -> claim above
      //   name / numbers  (most others) -> claim below
      final List<int> before = <int>[];
      for (int j = i - 1; j >= 0 && j >= i - 2; j--) {
        if (!isNumbers[j] || owner[j] != -1) {
          break;
        }
        before.insert(0, j);
      }
      if (before.isNotEmpty) {
        for (final int j in before) {
          owner[j] = i;
        }
        continue;
      }

      for (int j = i + 1; j < n && j <= i + 2; j++) {
        if (!isNumbers[j] || owner[j] != -1) {
          break;
        }
        owner[j] = i;
      }
    }

    final List<_Row> out = <_Row>[];
    for (int i = 0; i < n; i++) {
      if (owner[i] != -1) {
        continue; // claimed by a product name
      }
      if (isName[i]) {
        final List<int> claimed = <int>[
          for (int j = 0; j < n; j++)
            if (owner[j] == i) j,
        ];
        if (claimed.isNotEmpty) {
          // Keep the product name first: `_splitRow` reads the trailing
          // numbers, so numbers printed above the name move after it.
          final StringBuffer text = StringBuffer(rows[i].text);
          for (final int j in claimed) {
            final String numRow = rows[j].text.trim().replaceFirst(
              RegExp(r'^[LlTt][xX8*](?=\s+\S)'),
              '1x',
            );
            text.write(' $numRow');
          }
          out.add(_Row(<_Cell>[
            _Cell(text: text.toString().trim(), left: 0, right: 0),
          ]));
          continue;
        }
      }
      out.add(rows[i]);
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Stage 2 — split a row into name tokens and trailing numbers
  // ---------------------------------------------------------------------------

  /// Repairs amounts that ML Kit split at the thousands separator.
  ///
  /// On real receipts the recognizer frequently emits `18,` + `000`, `8,` +
  /// `500` or `(1,` + `300)` as separate tokens, especially when the column is
  /// narrow. Joining them back is essential: read on their own, `8,` becomes
  /// `8` and `500` becomes `500`, which silently corrupts the line total.
  List<String> _joinSplitAmounts(List<String> tokens) {
    final List<String> out = <String>[];
    final RegExp threeDigits = RegExp(r'^\d{3}[.,;:)\]]*$');
    final RegExp threeZeroLookalikes = RegExp(r'^[0-9OQDGoqd]{3}[.,;:)\]]*$');
    final RegExp startsWithDotDigits = RegExp(r'^[.]\d{3}');

    for (int i = 0; i < tokens.length; i++) {
      final String token = tokens[i];
      // `(` + `1` + `)` -> `(1)` so parenthesised discounts parse cleanly.
      if (token == '(' &&
          i + 2 < tokens.length &&
          tokens[i + 2] == ')' &&
          RegExp(r'^[\d.,]+$').hasMatch(tokens[i + 1])) {
        out.add('(${tokens[i + 1]})');
        i += 2;
        continue;
      }

      final bool endsWithSeparator = RegExp(r'[.,]$').hasMatch(token);
      if (endsWithSeparator && i + 1 < tokens.length) {
        final String next = tokens[i + 1];
        // `18,` + `000`  -> `18,000`;  `(1,` + `300)` -> `(1,300)`
        if (threeDigits.hasMatch(next)) {
          out.add('$token$next');
          i++;
          continue;
        }
        // `18,` + `OOO` -> `18,000` when trailing 000 was misread as O/Q/D.
        if (RegExp(r'^\(?-?\d{1,3}[.,]$').hasMatch(token) &&
            threeZeroLookalikes.hasMatch(next)) {
          final String fixedNext = next
              .replaceAll(RegExp(r'[OQDoqd]'), '0')
              .replaceAll('G', '6');
          out.add('$token$fixedNext');
          i++;
          continue;
        }
        // `8,` + `500` where the next token carries trailing punctuation.
        if (RegExp(r'^\d{3}$').hasMatch(next)) {
          out.add('$token$next');
          i++;
          continue;
        }
      }

      // `18` + `.000` -> `18.000` — OCR splits at the Indonesian thousands
      // separator (dot) when the dot begins a new token.
      if (i + 1 < tokens.length &&
          RegExp(r'^\d{1,3}$').hasMatch(token) &&
          startsWithDotDigits.hasMatch(tokens[i + 1])) {
        out.add('$token${tokens[i + 1]}');
        i++;
        continue;
      }

      // `18` + `.` + `000` -> `18.000` — isolated separator token between digits.
      if (i + 2 < tokens.length &&
          RegExp(r'^\d{1,3}$').hasMatch(token) &&
          (tokens[i + 1] == '.' || tokens[i + 1] == ',') &&
          threeDigits.hasMatch(tokens[i + 2])) {
        out.add('$token.${tokens[i + 2]}');
        i += 2;
        continue;
      }

      out.add(token);
    }
    return out;
  }

  _RowParts _splitRow(_Row row) {
    final List<String> tokens = _joinSplitAmounts(
      row.text
          .split(RegExp(r'\s+'))
          .where((String t) => t.isNotEmpty)
          .toList(),
    );

    // Drop a stray 1-char trailing border/tax flag after a printed amount
    // (e.g. `9.600 L` or `12.200 I`).
    if (tokens.length >= 2 &&
        amountValue(tokens.last) == null &&
        tokens.last.length == 1 &&
        RegExp(r'^[A-Za-z|\-_.,)]$').hasMatch(tokens.last)) {
      final int? prevAmt = amountValue(tokens[tokens.length - 2]);
      if (prevAmt != null && prevAmt >= 100) {
        tokens.removeLast();
      }
    }

    // Scan from the right for the trailing numeric block.
    bool hasMultiplier = false;
    int? gluedQty;
    int cut = tokens.length;
    int? garbledTrailingCopy;
    if (tokens.length >= 2 &&
        amountValue(tokens.last) == null &&
        _garbledThousandsToken.hasMatch(tokens.last)) {
      final int? prevAmt = amountValue(tokens[tokens.length - 2]);
      if (prevAmt != null && prevAmt >= 500) {
        garbledTrailingCopy = prevAmt;
        cut = tokens.length - 1;
      }
    }
    final int numericEnd = cut;
    while (cut > 0) {
      final String token = tokens[cut - 1];
      if (amountValue(token) != null) {
        cut--;
        continue;
      }
      if (_isMultiplier(token) && cut < numericEnd) {
        hasMultiplier = true;
        cut--;
        continue;
      }
      if (cut < numericEnd) {
        final int? nextAmt = amountValue(tokens[cut]);
        final int? q = _parseGluedMultiplier(
          token,
          allowLeadingL: cut == 1 && nextAmt != null && nextAmt >= 500,
        );
        if (q != null && q > 0) {
          hasMultiplier = true;
          gluedQty = q;
          cut--;
          continue;
        }
      }
      break;
    }

    // Positions of the amount tokens inside the numeric block.
    final List<int> amountPositions = <int>[];
    for (int i = cut; i < numericEnd; i++) {
      if (amountValue(tokens[i]) != null) {
        amountPositions.add(i);
      }
    }

    // A lone small integer is far more likely to be part of the product name
    // ("ROKOK MILD 16", "TELUR 1 KG") than a price.
    if (amountPositions.length == 1 &&
        !hasMultiplier &&
        garbledTrailingCopy == null) {
      final String token = tokens[amountPositions.first];
      final int value = amountValue(token)!;
      // Parenthesised/minus values are discounts, never part of a name.
      if (value > 0 &&
          value < 1000 &&
          !token.contains(RegExp(r'[.,()\-]'))) {
        amountPositions.clear();
        cut = tokens.length;
        garbledTrailingCopy = null;
      }
    }

    // Recover corrupted middle unit price between `qty` and divisible `subtotal`
    // (e.g. `PIUK 350N 2 3,MO 7.800` -> qty=2, unit=3900, total=7800).
    List<int>? recoveredMiddle;
    if (amountPositions.length == 1 &&
        !hasMultiplier &&
        garbledTrailingCopy == null &&
        cut >= 2) {
      final int tot = amountValue(tokens[amountPositions.first])!;
      final String garbledMid = tokens[cut - 1];
      final int? q = int.tryParse(tokens[cut - 2]);
      if (q != null &&
          q >= 1 &&
          q <= 99 &&
          RegExp(r'^\d{1,3}[.,][A-Za-z0-9]{2,3}[.,;:)\]]*$')
              .hasMatch(garbledMid) &&
          tot >= 500 &&
          tot % q == 0 &&
          (tot ~/ q) >= 100) {
        recoveredMiddle = <int>[q, tot ~/ q, tot];
        cut -= 2;
      }
    }

    // Only the last three numbers are `qty`, `unit` and `line total`. Anything
    // before them belongs to the product name, e.g. the `225` in
    // "FF LOW FAT VAN 225 1 5200 5,200".
    if (amountPositions.length > 3) {
      cut = amountPositions[amountPositions.length - 3];
    }

    // Three numbers that do not multiply out usually mean the first one is a
    // pack size rather than a quantity: "FF LOW FAT VAN 225 5200 5,200". A real
    // quantity is small (receipts rarely sell 100+ units), so a value above 99
    // in that position is folded back into the name.
    if (amountPositions.length == 3 && !hasMultiplier) {
      final int a = amountValue(tokens[amountPositions[0]])!;
      final int b = amountValue(tokens[amountPositions[1]])!;
      final int c = amountValue(tokens[amountPositions[2]])!;
      final bool multiplies = a * b == c || b * c == a || a * c == b;
      if (!multiplies && a > 99 && a < 1000 && c >= 1000) {
        cut = amountPositions[0] + 1;
      }
    }

    final List<int> numbers = recoveredMiddle ??
        <int>[
          ?gluedQty,
          for (int i = cut; i < numericEnd; i++)
            if (amountValue(tokens[i]) case final int v) v,
          ?garbledTrailingCopy,
        ];
    if (numbers.length > 3) {
      numbers.removeRange(0, numbers.length - 3);
    }

    final List<String> nameTokens = tokens
        .sublist(0, cut)
        .where((String t) => t.trim().isNotEmpty)
        .toList();
    while (nameTokens.isNotEmpty &&
        nameTokens.last.toLowerCase().replaceAll('.', '') == 'rp') {
      nameTokens.removeLast();
    }

    return _RowParts(
      name: _cleanName(nameTokens.join(' ')),
      numbers: numbers,
      hasMultiplier: hasMultiplier,
      hasGarbledCopy: garbledTrailingCopy != null,
    );
  }

  // ---------------------------------------------------------------------------
  // Stage 3 — interpret a row as an item
  // ---------------------------------------------------------------------------

  _ItemDraft? _readItem(_Row row, {bool isLowerHalf = false}) {
    final _RowParts parts = _splitRow(row);
    if (parts.numbers.isEmpty || parts.name.isEmpty) {
      return null;
    }
    if (_noiseWords.hasMatch(parts.name) ||
        _totalWords.hasMatch(parts.name) ||
        _isFuzzyNoiseOrSummary(parts.name, isLowerHalf: isLowerHalf)) {
      return null;
    }
    if (!_looksLikeProductName(parts.name)) {
      return null;
    }

    final List<_ItemCandidate> candidates =
        _candidates(parts.numbers, parts.hasMultiplier);
    if (candidates.isEmpty) {
      return null;
    }
    return _ItemDraft(parts.name, candidates);
  }

  static int _editDist(String a, String b) {
    if (a == b) return 0;
    if ((a.length - b.length).abs() > 2) return 99;
    List<int> prev = List<int>.generate(b.length + 1, (int i) => i);
    List<int> cur = List<int>.filled(b.length + 1, 0);
    for (int i = 1; i <= a.length; i++) {
      cur[0] = i;
      for (int j = 1; j <= b.length; j++) {
        final int cost = a.codeUnitAt(i - 1) == b.codeUnitAt(j - 1) ? 0 : 1;
        cur[j] = math.min(
          math.min(prev[j] + 1, cur[j - 1] + 1),
          prev[j - 1] + cost,
        );
      }
      final List<int> swap = prev;
      prev = cur;
      cur = swap;
    }
    return prev[b.length];
  }

  /// Rejects OCR-corrupted footer/summary labels (`T.TAL LTO`, `TGMAI`,
  /// `KEMBAIIAD`) so they are not recorded as purchased items.
  static bool _isFuzzyNoiseOrSummary(String name, {required bool isLowerHalf}) {
    const List<String> shortLabels = <String>[
      'total',
      'tunai',
      'kembali',
      'diskon',
      'belanja',
    ];
    const List<String> longLabels = <String>[
      'kembalian',
      'subtotal',
      'discount',
      'andahemat',
    ];
    for (final String rawTok in name.toLowerCase().split(RegExp(r'\s+'))) {
      final String norm = rawTok
          .replaceAll('.', 'o')
          .replaceAll('0', 'o')
          .replaceAll('1', 'i')
          .replaceAll('3', 'e')
          .replaceAll('4', 'a')
          .replaceAll('5', 's')
          .replaceAll('6', 'g')
          .replaceAll('7', 't')
          .replaceAll('8', 'b')
          .replaceAll(RegExp(r'[^a-z]'), '');
      if (norm.length < 4) continue;
      for (final String target in shortLabels) {
        if (_editDist(norm, target) <= 1) {
          return true;
        }
      }
      for (final String target in longLabels) {
        final int maxDist = isLowerHalf ? 2 : 1;
        if (_editDist(norm, target) <= maxDist) {
          return true;
        }
      }
      if (isLowerHalf &&
          (RegExp(r'^t[gu][mn][a4][il]$').hasMatch(norm) ||
              RegExp(r'^k[aeo][mn]x?[bp][a4][l1i]{1,2}$').hasMatch(norm))) {
        return true;
      }
    }
    return false;
  }

  /// Every plausible `(qty, unit, lineTotal)` reading of the trailing numbers.
  ///
  /// The first candidate is the most likely one; the rest exist so
  /// [_reconcile] can switch to a different column order when the receipt
  /// total proves it.
  List<_ItemCandidate> _candidates(List<int> numbers, bool hasMultiplier) {
    if (numbers.isEmpty) {
      return const <_ItemCandidate>[];
    }

    // Keep only the last three numbers; anything earlier belongs to the name.
    final List<int> nums = numbers.length > 3
        ? numbers.sublist(numbers.length - 3)
        : numbers;

    if (nums.length >= 3) {
      final int a = nums[nums.length - 3];
      final int b = nums[nums.length - 2];
      final int c = nums[nums.length - 1];
      final List<_ItemCandidate> out = <_ItemCandidate>[];

      // `qty x unit total` — the explicit multiplier settles the order.
      if (hasMultiplier && a > 0 && a <= 999 && b >= 0) {
        out.add(_ItemCandidate(qty: a, unit: b, lineTotal: c));
      }
      // `qty unit total`
      if (a > 0 && a <= 999 && a * b == c) {
        out.add(_ItemCandidate(qty: a, unit: b, lineTotal: c));
      }
      // `total qty unit` (Indomaret prints the line total above the name)
      if (b > 0 && b <= 999 && b * c == a) {
        out.add(_ItemCandidate(qty: b, unit: c, lineTotal: a));
      }
      // `qty total unit`
      if (a > 0 && a <= 999 && a * c == b) {
        out.add(_ItemCandidate(qty: a, unit: c, lineTotal: b));
      }
      // When unit price equals line total (`1x` misread as `18` -> `[18, 10000, 10000]`),
      // recover `qty = 1` (or `a ~/ 10`) before keeping the unmatched `a`.
      if (b > 0 && b == c && a > 1) {
        if (a >= 10 && a <= 99 && (a ~/ 10) * b == c) {
          out.add(_ItemCandidate(qty: a ~/ 10, unit: b, lineTotal: c));
        }
        out.add(_ItemCandidate(qty: 1, unit: b, lineTotal: c));
      }
      // As printed, even if it does not multiply out (line-level discount).
      if (a > 0 && a <= 999 && c > 0) {
        out.add(_ItemCandidate(qty: a, unit: b, lineTotal: c));
      }
      return _dedupe(out);
    }

    if (nums.length == 2) {
      final int a = nums[0];
      final int b = nums[1];
      final List<_ItemCandidate> out = <_ItemCandidate>[];

      // `qty lineTotal` — receipts print the line total, so when it divides
      // evenly the first number is the quantity.
      if (a > 0 && a <= 99 && b % a == 0 && b ~/ a >= 100) {
        out.add(_ItemCandidate(qty: a, unit: b ~/ a, lineTotal: b));
      }
      // `qty unit`
      if (a > 0 && a <= 99) {
        out.add(_ItemCandidate(qty: a, unit: b, lineTotal: a * b));
      }
      // `unit total`
      if (a > 0 && b > 0 && b % a == 0 && b ~/ a <= 99) {
        out.add(_ItemCandidate(qty: b ~/ a, unit: a, lineTotal: b));
      }
      return _dedupe(out);
    }

    // Single amount: one unit at that price.
    if (nums.first <= 0) {
      return const <_ItemCandidate>[];
    }
    return <_ItemCandidate>[
      _ItemCandidate(qty: 1, unit: nums.first, lineTotal: nums.first),
    ];
  }

  List<_ItemCandidate> _dedupe(List<_ItemCandidate> candidates) {
    final List<_ItemCandidate> out = <_ItemCandidate>[];
    final Set<String> seen = <String>{};
    for (final _ItemCandidate c in candidates) {
      if (c.qty <= 0 || c.unit <= 0 || c.lineTotal <= 0) {
        continue;
      }
      final String key = '${c.qty}:${c.unit}:${c.lineTotal}';
      if (seen.add(key)) {
        out.add(c);
      }
    }
    return out;
  }

  /// Picks, for each ambiguous row, the reading that brings the item sum
  /// closest to the receipt's own total. Two passes are enough to settle
  /// interacting choices.
  void _reconcile(List<_ItemDraft> drafts, int? target) {
    if (target == null || drafts.isEmpty) {
      return;
    }
    int sum() => drafts.fold<int>(0, (int s, _ItemDraft d) => s + d.selected.subtotal);

    for (int pass = 0; pass < 2; pass++) {
      for (final _ItemDraft draft in drafts) {
        if (draft.candidates.length < 2) {
          continue;
        }
        final int current = sum();
        int bestIndex = draft.index;
        int bestDiff = (current - target).abs();
        for (int i = 0; i < draft.candidates.length; i++) {
          if (i == draft.index) {
            continue;
          }
          final int candidateSum =
              current - draft.selected.subtotal + draft.candidates[i].subtotal;
          final int diff = (candidateSum - target).abs();
          if (diff < bestDiff) {
            bestDiff = diff;
            bestIndex = i;
          }
        }
        draft.index = bestIndex;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Stage 4 — totals, discounts, payment
  // ---------------------------------------------------------------------------

  /// Finds the printed total.
  ///
  /// Searched from the bottom up, preferring `GRAND TOTAL` and skipping
  /// `SUB TOTAL`, because receipts print subtotals above the final total.
  int? _extractTotal(List<_Row> rows) {
    int? best;
    int bestRank = -1;

    for (int i = rows.length - 1; i >= 0; i--) {
      final _Row row = rows[i];
      final String text = row.text;
      if (!_totalWords.hasMatch(text) ||
          _subtotalWords.hasMatch(text) ||
          _pointOrCouponWords.hasMatch(text) ||
          _changeWords.hasMatch(text)) {
        continue;
      }
      if (_paymentWords.hasMatch(text)) {
        continue;
      }
      // Alfamart prints `TOTAL ITEM  2  12.200` on the same row. Reject pure
      // count rows (`TOTAL ITEM : 2`), but keep `TOTAL ITEM` when it ends with
      // a separate monetary amount (>= 500 or formatted thousands).
      final bool hasItemCountWord = _itemCountWords.hasMatch(text);
      if (hasItemCountWord) {
        final _RowParts parts = _splitRow(row);
        final bool hasMoneyWithCount = parts.numbers.length >= 2 &&
            parts.numbers.last >= 500;
        final bool hasFormattedThousands =
            RegExp(r'\d[.,]\d{3}\b').hasMatch(text) &&
                parts.numbers.isNotEmpty &&
                parts.numbers.last >= 500;
        if (!hasMoneyWithCount && !hasFormattedThousands) {
          continue;
        }
      }
      final int rank = _totalRank(text);
      final int? value = _valueForLabel(rows, i);
      if (value == null || value <= 0) {
        continue;
      }
      // On Indonesian receipts, a monetary total in Rupiah is never under 100
      // (products are almost always >= 500 IDR). Small values are item counts.
      if (value < 100) {
        continue;
      }
      if (rank > bestRank ||
          (rank == bestRank && (best == null || value > best))) {
        best = value;
        bestRank = rank;
      }
    }
    return best;
  }

  int _totalRank(String text) {
    final String t = text.toLowerCase();
    if (t.contains('grand')) {
      return 4;
    }
    if (t.contains('total bayar') ||
        t.contains('total akhir') ||
        t.contains('total belanja') ||
        t.contains('total tagihan') ||
        t.contains('total pembelian') ||
        t.contains('total harga')) {
      return 3;
    }
    if (_itemCountWords.hasMatch(t)) {
      return 1;
    }
    if (t.contains('total')) {
      return 2;
    }
    if (t.contains('jumlah') || t.contains('tagihan')) {
      return 1;
    }
    return 0;
  }

  /// Sum of all discount lines, as a positive number.
  int _extractDiscounts(List<_Row> rows) {
    int total = 0;
    for (int i = 0; i < rows.length; i++) {
      final _Row row = rows[i];
      if (!_discountWords.hasMatch(row.text) ||
          _totalWords.hasMatch(row.text) ||
          _savingsSummary.hasMatch(row.text)) {
        continue;
      }
      final int? value = _valueForLabel(rows, i);
      if (value != null) {
        total += value.abs();
      }
    }
    return total;
  }

  /// The amount belonging to a label row: the right-most number on the row, or
  /// the first number on the next row when OCR split the label from its value.
  int? _valueForLabel(List<_Row> rows, int index) {
    final _RowParts parts = _splitRow(rows[index]);
    if (parts.numbers.isNotEmpty) {
      return parts.numbers.last;
    }
    // "TOTAL :" on one line, "33,900" on the next.
    for (int j = index + 1; j < rows.length && j <= index + 2; j++) {
      final _RowParts next = _splitRow(rows[j]);
      if (next.numbers.isEmpty) {
        if (rows[j].text.trim().isEmpty) {
          continue;
        }
        return null;
      }
      if (_hasLetters(rows[j].text)) {
        return null;
      }
      return next.numbers.first;
    }
    return null;
  }

  String _extractPayment(List<_Row> rows) {
    for (int i = rows.length - 1; i >= 0; i--) {
      final RegExpMatch? m = _paymentWords.firstMatch(rows[i].text);
      if (m == null) {
        continue;
      }
      return _normalizePayment(m.group(1)!);
    }
    return 'Tunai';
  }

  String _normalizePayment(String raw) {
    switch (raw.toLowerCase().replaceAll('-', '')) {
      case 'cash':
        return 'Tunai';
      case 'gopay':
        return 'GoPay';
      case 'ovo':
        return 'OVO';
      case 'dana':
        return 'DANA';
      case 'qris':
        return 'QRIS';
      case 'shopeepay':
        return 'ShopeePay';
      case 'linkaja':
        return 'LinkAja';
      default:
        final String t = raw.trim();
        return t.isEmpty ? 'Tunai' : t[0].toUpperCase() + t.substring(1).toLowerCase();
    }
  }

  // ---------------------------------------------------------------------------
  // Store
  // ---------------------------------------------------------------------------

  String _extractStore(List<_Row> rows) {
    final String haystack =
        rows.map((_Row r) => r.text).join(' ').toLowerCase();
    for (final String merchant in _knownMerchants) {
      if (haystack.contains(merchant)) {
        return _titleCase(merchant);
      }
    }
    // OCR mangles merchant names in footers ("INDOMAREMBALI", "ALFAMART" with
    // a dropped letter). Match on the distinctive stem instead of the full
    // name so the store is still recognised.
    for (final MapEntry<String, String> alias in _merchantStems.entries) {
      if (haystack.contains(alias.key)) {
        return alias.value;
      }
    }
    if (RegExp(r'a[l1i]fa[,.\s]*[mn]?[a4]rt').hasMatch(haystack)) {
      return 'Alfamart';
    }
    if (RegExp(r'a[l1i]fa[mn][i1l]d[i1l]').hasMatch(haystack)) {
      return 'Alfamidi';
    }
    if (RegExp(r'[l1i]nd[o0][mn][a4]r').hasMatch(haystack)) {
      return 'Indomaret';
    }

    for (final _Row row in rows.take(6)) {
      final String t = row.text.trim();
      if (t.length < 3 || t.length > 40) {
        continue;
      }
      if (_nonStoreLine.hasMatch(t) || _datePattern.hasMatch(t)) {
        continue;
      }
      if (_noiseWords.hasMatch(t) || _paymentWords.hasMatch(t)) {
        continue;
      }
      if (_digitRatio(t) > 0.4) {
        continue;
      }
      final String letters = t.replaceAll(RegExp(r'[^A-Za-z]'), '');
      if (letters.length >= 3) {
        return _titleCase(t);
      }
    }
    return 'Struk Tanpa Nama';
  }

  double _digitRatio(String s) {
    if (s.isEmpty) {
      return 0;
    }
    return s.replaceAll(RegExp(r'[^0-9]'), '').length / s.length;
  }

  String _titleCase(String s) {
    return s
        .toLowerCase()
        .split(' ')
        .map((String w) =>
            w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
        .join(' ');
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  bool _hasLetters(String text) => RegExp(r'[A-Za-z]').hasMatch(text);

  /// True when [name] plausibly is a product name rather than metadata.
  ///
  /// The leading character is not required to be a letter: OCR routinely reads
  /// `S/ROTI` as `5/ROTI`. The letter ratio is what rejects register lines such
  /// as `2.0. 31 914115/ALIA MU/01`.
  ///
  /// Thresholds are intentionally lenient (≥ 2 letters, ≥ 30% letter ratio)
  /// because minimarket receipts abbreviate aggressively (`GG FLT 12`,
  /// `FF LW FAT 225`), so the product name often contains more digits and
  /// symbols than letters.
  bool _looksLikeProductName(String name) {
    final String t = name.trim();
    if (t.isEmpty) {
      return false;
    }
    final int letters = t.replaceAll(RegExp(r'[^A-Za-z]'), '').length;
    if (letters < 2) {
      return false;
    }
    if (letters / t.length < 0.3) {
      return false;
    }
    // Reject digit-only-prefixed OCR remnants such as `18 5.OAD` where no
    // word has letters without an embedded price dot/comma.
    final List<String> toks =
        t.split(RegExp(r'\s+')).where((String s) => s.isNotEmpty).toList();
    final bool hasRealWord = toks.any(
      (String tok) =>
          RegExp(r'^[A-Za-z]{2,}').hasMatch(tok) ||
          RegExp(r'^\d+[/-]?[A-Za-z]{2,}$').hasMatch(tok),
    );
    return hasRealWord;
  }

  String _cleanName(String s) {
    return s
        .replaceAll(RegExp(r'[\s.\-:*]+$'), '')
        .replaceAll(RegExp(r'^\s*\d+[.)]\s*'), '')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
  }

  // ---------------------------------------------------------------------------
  // Date
  // ---------------------------------------------------------------------------

  static final RegExp _datePattern = RegExp(
    r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})',
  );

  static final RegExp _timePattern = RegExp(r'(\d{1,2})[:.](\d{2})');

  /// Colon-separated time, preferred over [_timePattern] because a dotted date
  /// (`14.09.16`) also looks like `hh.mm`.
  static final RegExp _timeColon = RegExp(r'(\d{1,2}):(\d{2})');

  /// Matches a plausible date: `dd/mm/yyyy`, `dd-mm-yy`, `dd.mm.yy` or
  /// `yyyy-mm-dd`. Tolerates stray spaces around separators (`13/09/ 2026`)
  /// and uses `(?!\d)` so it does not fire inside an NPWP.
  static final RegExp _dateCandidate = RegExp(
    r'(?<!\d)(\d{1,4})\s*([/\-.])\s*(\d{1,2})\s*\2\s*(\d{1,4})(?!\d)',
  );

  static final RegExp _dateNoise = RegExp(
    r'\b(npwp|telp|tel|phone|hp|no\.?\s*(telp|hp)|fax)\b',
    caseSensitive: false,
  );

  DateTime? _extractDate(String fullText) {
    _DateParts? date;
    String? dateLine;
    for (final String line in fullText.split('\n')) {
      if (_dateNoise.hasMatch(line)) {
        continue;
      }
      date = _parseDateCandidate(line);
      if (date != null) {
        dateLine = line;
        break;
      }
    }
    date ??= _parseDateCandidate(fullText);
    if (date == null) {
      return null;
    }

    int hour = 0;
    int minute = 0;
    final String timeSource = dateLine ?? fullText;
    final RegExpMatch? t =
        _timeColon.firstMatch(timeSource) ?? _timePattern.firstMatch(timeSource);
    if (t != null) {
      hour = int.tryParse(t.group(1)!) ?? 0;
      minute = int.tryParse(t.group(2)!) ?? 0;
      if (hour > 23 || minute > 59) {
        hour = 0;
        minute = 0;
      }
    }
    try {
      return DateTime(date.year, date.month, date.day, hour, minute);
    } catch (_) {
      return null;
    }
  }

  _DateParts? _parseDateCandidate(String text) {
    for (final RegExpMatch m in _dateCandidate.allMatches(text)) {
      final int a = int.parse(m.group(1)!);
      final int c = int.parse(m.group(4)!);
      final String sep = m.group(2)!;

      int day;
      int month;
      int year;

      if (a > 31) {
        if (a < 1900 || a > 2100) {
          continue;
        }
        year = a;
        month = int.parse(m.group(3)!);
        day = c;
      } else {
        if (c < 100) {
          year = 2000 + c;
        } else if (c >= 1900 && c <= 2100) {
          year = c;
        } else {
          continue;
        }
        day = a;
        month = int.parse(m.group(3)!);
      }

      // Dotted dates are common (`14.09.16`), but a dotted number is also how
      // amounts are written, so only accept shapes that cannot be an amount.
      if (sep == '.') {
        final bool isDmy = m.group(1)!.length <= 2 &&
            m.group(3)!.length == 2 &&
            (m.group(4)!.length == 2 || m.group(4)!.length == 4);
        final bool isYmd = m.group(1)!.length == 4 &&
            m.group(3)!.length == 2 &&
            m.group(4)!.length == 2;
        if (!isDmy && !isYmd) {
          continue;
        }
      }

      if (month < 1 || month > 12 || day < 1 || day > 31) {
        continue;
      }
      return _DateParts(year: year, month: month, day: day);
    }
    return null;
  }
}

/// A row split into its product name and its trailing numbers.
class _RowParts {
  const _RowParts({
    required this.name,
    required this.numbers,
    required this.hasMultiplier,
    this.hasGarbledCopy = false,
  });

  final String name;
  final List<int> numbers;
  final bool hasMultiplier;
  final bool hasGarbledCopy;
}
