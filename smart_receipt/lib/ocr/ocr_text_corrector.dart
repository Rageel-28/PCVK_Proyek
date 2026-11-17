// Post-OCR text correction for Indonesian receipt lines.
//
// OCR engines frequently confuse visually similar characters, especially on
// small thermal-receipt fonts. This corrector applies context-aware fixes
// AFTER ML Kit has produced its raw text and BEFORE the receipt parser runs:
//
//   - In **price / amount positions** (right-side numbers): letters that look
//     like digits are replaced by the digit they resemble — `O`→`0`, `l`→`1`,
//     `I`→`1`, `S`→`5`, `B`→`8`, `Z`→`2`, `G`→`6`.
//
//   - In **product-name positions** (left-side text): digits that look like
//     letters are replaced by the letter they resemble — `0`→`O`, `1`→`I`,
//     `5`→`S`, `8`→`B` — but only when the surrounding characters are letters,
//     so actual numbers embedded in product codes (`MILD 16`, `AQUA 600`) are
//     left alone.
//
//   - **Known receipt keywords** (`TOTAL`, `TUNAI`, `SUBTOTAL`, etc.) are
//     corrected via a lookup table regardless of position, because the parser
//     relies on them to find totals and payment methods.
//
//   - **Amount formatting** is normalised: comma-as-decimal is converted to
//     comma-as-thousands-separator so the parser's amount regex matches
//     consistently.
import 'ocr_raw_text.dart';

/// Applies post-OCR character correction to every line in [raw] and returns a
/// new [OcrRawResult] with the corrected text.
///
/// The original geometry (bounding boxes) is preserved unchanged; only the
/// text content is altered.
OcrRawResult correctOcrText(OcrRawResult raw) {
  if (raw.isEmpty) {
    return raw;
  }

  final List<OcrLine> corrected = <OcrLine>[
    for (final OcrLine line in raw.lines)
      OcrLine(
        text: _correctLine(line.text),
        left: line.left,
        top: line.top,
        right: line.right,
        bottom: line.bottom,
      ),
  ];

  final String correctedFull =
      corrected.map((OcrLine l) => l.text).join('\n');

  return OcrRawResult(lines: corrected, fullText: correctedFull);
}

// ---------------------------------------------------------------------------
// Per-line correction
// ---------------------------------------------------------------------------

String _correctLine(String line) {
  String t = line.trim();
  if (t.isEmpty) {
    return line;
  }

  // 1. Fix known receipt keywords first — these are unambiguous regardless of
  //    position and help the parser find totals and payment labels.
  t = _fixKeywords(t);

  // 2. Split the line into a leading text part and a trailing numeric part.
  //    The split must happen AFTER keyword fixes so that `T0TAL` → `TOTAL`
  //    before we decide whether `TOTAL` is a name or a label.
  final _LineParts parts = _splitLine(t);
  final String name = _fixNamePart(parts.name);
  final String numbers = _fixAmountPart(parts.numbers);

  if (numbers.isEmpty) {
    return name.trimRight();
  }
  return '$name  ${numbers.trim()}'.trimRight();
}

// ---------------------------------------------------------------------------
// Keyword correction
// ---------------------------------------------------------------------------

/// Common receipt keywords that OCR mangles.  The keys use a regex so one
/// entry covers several OCR variants (`T0TAL`, `T0TAl`, `T0lAL`, …).
final List<_KeywordFix> _keywordFixes = <_KeywordFix>[
  // Longer patterns first so they are not partially consumed by shorter ones.
  _KeywordFix(RegExp(r'\bGRAND\s*T[O0]TAL\b', caseSensitive: false), 'GRAND TOTAL'),
  _KeywordFix(RegExp(r'\bSUBT[O0]TAL\b', caseSensitive: false), 'SUBTOTAL'),
  _KeywordFix(RegExp(r'\bSUB\s*T[O0]TAL\b', caseSensitive: false), 'SUB TOTAL'),
  _KeywordFix(RegExp(r'\b[T7][O0GC.]T[A4][LI1]\s*BEL[A4]NJ[A4]\b', caseSensitive: false), 'TOTAL BELANJA'),
  _KeywordFix(RegExp(r'\bH[A4]RG[A4]\s*[IJ1][UV][A4][LI1]\b', caseSensitive: false), 'HARGA JUAL'),
  _KeywordFix(RegExp(r'\b[T7][O0GC.]T[A4][LI1]\b', caseSensitive: false), 'TOTAL'),
  _KeywordFix(RegExp(r'\bTUNA[lI1]\b', caseSensitive: false), 'TUNAI'),
  _KeywordFix(RegExp(r'\bKEMBAL[lI1]\b', caseSensitive: false), 'KEMBALI'),
  _KeywordFix(RegExp(r'\bD[lI1]SK[O0]N\b', caseSensitive: false), 'DISKON'),
  _KeywordFix(RegExp(r'\bJUMLAH\b', caseSensitive: false), 'JUMLAH'),
  _KeywordFix(RegExp(r'\bPEMBAYARAN\b', caseSensitive: false), 'PEMBAYARAN'),
  _KeywordFix(RegExp(r'\bBELANJA\b', caseSensitive: false), 'BELANJA'),
  _KeywordFix(RegExp(r'\bSTRUK\b', caseSensitive: false), 'STRUK'),
  _KeywordFix(RegExp(r'\bKAS[lI1]R\b', caseSensitive: false), 'KASIR'),
  _KeywordFix(RegExp(r'\bTANGGAL\b', caseSensitive: false), 'TANGGAL'),
  _KeywordFix(RegExp(r'\bQR[lI1]S\b', caseSensitive: false), 'QRIS'),
  _KeywordFix(RegExp(r'\bINV[O0][lI1]CE\b', caseSensitive: false), 'INVOICE'),
  _KeywordFix(RegExp(r'\bN[O0]TA\b', caseSensitive: false), 'NOTA'),
  // Currency prefix normalisation.
  _KeywordFix(RegExp(r'\bRp\.?\s*', caseSensitive: false), 'Rp '),
];

String _fixKeywords(String text) {
  String t = text;
  for (final _KeywordFix fix in _keywordFixes) {
    t = t.replaceAll(fix.pattern, fix.replacement);
  }
  return t;
}

class _KeywordFix {
  const _KeywordFix(this.pattern, this.replacement);
  final Pattern pattern;
  final String replacement;
}

// ---------------------------------------------------------------------------
// Line splitting
// ---------------------------------------------------------------------------

/// Splits a receipt line into its left (name/label) part and its right
/// (numeric) part.
///
/// Strategy: scan tokens from the RIGHT. While the rightmost token looks like
/// a printed amount or a multiplier (`x`, `@`), it belongs to the numeric
/// block. Everything to the left is the name.
_LineParts _splitLine(String line) {
  final List<String> tokens = line.split(RegExp(r'\s+'));
  if (tokens.isEmpty) {
    return _LineParts(name: line, numbers: '');
  }

  int cut = tokens.length;
  while (cut > 0) {
    final String tok = tokens[cut - 1];
    if (_looksLikeAmount(tok) || _isMultiplier(tok)) {
      cut--;
    } else {
      break;
    }
  }

  // If EVERYTHING looks like numbers, treat the whole thing as numbers.
  if (cut == 0) {
    return _LineParts(name: '', numbers: line);
  }

  final String name = tokens.sublist(0, cut).join(' ');
  final String numbers =
      cut < tokens.length ? tokens.sublist(cut).join(' ') : '';
  return _LineParts(name: name, numbers: numbers);
}

/// Returns true when [token] is a printed amount — possibly containing OCR
/// letter-substitutions like `O` for `0` or `l` for `1`.
///
/// Accepts:
///   - Pure digit strings: `3500`, `15000`
///   - Thousands-separated: `3.500`, `15,000`, `3.S00`, `1O.OOO`
///   - With `Rp` prefix: `Rp3.500`, `Rp 10.000`
///   - With parentheses: `(1.300)`
///   - Tokens where the majority of non-separator characters are digits or
///     common OCR letter-substitutes for digits (O, l, I, S, B, Z, G).
bool _looksLikeAmount(String token) {
  String t = token
      .replaceAll(RegExp(r'^[Rr][Pp]\.?\s*'), '')
      .replaceAll(RegExp(r'^[(]'), '')
      .replaceAll(RegExp(r'[),;:]+$'), '');

  if (t.isEmpty) {
    return false;
  }

  // Replace known OCR digit-substitutes before checking pattern.
  final String normalized = t
      .replaceAll('O', '0')
      .replaceAll('o', '0')
      .replaceAll('D', '0')
      .replaceAll('d', '0')
      .replaceAll('Q', '0')
      .replaceAll('q', '0')
      .replaceAll('l', '1')
      .replaceAll('I', '1')
      .replaceAll('S', '5')
      .replaceAll('B', '8')
      .replaceAll('Z', '2')
      .replaceAll('G', '6');

  return RegExp(r'^\d[\d.,]*$').hasMatch(normalized) && normalized.isNotEmpty;
}

bool _isMultiplier(String token) {
  return token == 'x' || token == 'X' || token == '*' || token == '@';
}

class _LineParts {
  const _LineParts({required this.name, required this.numbers});
  final String name;
  final String numbers;
}

// ---------------------------------------------------------------------------
// Name-part correction  (digits → letters when surrounded by letters)
// ---------------------------------------------------------------------------

/// In a product name, a digit that sits BETWEEN two letters is very likely an
/// OCR error (e.g. `SAB0N` → `SABON`, `1NDOMIE` → `INDOMIE`).
///
/// The rule: substitute only when BOTH neighbours are letters, OR the digit is
/// at word boundary (start/end of the whole string) and the adjacent character
/// is a letter. This keeps `600ML` intact (`6` and `M` flank the `0` but `6`
/// is a digit so no substitution on that position).
String _fixNamePart(String name) {
  if (name.isEmpty) {
    return name;
  }
  final String trimmed = name.trim();
  // Short stacked-quantity tokens like 'L8' or 'T8' (misread '1x') must not be
  // rewritten into letters ('LB'/'TB') before the receipt parser handles them.
  if (trimmed.length < 4 && RegExp(r'^[1lILT|]?[0-9]+$', caseSensitive: false).hasMatch(trimmed)) {
    return name;
  }
  final StringBuffer buf = StringBuffer();
  for (int i = 0; i < name.length; i++) {
    final String ch = name[i];
    final String? fix = _digitToLetter[ch];
    if (fix != null) {
      final bool prevLetter =
          i > 0 && RegExp(r'[A-Za-z]').hasMatch(name[i - 1]);
      final bool nextLetter =
          i < name.length - 1 && RegExp(r'[A-Za-z]').hasMatch(name[i + 1]);
      final bool atStart = i == 0;
      final bool atEnd = i == name.length - 1;
      // Require both sides to be letters unless at the very start or end of
      // the string (where one side cannot exist).
      final bool shouldFix = (prevLetter && nextLetter) ||
          (atStart && nextLetter) ||
          (atEnd && prevLetter);
      if (shouldFix) {
        buf.write(fix);
        continue;
      }
    }
    buf.write(ch);
  }
  return buf.toString();
}

/// Digits that are visually similar to letters and frequently appear in
/// product names due to OCR misreads.
const Map<String, String> _digitToLetter = <String, String>{
  '0': 'O',
  '1': 'I',
  '5': 'S',
  '8': 'B',
  '6': 'G',
  '2': 'Z',
};

// ---------------------------------------------------------------------------
// Amount-part correction  (letters → digits in numeric fields)
// ---------------------------------------------------------------------------

/// In an amount token, any letter that looks like a digit is replaced by that
/// digit.  For example `1O.OOO` → `10.000`, `l5.000` → `15.000`.
String _fixAmountPart(String numbers) {
  if (numbers.isEmpty) {
    return numbers;
  }
  final List<String> tokens = numbers.trim().split(RegExp(r'\s+'));
  final List<String> fixed = <String>[];
  for (final String token in tokens) {
    if (token.isEmpty) continue;
    if (_isMultiplier(token)) {
      fixed.add(token);
      continue;
    }
    fixed.add(_fixAmountToken(token));
  }
  return fixed.isEmpty ? '' : fixed.join(' ');
}

String _fixAmountToken(String token) {
  // Strip leading Rp prefix and surrounding punctuation before checking.
  final bool hasRp = RegExp(r'^[Rr][Pp]\.?\s*').hasMatch(token);
  String inner = token
      .replaceAll(RegExp(r'^[Rr][Pp]\.?\s*'), '')
      .replaceAll(RegExp(r'^[(]'), '')
      .replaceAll(RegExp(r'[)]+$'), '');

  // If the inner part has no digit at all, it is not an amount — skip.
  if (!RegExp(r'\d').hasMatch(inner)) {
    return token;
  }

  final StringBuffer buf = StringBuffer();
  for (int i = 0; i < inner.length; i++) {
    final String ch = inner[i];
    final String? fix = _letterToDigit[ch];
    if (fix != null) {
      buf.write(fix);
    } else {
      buf.write(ch);
    }
  }

  String result = buf.toString();
  // Normalise comma-decimal to dot-thousands so the parser sees `10.000`
  // not `10,000` (both are used in Indonesian receipts but the parser
  // handles both; unifying here keeps the corrector output consistent).
  result = _normalizeThousandsSep(result);

  if (hasRp) {
    return 'Rp $result';
  }
  return result;
}

/// Letters visually similar to digits, common in OCR output from amount fields.
const Map<String, String> _letterToDigit = <String, String>{
  'O': '0',
  'o': '0',
  'D': '0',
  'd': '0',
  'Q': '0',
  'q': '0',
  'l': '1',
  'I': '1',
  'i': '1', // lowercase i in some fonts
  'S': '5',
  's': '5',
  'B': '8',
  'Z': '2',
  'z': '2',
  'G': '6',
  'g': '6',
  'T': '1', // rare but seen: capital T misread as 1 on faded receipts
};

/// Converts a comma-separated decimal (`10,000`) to a dot-separated one
/// (`10.000`) when the pattern clearly represents an Indonesian thousands
/// amount, not a decimal fraction.
///
/// Rule: if the string ends with `,NNN` or `,NN` where NNN ≥ 100, it is a
/// thousands separator, not a decimal point — replace the comma with a dot.
String _normalizeThousandsSep(String amount) {
  // `10,000` → `10.000`;  `10,500` → `10.500`
  // But `10,50` is ambiguous — leave it alone (the parser handles both).
  return amount.replaceAllMapped(
    RegExp(r'(\d),(\d{3})'),
    (Match m) => '${m.group(1)}.${m.group(2)}',
  );
}
