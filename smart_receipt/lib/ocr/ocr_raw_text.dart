/// Raw OCR output: the recognised lines with their on-image geometry plus the
/// concatenated text. Kept free of any plugin types so it can be parsed and
/// unit-tested without a device.
class OcrRawResult {
  const OcrRawResult({required this.lines, required this.fullText});

  const OcrRawResult.empty()
      : lines = const <OcrLine>[],
        fullText = '';

  final List<OcrLine> lines;
  final String fullText;

  bool get isEmpty => lines.isEmpty || fullText.trim().isEmpty;

  /// Average per-line recognition confidence is not exposed by the engine, so we
  /// approximate text quality from the number of usable lines.
  int get lineCount => lines.length;
}

/// One recognised line of text and its bounding box in image pixels.
class OcrLine {
  const OcrLine({
    required this.text,
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  final String text;
  final double left;
  final double top;
  final double right;
  final double bottom;

  double get centerY => (top + bottom) / 2;

  @override
  String toString() => 'OcrLine("$text")';
}
