import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../ocr_engine.dart';
import '../ocr_raw_text.dart';
import 'binarization.dart';
import 'char_classifier.dart';
import 'char_normalizer.dart';
import 'classical_config.dart';
import 'connected_components.dart';
import 'deskew.dart';
import 'gray_image.dart';
import 'hog.dart';
import 'lexicon_corrector.dart';
import 'paper_detector.dart';
import 'segmenter.dart';

/// Satu karakter hasil prediksi beserta posisinya.
class PredictedChar {
  const PredictedChar({
    required this.char,
    required this.confidence,
    required this.margin,
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  final String char;
  final double confidence;
  final double margin;
  final int left;
  final int top;
  final int right;
  final int bottom;

  int get width => right - left + 1;
}

/// Satu baris hasil OCR.
class PredictedLine {
  const PredictedLine({
    required this.text,
    required this.confidence,
    required this.top,
    required this.bottom,
    required this.chars,
  });

  final String text;
  final double confidence;
  final int top;
  final int bottom;
  final List<PredictedChar> chars;
}

/// Hasil lengkap satu pemindaian.
class ClassicalOcrResult {
  const ClassicalOcrResult({
    required this.text,
    required this.rawText,
    required this.lines,
    required this.meanConfidence,
    required this.lowConfidenceChars,
    required this.skewAngle,
    required this.lexiconFixes,
    required this.upscaleFactor,
  });

  final String text;
  final String rawText;
  final List<PredictedLine> lines;
  final double meanConfidence;
  final int lowConfidenceChars;
  final double skewAngle;
  final int lexiconFixes;
  final double upscaleFactor;

  Map<String, Object?> asJson() => <String, Object?>{
        'text': text,
        'raw_text': rawText,
        'mean_confidence': double.parse(meanConfidence.toStringAsFixed(4)),
        'low_confidence_chars': lowConfidenceChars,
        'skew_angle': double.parse(skewAngle.toStringAsFixed(2)),
        'lexicon_fixes': lexiconFixes,
        'upscale_factor': upscaleFactor,
        'n_lines': lines.length,
      };
}

/// Pipeline OCR klasik lengkap: gambar struk -> teks.
///
/// Alur:
///   deteksi kertas -> grayscale -> denoise -> binarisasi -> upscale
///   -> deskew -> segmentasi baris & karakter -> normalisasi 32x32
///   -> HOG -> Linear SVM -> pengelompokan kata -> koreksi leksikon
///
/// Tidak memakai library OCR siap pakai dan tidak memakai deep learning.
class ClassicalOcrEngine implements OcrEngine {
  ClassicalOcrEngine({
    this.method = BinarizationMethod.sauvola,
    this.useDeskew = true,
    this.useLexicon = false,
    this.usePaperCrop = true,
    this.useContext = true,
    CharClassifier? classifier,
    LexiconCorrector? lexicon,
  })  : _classifier = classifier ?? CharClassifier(),
        _lexicon = lexicon ?? LexiconCorrector();

  /// Metode binarisasi yang dipakai.
  final BinarizationMethod method;

  /// Koreksi kemiringan aktif.
  final bool useDeskew;

  /// Koreksi leksikon aktif.
  ///
  /// Default **false**. Koreksi kamus dirancang untuk data sintetis yang
  /// kosakatanya berasal dari katalog yang sama, tetapi pada 28 baris struk
  /// asli ia menurunkan exact match dari 67,9% ke 60,7%: nama produk yang
  /// benar seperti `GAM` dan `BACTERIAL` justru "dikoreksi" menjadi `GA`
  /// dan `BATERAI`. Aktifkan hanya bila kamusnya memang mencakup kosakata
  /// struk yang sedang dibaca.
  final bool useLexicon;

  /// Deteksi dan potong area kertas aktif.
  final bool usePaperCrop;

  /// Batasi kandidat karakter sesuai konteks kata (angka vs huruf).
  final bool useContext;

  final CharClassifier _classifier;
  final LexiconCorrector _lexicon;

  bool _ready = false;

  /// Muat model karakter dan leksikon. Aman dipanggil berulang.
  Future<void> load() async {
    if (_ready) return;
    await _classifier.load();
    if (useLexicon) {
      try {
        await _lexicon.load();
      } catch (_) {
        // Leksikon opsional: tanpa kamus OCR tetap berjalan.
      }
    }
    _ready = true;
  }

  bool get isReady => _ready;

  // -- Tahap citra --------------------------------------------------------

  /// Binarisasi awal tanpa upscale/deskew.
  ///
  /// Mengembalikan grayscale yang **sudah didenoise** karena patch karakter
  /// dipotong dari citra inilah, bukan dari citra mentah. Versi Python
  /// berperilaku sama; bila berbeda, fitur HOG akan menyimpang.
  ({GrayImage gray, GrayImage ink}) _binarizeOnce(GrayImage input) {
    final GrayImage gray = medianBlur3(input);
    final GrayImage ink = removeSmallComponents(
      binarize(gray, method),
      kCharMinInk,
    );
    return (gray: gray, ink: ink);
  }

  /// Faktor pembesaran agar tinggi baris mendekati target.
  ///
  /// Teks struk kadang hanya 10-15 px tingginya. Stroke setipis 1 px
  /// mudah terputus saat binarisasi sehingga satu huruf terpecah menjadi
  /// beberapa komponen. Pembesaran menaikkan jumlah baris yang
  /// tersegmentasi dengan benar secara signifikan.
  double _adaptiveScale(GrayImage ink) {
    final List<({int top, int bottom})> lines = segmentLines(ink);
    if (lines.isEmpty) return 1.0;

    final List<int> heights = <int>[
      for (final ({int top, int bottom}) l in lines) l.bottom - l.top + 1,
    ]..sort();
    final double median = heights[heights.length ~/ 2].toDouble();
    if (median <= 0) return 1.0;

    double factor = kTargetLineHeight / median;
    factor = factor.clamp(1.0, kMaxUpscale);
    // Bulatkan ke 0.5 supaya deterministik dan hemat.
    return (factor * 2).round() / 2.0;
  }

  /// Siapkan grayscale dan mask tinta siap segmentasi.
  ({GrayImage gray, GrayImage ink, double angle, double scale}) prepare(
    GrayImage source,
  ) {
    GrayImage gray = source;

    // Foto struk sering berlatar gelap; potong dulu area kertasnya.
    if (usePaperCrop) {
      gray = cropPaper(gray).image;
    }

    var first = _binarizeOnce(gray);
    double factor = _adaptiveScale(first.ink);

    if (factor > 1.0) {
      // Pembulatan mengikuti Python/cv2 (setengah ke genap) agar ukuran
      // hasil resize identik dengan versi pelatihan.
      final int nw = math.max(1, pythonRound(gray.width * factor));
      final int nh = math.max(1, pythonRound(gray.height * factor));
      // Yang diperbesar adalah grayscale yang SUDAH didenoise, sama
      // seperti versi Python. Memperbesar citra mentah membuat noise
      // ikut membesar dan binarisasi menghasilkan bintik palsu.
      gray = first.gray.resizedBicubic(nw, nh);
      first = _binarizeOnce(gray);
      // Patch karakter dipotong dari grayscale yang sudah didenoise ulang.
      // Tanpa ini, Dart memotong dari citra yang masih ber-noise sementara
      // Python memotong dari citra bersih, dan fitur HOG menyimpang.
      gray = first.gray;
    } else {
      factor = 1.0;
    }

    double angle = 0.0;
    GrayImage ink = first.ink;
    if (useDeskew) {
      final ({GrayImage gray, GrayImage ink, double angle}) deskewed =
          autoDeskewPair(gray, ink);
      gray = deskewed.gray;
      ink = deskewed.ink;
      angle = deskewed.angle;
    }

    return (gray: gray, ink: ink, angle: angle, scale: factor);
  }

  // -- Tahap pengenalan ---------------------------------------------------

  /// Baca satu baris: segmentasi karakter -> HOG -> SVM.
  PredictedLine _readLine(
    GrayImage gray,
    GrayImage ink,
    ({int top, int bottom}) lineBox,
  ) {
    final List<({int left, int top, int right, int bottom})> boxes =
        segmentCharacters(ink, lineBox);

    if (boxes.isEmpty) {
      return PredictedLine(
        text: '',
        confidence: 0,
        top: lineBox.top,
        bottom: lineBox.bottom,
        chars: const <PredictedChar>[],
      );
    }

    final List<PredictedChar> chars = <PredictedChar>[];
    final List<Float32List> features = <Float32List>[];
    for (final ({int left, int top, int right, int bottom}) box in boxes) {
      final GrayImage patch = normalizeChar(gray, box);
      final Float32List vec = hog(patch);
      final CharPrediction pred = _classifier.predictFromFeatures(vec);
      features.add(vec);
      chars.add(
        PredictedChar(
          char: pred.char,
          confidence: pred.confidence,
          margin: pred.margin,
          left: box.left,
          top: box.top,
          right: box.right,
          bottom: box.bottom,
        ),
      );
    }

    final List<PredictedChar> finalChars =
        useContext ? _applyContext(chars, features, lineBox) : chars;

    final double confidence = finalChars.isEmpty
        ? 0.0
        : finalChars.map((PredictedChar c) => c.confidence).reduce((double a, double b) => a + b) /
            finalChars.length;

    return PredictedLine(
      text: _joinWithSpaces(finalChars),
      confidence: confidence,
      top: lineBox.top,
      bottom: lineBox.bottom,
      chars: finalChars,
    );
  }

  /// Prediksi ulang tiap kata dengan kandidat yang sesuai konteks.
  ///
  /// Struk thermal sangat sering menukar angka dengan huruf yang bentuknya
  /// mirip: `0` dengan `O`/`D`, `1` dengan `I`/`L`, `5` dengan `S`, `8`
  /// dengan `B`, `2` dengan `Z`, dan koma dengan titik. Model karakter tidak
  /// bisa memutuskan sendiri karena bentuknya memang nyaris sama pada kanvas
  /// 32x32.
  ///
  /// Konteks yang dipakai:
  ///
  /// - **Ukuran glyph.** Tanda baca kecil hanya menempati sebagian kecil
  ///   tinggi baris. Pada 3-4 piksel, HOG praktis tidak informatif dan
  ///   prediksi bisa berubah hanya karena satu baris piksel berbeda.
  ///   Glyph seperti ini dibatasi ke `.,-` sehingga hasilnya deterministik.
  /// - **Isi kata.** Kata yang mayoritasnya angka dibatasi ke angka dan
  ///   pemisah; kata yang mayoritasnya huruf dibatasi ke huruf.
  List<PredictedChar> _applyContext(
    List<PredictedChar> chars,
    List<Float32List> features,
    ({int top, int bottom}) lineBox,
  ) {
    if (chars.isEmpty) return chars;
    final List<PredictedChar> out = List<PredictedChar>.of(chars);

    final int lineH = lineBox.bottom - lineBox.top + 1;
    final List<int> widths = <int>[
      for (final PredictedChar c in chars) c.right - c.left + 1,
    ]..sort();
    final double medianW = widths.length.isOdd
        ? widths[widths.length ~/ 2].toDouble()
        : (widths[widths.length ~/ 2 - 1] + widths[widths.length ~/ 2]) / 2.0;

    // 1. Glyph sangat kecil -> tanda baca.
    if (lineH > 0) {
      for (int i = 0; i < out.length; i++) {
        final PredictedChar c = out[i];
        final int h = c.bottom - c.top + 1;
        final int w = c.right - c.left + 1;
        if (h < lineH * kTinyGlyphHeightRatio && w < medianW * kTinyGlyphWidthRatio) {
          final CharPrediction pred =
              _classifier.predictRestricted(features[i], kCharsetPunct);
          out[i] = PredictedChar(
            char: pred.char,
            confidence: pred.confidence,
            margin: pred.margin,
            left: c.left,
            top: c.top,
            right: c.right,
            bottom: c.bottom,
          );
        }
      }
    }

    // 2. Konteks kata.
    for (final List<int> group in _wordGroups(out)) {
      if (group.length < 2) continue;
      final List<String> chars = <String>[for (final int i in group) out[i].char];
      final String? allowed = charsetForToken(chars);
      if (allowed == null) continue;

      for (final int i in group) {
        final CharPrediction pred =
            _classifier.predictRestricted(features[i], allowed);
        out[i] = PredictedChar(
          char: pred.char,
          confidence: pred.confidence,
          margin: pred.margin,
          left: out[i].left,
          top: out[i].top,
          right: out[i].right,
          bottom: out[i].bottom,
        );
      }
    }
    return out;
  }

  /// Kandidat yang boleh dipakai untuk sebuah kata, atau `null`.
  ///
  /// Aturannya **murni**: kata hanya dibatasi bila seluruh isinya memang
  /// satu jenis. Ini penting karena struk penuh token campuran seperti
  /// `600ML`, `250ML`, `140ML`, dan `403/5`. Versi pertama aturan ini
  /// memakai mayoritas (>60%), dan akibatnya `600ML` dianggap kata angka
  /// sehingga `M` dan `L` dibatasi menjadi digit.
  ///
  /// Harus sama dengan `charset_for_token` di `ML/ocr_classical/config.py`.
  static String? charsetForToken(List<String> chars) {
    if (chars.isEmpty) return null;
    bool allNumeric = true;
    bool allAlpha = true;
    for (final String c in chars) {
      if (c.isEmpty) return null;
      if (!kNumericTokenChars.contains(c)) allNumeric = false;
      if (!kAlphaTokenChars.contains(c)) allAlpha = false;
    }
    if (allNumeric) return kCharsetNumeric;
    if (allAlpha) return kCharsetAlpha;
    return null;
  }

  /// Ambang celah (piksel) yang dianggap batas kata.
  ///
  /// Versi lama memakai kelipatan median celah. Itu gagal pada struk karena
  /// tanda baca punya celah lebar di kedua sisinya: pada `121,500` celah di
  /// sekitar koma lebih besar daripada median celah, sehingga spasi
  /// disisipkan di tengah angka. Lebar karakter jauh lebih stabil.
  static double _spaceThreshold(List<PredictedChar> chars) {
    if (chars.isEmpty) return 2.0;
    final List<int> widths = <int>[
      for (final PredictedChar c in chars) c.right - c.left + 1,
    ]..sort();
    final double medianW = widths.length.isOdd
        ? widths[widths.length ~/ 2].toDouble()
        : (widths[widths.length ~/ 2 - 1] + widths[widths.length ~/ 2]) / 2.0;
    return math.max(2.0, medianW * kSpaceWidthRatio);
  }

  /// True bila dua karakter tidak boleh dipisah spasi.
  ///
  /// Pemisah ribuan pada struk selalu menempel: `113,100` tidak pernah
  /// ditulis `113, 100`.
  static bool _gluedBySeparator(String left, String right) {
    const String separators = ',.';
    bool isDigit(String s) =>
        s.isNotEmpty && '0123456789'.contains(s);
    if (separators.contains(left) && (isDigit(right) || separators.contains(right))) {
      return true;
    }
    if (separators.contains(right) && isDigit(left)) {
      return true;
    }
    return false;
  }

  /// Kelompokkan indeks karakter menjadi kata memakai ambang yang sama
  /// dengan [_joinWithSpaces].
  static List<List<int>> _wordGroups(List<PredictedChar> chars) {
    if (chars.isEmpty) return const <List<int>>[];
    if (chars.length == 1) return <List<int>>[<int>[0]];

    final double threshold = _spaceThreshold(chars);
    final List<List<int>> groups = <List<int>>[
      <int>[0],
    ];
    for (int i = 1; i < chars.length; i++) {
      final int gap = chars[i].left - chars[i - 1].right;
      final bool glued = _gluedBySeparator(chars[i - 1].char, chars[i].char);
      if (gap > threshold && !glued) {
        groups.add(<int>[i]);
      } else {
        groups.last.add(i);
      }
    }
    return groups;
  }

  /// Sisipkan spasi berdasarkan lebar karakter pada baris itu.
  ///
  /// Celah di dalam kata dan celah antar kata saling tumpang tindih bila
  /// diukur dengan median celah, sehingga ambangnya memakai lebar karakter
  /// (lihat [_spaceThreshold]).
  static String _joinWithSpaces(List<PredictedChar> chars) {
    if (chars.isEmpty) return '';
    if (chars.length == 1) return chars.first.char;

    final double threshold = _spaceThreshold(chars);

    final StringBuffer sb = StringBuffer()..write(chars.first.char);
    for (int i = 1; i < chars.length; i++) {
      final int gap = chars[i].left - chars[i - 1].right;
      if (gap > threshold && !_gluedBySeparator(chars[i - 1].char, chars[i].char)) {
        sb.write(' ');
      }
      sb.write(chars[i].char);
    }
    return sb.toString();
  }

  // -- Publik -------------------------------------------------------------

  /// Jalankan pipeline pada citra grayscale yang sudah didekode.
  ClassicalOcrResult run(
    GrayImage source, {
    bool? applyLexicon,
  }) {
    if (!_classifier.isLoaded) {
      throw OcrEngineException('Model belum dimuat. Panggil load() dulu.');
    }

    final ({GrayImage gray, GrayImage ink, double angle, double scale}) prep =
        prepare(source);

    final List<({int top, int bottom})> lineBoxes = segmentLines(prep.ink);

    final List<PredictedLine> lines = <PredictedLine>[];
    for (final ({int top, int bottom}) box in lineBoxes) {
      final PredictedLine line = _readLine(prep.gray, prep.ink, box);
      if (line.text.trim().isNotEmpty) {
        lines.add(line);
      }
    }

    final String rawText = lines.map((PredictedLine l) => l.text).join('\n');

    final bool doLexicon = applyLexicon ?? useLexicon;
    String text = rawText;
    int fixes = 0;
    if (doLexicon && _lexicon.isLoaded) {
      final List<String> corrected = <String>[];
      for (final String line in text.split('\n')) {
        final ({String text, int fixes}) r = _lexicon.correctText(line);
        fixes += r.fixes;
        corrected.add(r.text);
      }
      text = corrected.join('\n');
    }

    final List<PredictedChar> allChars = <PredictedChar>[
      for (final PredictedLine l in lines) ...l.chars,
    ];
    final double meanConfidence = allChars.isEmpty
        ? 0.0
        : allChars
                .map((PredictedChar c) => c.confidence)
                .reduce((double a, double b) => a + b) /
            allChars.length;
    final int lowConfidence = allChars
        .where((PredictedChar c) => c.confidence < _classifier.threshold)
        .length;

    return ClassicalOcrResult(
      text: text,
      rawText: rawText,
      lines: lines,
      meanConfidence: meanConfidence,
      lowConfidenceChars: lowConfidence,
      skewAngle: prep.angle,
      lexiconFixes: fixes,
      upscaleFactor: prep.scale,
    );
  }

  /// Baca berkas gambar dari disk.
  Future<ClassicalOcrResult> runFile(String path) async {
    final File file = File(path);
    if (!file.existsSync()) {
      throw OcrEngineException('File gambar tidak ditemukan: $path');
    }
    final List<int> bytes = await file.readAsBytes();
    final img.Image? decoded = img.decodeImage(Uint8List.fromList(bytes));
    if (decoded == null) {
      throw OcrEngineException('Gambar tidak dapat didekode: $path');
    }
    final img.Image upright = img.bakeOrientation(decoded);
    return run(GrayImage.fromImage(upright));
  }

  // -- Kontrak OcrEngine --------------------------------------------------

  /// Implementasi [OcrEngine]: hasilkan baris per kelompok kata.
  ///
  /// Penting: satu `OcrLine` per **kelompok kata**, bukan satu baris
  /// selebar gambar. `receipt_parser` mengelompokkan `OcrLine` menurut
  /// `centerY` lalu memakai `left`/`right` untuk memisahkan kolom nama
  /// dan harga. Bila seluruh baris dikirim sebagai satu OcrLine, kolom
  /// harga hilang dan total struk tidak terbaca.
  @override
  Future<OcrRawResult> recognize(String imagePath) async {
    if (!_ready) {
      await load();
    }

    final ClassicalOcrResult result = await runFile(imagePath);

    final List<OcrLine> out = <OcrLine>[];
    for (final PredictedLine line in result.lines) {
      out.addAll(_splitIntoWordLines(line));
    }

    return OcrRawResult(
      lines: out,
      fullText: result.text,
    );
  }

  /// Pecah satu baris menjadi beberapa `OcrLine` per kelompok kata.
  List<OcrLine> _splitIntoWordLines(PredictedLine line) {
    final List<PredictedChar> chars = line.chars;
    if (chars.isEmpty) return const <OcrLine>[];

    final double threshold = _spaceThreshold(chars);

    final List<OcrLine> words = <OcrLine>[];
    final StringBuffer current = StringBuffer();
    int startLeft = chars.first.left;
    int endRight = chars.first.right;
    int top = chars.first.top;
    int bottom = chars.first.bottom;

    void flush() {
      final String text = current.toString().trim();
      if (text.isNotEmpty) {
        words.add(
          OcrLine(
            text: text,
            left: startLeft.toDouble(),
            top: top.toDouble(),
            right: endRight.toDouble(),
            bottom: bottom.toDouble(),
          ),
        );
      }
      current.clear();
    }

    current.write(chars.first.char);
    for (int i = 1; i < chars.length; i++) {
      final PredictedChar c = chars[i];
      final int gap = c.left - chars[i - 1].right;
      final bool glued = _gluedBySeparator(chars[i - 1].char, c.char);
      if (gap > threshold && !glued) {
        flush();
        startLeft = c.left;
        top = c.top;
        bottom = c.bottom;
      }
      current.write(c.char);
      endRight = c.right;
      if (c.top < top) top = c.top;
      if (c.bottom > bottom) bottom = c.bottom;
    }
    flush();

    return words;
  }

  @override
  Future<void> dispose() async {
    // Tidak ada sumber daya native yang perlu dilepas.
  }
}
