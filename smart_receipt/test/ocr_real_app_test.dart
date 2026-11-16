import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:smart_receipt/ocr/classical/classical_ocr_engine.dart';
import 'package:smart_receipt/ocr/image_quality.dart';
import 'package:smart_receipt/ocr/ocr_raw_text.dart';
import 'package:smart_receipt/ocr/receipt_parser.dart';

/// Pemeriksaan jalur aplikasi penuh pada foto struk **asli**:
/// pra-proses (rotasi/blur/kontras) -> OCR klasik -> parser struk.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('jalur aplikasi membaca struk asli', () async {
    final Directory dir = Directory(r'C:\laragon\www\pcvk\ML\Dataset Citra');
    if (!dir.existsSync()) {
      // ignore: avoid_print
      print('Dataset Citra tidak ada; uji dilewati.');
      return;
    }
    final List<File> all = dir
        .listSync()
        .whereType<File>()
        .where((File f) => f.path.toLowerCase().endsWith('.jpg'))
        .toList()
      ..sort((File a, File b) => a.path.compareTo(b.path));

    final List<File> sample = <File>[
      ...all.where((File f) => f.path.contains('receipt-00')).take(3),
      ...all.where((File f) => f.path.contains('CamScanner')).take(3),
    ];

    final ImagePreprocessor pre = ImagePreprocessor();
    final ClassicalOcrEngine engine = ClassicalOcrEngine();
    await engine.load();
    final ReceiptParser parser = ReceiptParser();
    final Directory tmp = Directory.systemTemp.createTempSync('ocr_app');

    int withText = 0;
    for (final File f in sample) {
      final String out = '${tmp.path}/${f.uri.pathSegments.last}.jpg';
      await pre.prepare(sourcePath: f.path, outputPath: out);
      final OcrRawResult raw = await engine.recognize(out);
      final ParsedReceipt parsed = parser.parse(raw);
      if (!raw.isEmpty) withText++;
      // ignore: avoid_print
      print(
        '${f.uri.pathSegments.last.substring(0, 26)} '
        'kata=${raw.lineCount} '
        'item=${parsed.items.length} '
        'total=${parsed.total} '
        'toko="${parsed.store}"',
      );
    }
    tmp.deleteSync(recursive: true);

    expect(withText, sample.length, reason: 'semua struk harus menghasilkan teks');
  }, timeout: const Timeout(Duration(minutes: 10)));
}
