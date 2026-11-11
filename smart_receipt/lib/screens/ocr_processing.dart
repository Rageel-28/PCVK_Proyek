import 'dart:io';

import 'package:flutter/material.dart';

import '../theme/app_colors.dart';
import '../widgets/app_widgets.dart';
import 'category_classification.dart';
import '../ocr/image_quality.dart';
import '../ocr/ocr_raw_text.dart';
import '../ocr/ocr_service.dart';
import '../widgets/receipt_mock.dart';
import '../theme/theme_controller.dart';

/// Screen 3 - OCR Processing.
///
/// Runs the real OCR pipeline on the captured image:
/// preprocess (rotation / blur / low-light) -> classical OCR
/// (binarization, segmentation, HOG + Linear SVM) -> structured
/// transaction parsing. Shows live progress and handles failures
/// (blurred, dark, or unreadable receipts).
class OcrProcessingScreen extends StatefulWidget {
  const OcrProcessingScreen({super.key, required this.imagePath});

  final String imagePath;

  @override
  State<OcrProcessingScreen> createState() => _OcrProcessingScreenState();
}

class _OcrProcessingScreenState extends State<OcrProcessingScreen> {
  final OcrService _service = OcrService();

  OcrStage _stage = OcrStage.preprocessing;
  OcrFailure? _failure;
  OcrScanResult? _result;

  @override
  void initState() {
    super.initState();
    _run();
  }

  @override
  void dispose() {
    _service.dispose();
    super.dispose();
  }

  Future<void> _run() async {
    setState(() {
      _failure = null;
      _result = null;
      _stage = OcrStage.preprocessing;
    });
    try {
      final OcrScanResult result = await _service.process(
        widget.imagePath,
        onStage: (OcrStage stage) {
          if (mounted) {
            setState(() => _stage = stage);
          }
        },
      );
      if (!mounted) {
        return;
      }
      setState(() => _result = result);
      // Brief pause so the completed checklist is visible before moving on.
      await Future<void>.delayed(const Duration(milliseconds: 350));
      if (!mounted) {
        return;
      }
      Navigator.pushReplacement(
        context,
        MaterialPageRoute<void>(
          builder: (BuildContext context) =>
              CategoryClassificationScreen(scan: result),
        ),
      );
    } on OcrFailure catch (e) {
      if (mounted) {
        setState(() => _failure = e);
      }
    } catch (e) {
      if (mounted) {
        setState(
          () => _failure = OcrFailure('Terjadi kesalahan tak terduga.', hint: '$e'),
        );
      }
    }
  }

  int get _activeIndex {
    switch (_stage) {
      case OcrStage.preprocessing:
        return 0;
      case OcrStage.recognizing:
        return 1;
      case OcrStage.parsing:
      case OcrStage.done:
        return 2;
    }
  }

  @override
  Widget build(BuildContext context) {
    ThemeScope.watch(context);

    if (_failure != null) {
      return _FailureView(
        failure: _failure!,
        onRetry: _run,
        onBack: () => Navigator.maybePop(context),
      );
    }

    final List<_OcrStep> steps = <_OcrStep>[
      const _OcrStep(
        label: 'Menyeimbangkan & memutar gambar',
        icon: Icons.auto_fix_high,
      ),
      _OcrStep(
        label: _result != null && _result!.hasRecognizedText
            ? 'Membaca teks (OCR) — ${_result!.raw.lineCount} baris'
            : 'Membaca teks (OCR)',
        icon: Icons.text_fields,
      ),
      const _OcrStep(
        label: 'Menstrukturkan data transaksi',
        icon: Icons.table_rows_outlined,
      ),
    ];

    final double progress = _result != null
        ? 1.0
        : (_stage.index + 0.5).clamp(0.15, 0.95).toDouble();

    return Scaffold(
      key: ValueKey<AppPalette>(AppColors.palette),
      backgroundColor: AppColors.bg,
      body: SafeArea(
        bottom: false,
        child: Column(
          children: <Widget>[
            const AppStatusBar(),
            AppTopBar(
              title: 'Memproses OCR',
              subtitle: _result != null ? 'Selesai' : 'Mohon tunggu sebentar',
              onBack: () => Navigator.maybePop(context),
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(22, 6, 22, 24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: <Widget>[
                    _Preview(imagePath: widget.imagePath),
                    if (_result != null && _result!.quality.hasIssue) ...<Widget>[
                      const SizedBox(height: 16),
                      _QualityNotice(report: _result!.quality),
                    ],
                    const SizedBox(height: 22),
                    Row(
                      children: <Widget>[
                        Text(
                          'Progres',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w700,
                            color: AppColors.textDark,
                          ),
                        ),
                        const Spacer(),
                        Text(
                          '${(progress * 100).round()}%',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w700,
                            color: AppColors.amber,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: TweenAnimationBuilder<double>(
                        tween: Tween<double>(begin: 0, end: progress),
                        duration: const Duration(milliseconds: 400),
                        curve: Curves.easeOut,
                        builder: (BuildContext context, double value, Widget? child) {
                          return LinearProgressIndicator(
                            value: value,
                            minHeight: 9,
                            backgroundColor: AppColors.track,
                            valueColor: AlwaysStoppedAnimation<Color>(AppColors.orange),
                          );
                        },
                      ),
                    ),
                    const SizedBox(height: 22),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 6),
                      decoration: BoxDecoration(
                        color: AppColors.card,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Column(
                        children: <Widget>[
                          for (int i = 0; i < steps.length; i++)
                            _StepRow(
                              step: steps[i],
                              done: i < _activeIndex || _result != null,
                              active: i == _activeIndex && _result == null,
                              last: i == steps.length - 1,
                            ),
                        ],
                      ),
                    ),
                    if (_result != null && _result!.hasRecognizedText) ...<Widget>[
                      const SizedBox(height: 18),
                      _RawTextCard(lines: _result!.raw.lines),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _OcrStep {
  const _OcrStep({required this.label, required this.icon});

  final String label;
  final IconData icon;
}

class _Preview extends StatelessWidget {
  const _Preview({required this.imagePath});

  final String imagePath;

  @override
  Widget build(BuildContext context) {
    final File file = File(imagePath);
    return ClipRRect(
      borderRadius: BorderRadius.circular(22),
      child: Container(
        height: 220,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: AppColors.panelGradient,
          ),
        ),
        child: Stack(
          fit: StackFit.expand,
          children: <Widget>[
            // Show the actual captured receipt, falling back to a mock when the
            // file is unavailable.
            if (file.existsSync())
              Image.file(file, fit: BoxFit.cover)
            else
              Align(
                alignment: Alignment.bottomCenter,
                child: Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: Opacity(
                    opacity: 0.55,
                    child: Transform.rotate(
                      angle: -0.02,
                      child: const SizedBox(
                        width: 112,
                        height: 148,
                        child: FittedBox(
                          fit: BoxFit.contain,
                          child: ReceiptMock(width: 132),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            const Positioned(left: 0, right: 0, top: 104, child: _ScanLine()),
            const Positioned(
              top: 14,
              left: 14,
              child: _Tag(icon: Icons.document_scanner_outlined, label: 'Scan'),
            ),
          ],
        ),
      ),
    );
  }
}

class _ScanLine extends StatelessWidget {
  const _ScanLine();

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 3,
      margin: const EdgeInsets.symmetric(horizontal: 26),
      decoration: BoxDecoration(
        color: AppColors.orange,
        borderRadius: BorderRadius.circular(3),
        boxShadow: <BoxShadow>[
          BoxShadow(color: AppColors.orange.withValues(alpha: 0.7), blurRadius: 16),
        ],
      ),
    );
  }
}

class _Tag extends StatelessWidget {
  const _Tag({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.35),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          Icon(icon, size: 14, color: AppColors.orange),
          const SizedBox(width: 6),
          Text(
            label,
            style: const TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }
}

/// Warns the user about image problems detected by preprocessing.
class _QualityNotice extends StatelessWidget {
  const _QualityNotice({required this.report});

  final ImageQualityReport report;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.amberSoft,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Icon(Icons.warning_amber_rounded, color: AppColors.amber, size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text(
                  'Kualitas gambar perlu diperhatikan',
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textDark,
                  ),
                ),
                const SizedBox(height: 4),
                for (final String issue in report.issues)
                  Text(
                    '• $issue',
                    style: TextStyle(
                      fontSize: 12,
                      height: 1.4,
                      color: AppColors.textLabel,
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Shows the raw recognised text so the user can see what OCR captured.
class _RawTextCard extends StatelessWidget {
  const _RawTextCard({required this.lines});

  final List<OcrLine> lines;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            'Teks Terdeteksi',
            style: TextStyle(
              fontSize: 13.5,
              fontWeight: FontWeight.w700,
              color: AppColors.textDark,
            ),
          ),
          const SizedBox(height: 10),
          ConstrainedBox(
            constraints: const BoxConstraints(maxHeight: 180),
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  for (final OcrLine line in lines.take(40))
                    Padding(
                      padding: const EdgeInsets.only(bottom: 3),
                      child: Text(
                        line.text,
                        style: TextStyle(
                          fontSize: 12,
                          height: 1.35,
                          color: AppColors.textLabel,
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Failure state with a helpful hint and retry.
class _FailureView extends StatelessWidget {
  const _FailureView({
    required this.failure,
    required this.onRetry,
    required this.onBack,
  });

  final OcrFailure failure;
  final VoidCallback onRetry;
  final VoidCallback onBack;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: Column(
          children: <Widget>[
            const AppStatusBar(),
            AppTopBar(title: 'OCR Gagal', onBack: onBack),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(22, 12, 22, 22),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: <Widget>[
                    Container(
                      width: 96,
                      height: 96,
                      decoration: BoxDecoration(
                        color: AppColors.redSoft,
                        shape: BoxShape.circle,
                      ),
                      child: Icon(
                        Icons.error_outline,
                        size: 52,
                        color: AppColors.red,
                      ),
                    ),
                    const SizedBox(height: 22),
                    Text(
                      failure.message,
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textDark,
                      ),
                    ),
                    if (failure.hint != null) ...<Widget>[
                      const SizedBox(height: 10),
                      Text(
                        failure.hint!,
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 13,
                          height: 1.5,
                          color: AppColors.textMuted,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(22, 0, 22, 22),
              child: Column(
                children: <Widget>[
                  PrimaryButton(
                    label: 'Coba Lagi',
                    icon: Icons.refresh,
                    onPressed: onRetry,
                  ),
                  const SizedBox(height: 12),
                  SecondaryButton(
                    label: 'Ambil Ulang Foto',
                    icon: Icons.photo_camera_outlined,
                    onPressed: onBack,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _StepRow extends StatelessWidget {
  const _StepRow({
    required this.step,
    required this.done,
    required this.active,
    required this.last,
  });

  final _OcrStep step;
  final bool done;
  final bool active;
  final bool last;

  @override
  Widget build(BuildContext context) {
    final bool complete = done || active;
    return Padding(
      padding: EdgeInsets.only(top: 14, bottom: last ? 14 : 0),
      child: Row(
        children: <Widget>[
          Container(
            width: 34,
            height: 34,
            decoration: BoxDecoration(
              color: complete ? AppColors.navy : AppColors.iconBox,
              borderRadius: BorderRadius.circular(11),
            ),
            child: done
                ? Icon(Icons.check, size: 17, color: AppColors.onNavy)
                : Icon(
                    step.icon,
                    size: 17,
                    color: active ? AppColors.orange : const Color(0xFF8A93A8),
                  ),
          ),
          const SizedBox(width: 13),
          Expanded(
            child: Text(
              step.label,
              style: TextStyle(
                fontSize: 13.5,
                fontWeight: active ? FontWeight.w700 : FontWeight.w500,
                color: active ? AppColors.textDark : AppColors.textLabel,
              ),
            ),
          ),
          if (active)
            SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: AppColors.orange,
                backgroundColor: AppColors.track,
              ),
            )
          else if (done)
            Icon(Icons.check_circle, size: 18, color: AppColors.green),
        ],
      ),
    );
  }
}
