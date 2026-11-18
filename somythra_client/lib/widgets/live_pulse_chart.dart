import 'package:flutter/material.dart';
import '../theme/somythra_theme.dart';

class LivePulseChart extends StatelessWidget {
  final List<double> values;
  final Color lineColor;
  final double minVal;
  final double maxVal;
  final String label;

  const LivePulseChart({
    super.key,
    required this.values,
    this.lineColor = SomythraTheme.heartRose,
    this.minVal = 40.0,
    this.maxVal = 160.0,
    this.label = 'Real-Time Waveform',
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: SomythraTheme.card,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: SomythraTheme.cardBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                label.toUpperCase(),
                style: const TextStyle(
                  color: SomythraTheme.textSecondary,
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.0,
                ),
              ),
              Row(
                children: [
                  Container(
                    width: 8,
                    height: 8,
                    decoration: BoxDecoration(
                      color: lineColor,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 6),
                  const Text(
                    'LIVE FEED',
                    style: TextStyle(
                      color: SomythraTheme.textMuted,
                      fontSize: 10,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 16),
          SizedBox(
            height: 140,
            width: double.infinity,
            child: CustomPaint(
              painter: _PulseWavePainter(
                values: values,
                lineColor: lineColor,
                minVal: minVal,
                maxVal: maxVal,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _PulseWavePainter extends CustomPainter {
  final List<double> values;
  final Color lineColor;
  final double minVal;
  final double maxVal;

  _PulseWavePainter({
    required this.values,
    required this.lineColor,
    required this.minVal,
    required this.maxVal,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Draw background grid lines
    final gridPaint = Paint()
      ..color = SomythraTheme.cardBorder.withOpacity(0.3)
      ..strokeWidth = 1;

    for (int i = 1; i <= 3; i++) {
      final y = size.height * (i / 4.0);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    if (values.length < 2) return;

    final linePaint = Paint()
      ..color = lineColor
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final path = Path();
    final fillPath = Path();

    final stepX = size.width / (values.length - 1);
    final range = (maxVal - minVal) <= 0 ? 1.0 : (maxVal - minVal);

    for (int i = 0; i < values.length; i++) {
      final val = values[i].clamp(minVal, maxVal);
      final normalized = 1.0 - ((val - minVal) / range);
      final x = i * stepX;
      final y = normalized * (size.height - 12) + 6;

      if (i == 0) {
        path.moveTo(x, y);
        fillPath.moveTo(x, size.height);
        fillPath.lineTo(x, y);
      } else {
        path.lineTo(x, y);
        fillPath.lineTo(x, y);
      }
    }

    fillPath.lineTo(size.width, size.height);
    fillPath.close();

    final fillPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          lineColor.withOpacity(0.25),
          lineColor.withOpacity(0.0),
        ],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height))
      ..style = PaintingStyle.fill;

    canvas.drawPath(fillPath, fillPaint);
    canvas.drawPath(path, linePaint);

    // Draw glowing current point at the end
    final lastVal = values.last.clamp(minVal, maxVal);
    final lastNorm = 1.0 - ((lastVal - minVal) / range);
    final lastPoint = Offset(size.width, lastNorm * (size.height - 12) + 6);

    canvas.drawCircle(
      lastPoint,
      5,
      Paint()..color = lineColor,
    );
    canvas.drawCircle(
      lastPoint,
      10,
      Paint()..color = lineColor.withOpacity(0.35),
    );
  }

  @override
  bool shouldRepaint(covariant _PulseWavePainter oldDelegate) => true;
}
