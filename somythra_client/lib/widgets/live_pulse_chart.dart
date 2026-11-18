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
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: SomythraTheme.card,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: SomythraTheme.cardBorder, width: 1.2),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.35),
            blurRadius: 16,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header Row
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: lineColor.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(4),
                        border: Border.all(color: lineColor.withOpacity(0.4)),
                      ),
                      child: Text(
                        'OSC-01',
                        style: TextStyle(
                          color: lineColor,
                          fontSize: 9,
                          fontWeight: FontWeight.w800,
                          fontFamily: 'monospace',
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        label.toUpperCase(),
                        style: const TextStyle(
                          color: SomythraTheme.textSecondary,
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 1.1,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 10),
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 7,
                    height: 7,
                    decoration: BoxDecoration(
                      color: lineColor,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: lineColor,
                          blurRadius: 6,
                          spreadRadius: 1,
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 6),
                  const Text(
                    'LIVE 100Hz',
                    style: TextStyle(
                      color: SomythraTheme.textMuted,
                      fontSize: 9.5,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.8,
                      fontFamily: 'monospace',
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 14),

          // Custom Oscilloscope Canvas
          SizedBox(
            height: 150,
            width: double.infinity,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: CustomPaint(
                painter: _PulseWavePainter(
                  values: values,
                  lineColor: lineColor,
                  minVal: minVal,
                  maxVal: maxVal,
                ),
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
    // 1. Oscilloscope Background Screen
    final bgPaint = Paint()..color = const Color(0xFF090E15);
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), bgPaint);

    // 2. High-Tech Precision Grid
    final gridPaint = Paint()
      ..color = const Color(0xFF162335)
      ..strokeWidth = 1;

    final fineGridPaint = Paint()
      ..color = const Color(0xFF101A27)
      ..strokeWidth = 0.5;

    // Horizontal grid lines
    const hDivisions = 6;
    for (int i = 1; i < hDivisions; i++) {
      final y = size.height * (i / hDivisions);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), i % 2 == 0 ? gridPaint : fineGridPaint);
    }

    // Vertical grid lines
    const vDivisions = 12;
    for (int i = 1; i < vDivisions; i++) {
      final x = size.width * (i / vDivisions);
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), i % 3 == 0 ? gridPaint : fineGridPaint);
    }

    if (values.length < 2) return;

    // 3. Coordinate calculation
    final stepX = size.width / (values.length - 1);
    final range = (maxVal - minVal) <= 0 ? 1.0 : (maxVal - minVal);

    final path = Path();
    final fillPath = Path();

    for (int i = 0; i < values.length; i++) {
      final val = values[i].clamp(minVal, maxVal);
      final normalized = 1.0 - ((val - minVal) / range);
      final x = i * stepX;
      final y = normalized * (size.height - 18) + 9;

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

    // 4. Area Glow Fill Shader
    final fillPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          lineColor.withOpacity(0.22),
          lineColor.withOpacity(0.04),
          Colors.transparent,
        ],
        stops: const [0.0, 0.6, 1.0],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height))
      ..style = PaintingStyle.fill;

    canvas.drawPath(fillPath, fillPaint);

    // 5. Outer Laser Glow Pass (soft wide bloom)
    final glowPaint = Paint()
      ..color = lineColor.withOpacity(0.35)
      ..strokeWidth = 4.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    canvas.drawPath(path, glowPaint);

    // 6. Inner Sharp Laser Core Pass
    final corePaint = Paint()
      ..color = lineColor
      ..strokeWidth = 2.2
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    canvas.drawPath(path, corePaint);

    // 7. Oscilloscope Sweep Tracer Head (Glow dot + Radar Rings)
    final lastVal = values.last.clamp(minVal, maxVal);
    final lastNorm = 1.0 - ((lastVal - minVal) / range);
    final headPoint = Offset(size.width, lastNorm * (size.height - 18) + 9);

    // Outer pulse halo
    canvas.drawCircle(
      headPoint,
      10,
      Paint()..color = lineColor.withOpacity(0.25),
    );
    // Inner bright bead
    canvas.drawCircle(
      headPoint,
      4.5,
      Paint()..color = Colors.white,
    );
    canvas.drawCircle(
      headPoint,
      2.5,
      Paint()..color = lineColor,
    );
  }

  @override
  bool shouldRepaint(covariant _PulseWavePainter oldDelegate) => true;
}
