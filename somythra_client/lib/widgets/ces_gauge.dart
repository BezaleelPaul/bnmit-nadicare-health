import 'dart:math';
import 'package:flutter/material.dart';
import '../theme/somythra_theme.dart';

class CesGauge extends StatelessWidget {
  final double score;
  final String rating;

  const CesGauge({
    super.key,
    required this.score,
    required this.rating,
  });

  @override
  Widget build(BuildContext context) {
    Color gaugeColor;
    String tierLabel;

    if (score >= 80) {
      gaugeColor = SomythraTheme.safeGreen;
      tierLabel = 'ZONE A // OPTIMAL';
    } else if (score >= 60) {
      gaugeColor = SomythraTheme.primaryCyan;
      tierLabel = 'ZONE B // NOMINAL';
    } else if (score >= 40) {
      gaugeColor = SomythraTheme.warningOrange;
      tierLabel = 'ZONE C // STRAINED';
    } else {
      gaugeColor = SomythraTheme.criticalRed;
      tierLabel = 'ZONE D // EXHAUSTION';
    }

    return Container(
      padding: const EdgeInsets.all(20),
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
          BoxShadow(
            color: gaugeColor.withOpacity(0.08),
            blurRadius: 20,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        children: [
          // Header Row
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: gaugeColor.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(4),
                      border: Border.all(color: gaugeColor.withOpacity(0.4)),
                    ),
                    child: Text(
                      tierLabel,
                      style: TextStyle(
                        color: gaugeColor,
                        fontSize: 9,
                        fontWeight: FontWeight.w800,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  const Text(
                    'CARDIAC ENHANCEMENT SCORE',
                    style: TextStyle(
                      color: SomythraTheme.textSecondary,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 1.0,
                    ),
                  ),
                ],
              ),
              Icon(Icons.auto_graph_rounded, color: gaugeColor, size: 18),
            ],
          ),
          const SizedBox(height: 18),

          // High-Tech Radial Gauge
          SizedBox(
            height: 155,
            width: 155,
            child: Stack(
              alignment: Alignment.center,
              children: [
                CustomPaint(
                  size: const Size(155, 155),
                  painter: _GaugePainter(score: score, color: gaugeColor),
                ),
                Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      score.toStringAsFixed(1),
                      style: TextStyle(
                        color: SomythraTheme.textPrimary,
                        fontSize: 34,
                        fontWeight: FontWeight.w900,
                        fontFamily: 'monospace',
                        letterSpacing: -1.0,
                        shadows: [
                          Shadow(
                            color: gaugeColor.withOpacity(0.4),
                            blurRadius: 14,
                            offset: const Offset(0, 2),
                          ),
                        ],
                      ),
                    ),
                    Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          'CES',
                          style: TextStyle(
                            color: gaugeColor,
                            fontSize: 11,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 0.5,
                          ),
                        ),
                        const Text(
                          ' / 100',
                          style: TextStyle(
                            color: SomythraTheme.textMuted,
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // Rating Pill with Neon Halo
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
            decoration: BoxDecoration(
              color: gaugeColor.withOpacity(0.12),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: gaugeColor.withOpacity(0.4), width: 1.2),
              boxShadow: [
                BoxShadow(
                  color: gaugeColor.withOpacity(0.15),
                  blurRadius: 8,
                ),
              ],
            ),
            child: Text(
              rating,
              style: TextStyle(
                color: gaugeColor,
                fontSize: 11.5,
                fontWeight: FontWeight.w800,
                letterSpacing: 0.5,
              ),
              textAlign: TextAlign.center,
            ),
          ),
        ],
      ),
    );
  }
}

class _GaugePainter extends CustomPainter {
  final double score;
  final Color color;

  _GaugePainter({required this.score, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = min(size.width / 2, size.height / 2) - 14;
    const strokeWidth = 10.0;

    // Track arc (270 degrees, from 135 to 405 deg)
    const startAngle = 135 * (pi / 180);
    const sweepAngle = 270 * (pi / 180);

    // 1. Draw outer tick marks
    final tickPaint = Paint()
      ..color = SomythraTheme.cardBorder
      ..strokeWidth = 1.5;

    const totalTicks = 28;
    for (int i = 0; i <= totalTicks; i++) {
      final angle = startAngle + (sweepAngle * (i / totalTicks));
      final innerR = radius + 9;
      final outerR = radius + 14;
      final p1 = Offset(center.dx + innerR * cos(angle), center.dy + innerR * sin(angle));
      final p2 = Offset(center.dx + outerR * cos(angle), center.dy + outerR * sin(angle));
      canvas.drawLine(p1, p2, tickPaint);
    }

    // 2. Track Background Arc
    final trackPaint = Paint()
      ..color = const Color(0xFF162335)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle,
      sweepAngle,
      false,
      trackPaint,
    );

    // 3. Active score progress arc
    final progressSweep = sweepAngle * (score.clamp(0.0, 100.0) / 100.0);

    // Glow pass
    final glowPaint = Paint()
      ..color = color.withOpacity(0.35)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth + 4
      ..strokeCap = StrokeCap.round;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle,
      progressSweep,
      false,
      glowPaint,
    );

    // Core laser pass
    final activePaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle,
      progressSweep,
      false,
      activePaint,
    );
  }

  @override
  bool shouldRepaint(covariant _GaugePainter oldDelegate) =>
      oldDelegate.score != score || oldDelegate.color != color;
}
