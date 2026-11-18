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
    if (score >= 80) {
      gaugeColor = SomythraTheme.safeGreen;
    } else if (score >= 60) {
      gaugeColor = SomythraTheme.primaryCyan;
    } else if (score >= 40) {
      gaugeColor = SomythraTheme.warningOrange;
    } else {
      gaugeColor = SomythraTheme.criticalRed;
    }

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: SomythraTheme.card,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: SomythraTheme.cardBorder),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: const [
              Text(
                'CARDIAC ENHANCEMENT SCORE (CES)',
                style: TextStyle(
                  color: SomythraTheme.textSecondary,
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.0,
                ),
              ),
              Icon(Icons.auto_graph_rounded, color: SomythraTheme.primaryCyan, size: 18),
            ],
          ),
          const SizedBox(height: 20),
          SizedBox(
            height: 150,
            width: 150,
            child: Stack(
              alignment: Alignment.center,
              children: [
                CustomPaint(
                  size: const Size(150, 150),
                  painter: _GaugePainter(score: score, color: gaugeColor),
                ),
                Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      score.toStringAsFixed(1),
                      style: const TextStyle(
                        color: SomythraTheme.textPrimary,
                        fontSize: 34,
                        fontWeight: FontWeight.w800,
                        fontFamily: 'monospace',
                      ),
                    ),
                    const Text(
                      'CES / 100',
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
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
            decoration: BoxDecoration(
              color: gaugeColor.withOpacity(0.12),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: gaugeColor.withOpacity(0.3)),
            ),
            child: Text(
              rating,
              style: TextStyle(
                color: gaugeColor,
                fontSize: 12,
                fontWeight: FontWeight.w700,
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
    final radius = min(size.width / 2, size.height / 2) - 12;
    const strokeWidth = 12.0;

    // Track arc (240 degrees, from 150 to 390 deg)
    const startAngle = 135 * (pi / 180);
    const sweepAngle = 270 * (pi / 180);

    final trackPaint = Paint()
      ..color = SomythraTheme.cardBorder
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

    // Active score arc
    final progressSweep = sweepAngle * (score.clamp(0.0, 100.0) / 100.0);
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
