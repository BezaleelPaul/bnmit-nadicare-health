import 'dart:math';
import 'package:flutter/material.dart';
import '../models/user_profile.dart';
import '../models/telemetry.dart';
import '../theme/somythra_theme.dart';

class AutonomicNexusWidget extends StatefulWidget {
  final TelemetryData telemetry;
  final UserProfile profile;

  const AutonomicNexusWidget({
    super.key,
    required this.telemetry,
    required this.profile,
  });

  @override
  State<AutonomicNexusWidget> createState() => _AutonomicNexusWidgetState();
}

class _AutonomicNexusWidgetState extends State<AutonomicNexusWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);

    _pulseAnimation = Tween<double>(begin: 0.94, end: 1.08).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
  }

  @override
  void didUpdateWidget(covariant AutonomicNexusWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Adjust animation speed based on heart rate if valid
    final bpm = widget.telemetry.hr.clamp(40.0, 180.0);
    final periodMs = (60000.0 / bpm).round().clamp(350, 1500);
    if (_pulseController.duration?.inMilliseconds != periodMs) {
      _pulseController.duration = Duration(milliseconds: periodMs);
      if (!_pulseController.isAnimating) {
        _pulseController.repeat(reverse: true);
      }
    }
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final hr = widget.telemetry.hr;
    final maxHr = widget.profile.maxHr;
    final tanakaPct = (hr / maxHr).clamp(0.0, 1.0);
    final headroom = (maxHr - hr).clamp(0.0, 200.0);

    Color statusColor;
    String statusTitle;
    if (tanakaPct > 0.85) {
      statusColor = SomythraTheme.criticalRed;
      statusTitle = 'SYMPATHETIC HYPERAROUSAL';
    } else if (tanakaPct > 0.70) {
      statusColor = SomythraTheme.warningOrange;
      statusTitle = 'CARDIOVASCULAR EXERTION';
    } else {
      statusColor = SomythraTheme.safeGreen;
      statusTitle = 'AUTONOMIC HOMEOSTASIS';
    }

    return Container(
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        color: SomythraTheme.card,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: SomythraTheme.cardBorder, width: 1.2),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.4),
            blurRadius: 20,
            offset: const Offset(0, 6),
          ),
          BoxShadow(
            color: statusColor.withOpacity(0.08),
            blurRadius: 24,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header Bar
          Wrap(
            alignment: WrapAlignment.spaceBetween,
            crossAxisAlignment: WrapCrossAlignment.center,
            spacing: 12,
            runSpacing: 8,
            children: [
              Wrap(
                spacing: 8,
                runSpacing: 4,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
                    decoration: BoxDecoration(
                      color: SomythraTheme.primaryCyan.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(5),
                      border: Border.all(color: SomythraTheme.primaryCyan.withOpacity(0.4)),
                    ),
                    child: const Text(
                      'AUTONOMIC NEXUS',
                      style: TextStyle(
                        color: SomythraTheme.primaryCyan,
                        fontSize: 9.5,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.2,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ),
                  Text(
                    statusTitle,
                    style: TextStyle(
                      color: statusColor,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.8,
                    ),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: SomythraTheme.surface,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: SomythraTheme.cardBorder),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.shield_outlined, color: SomythraTheme.primaryCyan, size: 14),
                    const SizedBox(width: 6),
                    Text(
                      'Tanaka Headroom: +${headroom.toStringAsFixed(0)} BPM',
                      style: const TextStyle(
                        color: SomythraTheme.textSecondary,
                        fontSize: 10.5,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),

          // Central Visualizer: Concentric Bio-Rings & Beating Core
          Center(
            child: SizedBox(
              width: 220,
              height: 220,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  // Outer Tanaka Arc & Background Track
                  CustomPaint(
                    size: const Size(220, 220),
                    painter: _NexusRingsPainter(
                      progress: tanakaPct,
                      color: statusColor,
                      hrvProgress: (widget.telemetry.hrv / 100.0).clamp(0.1, 1.0),
                    ),
                  ),

                  // Beating Heart Core with Radial Glow
                  AnimatedBuilder(
                    animation: _pulseAnimation,
                    builder: (context, child) {
                      return Transform.scale(
                        scale: _pulseAnimation.value,
                        child: Container(
                          width: 110,
                          height: 110,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            gradient: RadialGradient(
                              colors: [
                                SomythraTheme.heartRose.withOpacity(0.22),
                                SomythraTheme.surface.withOpacity(0.85),
                              ],
                            ),
                            border: Border.all(
                              color: SomythraTheme.heartRose.withOpacity(0.5),
                              width: 1.5,
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: SomythraTheme.heartRose.withOpacity(0.35),
                                blurRadius: 18,
                                spreadRadius: 1,
                              ),
                            ],
                          ),
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.favorite_rounded,
                                color: SomythraTheme.heartRose,
                                size: 26,
                                shadows: [
                                  Shadow(
                                    color: SomythraTheme.heartRose,
                                    blurRadius: 10,
                                  ),
                                ],
                              ),
                              const SizedBox(height: 2),
                              Text(
                                hr.toStringAsFixed(0),
                                style: const TextStyle(
                                  color: SomythraTheme.textPrimary,
                                  fontSize: 26,
                                  fontWeight: FontWeight.w900,
                                  fontFamily: 'monospace',
                                  letterSpacing: -1.0,
                                ),
                              ),
                              const Text(
                                'BPM',
                                style: TextStyle(
                                  color: SomythraTheme.heartRose,
                                  fontSize: 9.5,
                                  fontWeight: FontWeight.w800,
                                  letterSpacing: 0.8,
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 18),

          // Lower Tri-Metric Status Bar
          Row(
            children: [
              Expanded(
                child: _buildNexusSubPill(
                  label: 'CARDIAC CAPACITY',
                  value: '${(tanakaPct * 100).toStringAsFixed(0)}%',
                  detail: 'Of ${maxHr.toStringAsFixed(0)} Max',
                  color: statusColor,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _buildNexusSubPill(
                  label: 'VAGAL MODULATION',
                  value: '${widget.telemetry.hrv.toStringAsFixed(1)} ms',
                  detail: 'Resting RMSSD',
                  color: SomythraTheme.primaryCyan,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _buildNexusSubPill(
                  label: 'RHYTHM INTERVAL',
                  value: '${widget.telemetry.meanRr.toStringAsFixed(0)} ms',
                  detail: 'Mean Inter-Beat (RR)',
                  color: SomythraTheme.cyanGlow,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildNexusSubPill({
    required String label,
    required String value,
    required String detail,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: SomythraTheme.surface,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: SomythraTheme.cardBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(
              color: SomythraTheme.textMuted,
              fontSize: 9,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.8,
            ),
          ),
          const SizedBox(height: 3),
          Text(
            value,
            style: TextStyle(
              color: color,
              fontSize: 15,
              fontWeight: FontWeight.w800,
              fontFamily: 'monospace',
            ),
          ),
          const SizedBox(height: 1),
          Text(
            detail,
            style: const TextStyle(
              color: SomythraTheme.textSecondary,
              fontSize: 9.5,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}

class _NexusRingsPainter extends CustomPainter {
  final double progress;
  final Color color;
  final double hrvProgress;

  _NexusRingsPainter({
    required this.progress,
    required this.color,
    required this.hrvProgress,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final outerRadius = size.width / 2 - 10;
    final innerRadius = size.width / 2 - 24;

    // 1. Outer Track (Tanaka Threshold)
    final outerTrackPaint = Paint()
      ..color = const Color(0xFF141F2D)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 6;
    canvas.drawCircle(center, outerRadius, outerTrackPaint);

    // 2. Active Tanaka Arc (270 degrees sweep)
    const startAngle = -pi / 2;
    final sweepAngle = 2 * pi * progress;

    // Outer Glow
    final outerGlowPaint = Paint()
      ..color = color.withOpacity(0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 10
      ..strokeCap = StrokeCap.round;
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: outerRadius),
      startAngle,
      sweepAngle,
      false,
      outerGlowPaint,
    );

    // Outer Core
    final outerArcPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 6
      ..strokeCap = StrokeCap.round;
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: outerRadius),
      startAngle,
      sweepAngle,
      false,
      outerArcPaint,
    );

    // 3. Middle Vagal Track (HRV)
    final innerTrackPaint = Paint()
      ..color = const Color(0xFF101924)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4;
    canvas.drawCircle(center, innerRadius, innerTrackPaint);

    final innerArcPaint = Paint()
      ..color = SomythraTheme.primaryCyan
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;

    final hrvSweep = 2 * pi * hrvProgress;
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: innerRadius),
      startAngle,
      hrvSweep,
      false,
      innerArcPaint,
    );

    // 4. Subtle Radial Ticks
    final tickPaint = Paint()
      ..color = SomythraTheme.cardBorder
      ..strokeWidth = 1;
    for (int i = 0; i < 36; i++) {
      final a = (i * 10) * (pi / 180);
      final r1 = outerRadius + 5;
      final r2 = outerRadius + (i % 9 == 0 ? 9 : 7);
      canvas.drawLine(
        Offset(center.dx + r1 * cos(a), center.dy + r1 * sin(a)),
        Offset(center.dx + r2 * cos(a), center.dy + r2 * sin(a)),
        tickPaint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _NexusRingsPainter oldDelegate) =>
      oldDelegate.progress != progress ||
      oldDelegate.color != color ||
      oldDelegate.hrvProgress != hrvProgress;
}
