import 'dart:async';
import 'package:flutter/material.dart';
import '../models/telemetry.dart';
import '../models/user_profile.dart';
import '../services/telemetry_stream.dart';
import '../theme/somythra_theme.dart';
import '../widgets/metric_card.dart';
import '../widgets/alert_banner.dart';
import '../widgets/live_pulse_chart.dart';
import '../widgets/ces_gauge.dart';

class DashboardScreen extends StatefulWidget {
  final UserProfile profile;
  const DashboardScreen({super.key, required this.profile});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  late TelemetryStreamService _streamService;
  StreamSubscription<TelemetryData>? _sub;

  TelemetryData? _latest;
  final List<double> _hrHistory = [];
  final int _maxHistory = 35;

  @override
  void initState() {
    super.initState();
    _streamService = TelemetryStreamService();
    _streamService.start();

    _sub = _streamService.stream.listen((data) {
      if (mounted) {
        setState(() {
          _latest = data;
          _hrHistory.add(data.hr);
          if (_hrHistory.length > _maxHistory) {
            _hrHistory.removeAt(0);
          }
        });
      }
    });
  }

  @override
  void dispose() {
    _sub?.cancel();
    _streamService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final telemetry = _latest ??
        TelemetryData(
          timestamp: DateTime.now(),
          hr: 72.0,
          hrv: 52.0,
          meanRr: 833.0,
          sdnn: 61.0,
          activityLoad: 1.0,
          cesScore: 78.0,
          cesRating: 'Optimal Interconnected State',
          safetyLevel: 'SAFE',
          safetyMessage: 'Physiological vitals stable within baseline limits.',
          stressLevel: 'Balanced / Low Stress',
          stressIndex: 42.0,
        );

    final isLiveWs = _streamService.isConnected;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header + Connection Badge
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Text(
                        'SOMYTHRA CORE',
                        style: TextStyle(
                          color: SomythraTheme.textPrimary,
                          fontSize: 24,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 0.5,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(
                          color: (isLiveWs
                                  ? SomythraTheme.safeGreen
                                  : SomythraTheme.warningOrange)
                              .withOpacity(0.15),
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(
                            color: isLiveWs
                                ? SomythraTheme.safeGreen
                                : SomythraTheme.warningOrange,
                          ),
                        ),
                        child: Text(
                          isLiveWs ? 'API CONNECTED' : 'SIMULATED FEED',
                          style: TextStyle(
                            color: isLiveWs
                                ? SomythraTheme.safeGreen
                                : SomythraTheme.warningOrange,
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'Real-time physiological telemetry and autonomic monitoring',
                    style: TextStyle(
                      color: SomythraTheme.textSecondary,
                      fontSize: 13,
                    ),
                  ),
                ],
              ),
              // User Profile Pill
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                decoration: BoxDecoration(
                  color: SomythraTheme.card,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: SomythraTheme.cardBorder),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.person_rounded,
                        color: SomythraTheme.primaryCyan, size: 16),
                    const SizedBox(width: 8),
                    Text(
                      '${widget.profile.name} (${widget.profile.age}y)',
                      style: const TextStyle(
                        color: SomythraTheme.textPrimary,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),

          // Tanaka Safety Alert Banner
          AlertBanner(
            level: telemetry.safetyLevel,
            message: telemetry.safetyMessage,
          ),
          const SizedBox(height: 24),

          // Metric Cards Grid
          LayoutBuilder(
            builder: (context, constraints) {
              final crossAxisCount = constraints.maxWidth > 900
                  ? 4
                  : (constraints.maxWidth > 600 ? 2 : 1);
              final itemWidth =
                  (constraints.maxWidth - ((crossAxisCount - 1) * 16)) /
                      crossAxisCount;

              return Wrap(
                spacing: 16,
                runSpacing: 16,
                children: [
                  SizedBox(
                    width: itemWidth,
                    child: MetricCard(
                      title: 'Heart Rate',
                      value: telemetry.hr.toStringAsFixed(0),
                      unit: 'BPM',
                      icon: Icons.favorite_rounded,
                      accentColor: SomythraTheme.heartRose,
                      subtitle:
                          'Tanaka Max: ${widget.profile.maxHr.toStringAsFixed(0)} BPM',
                    ),
                  ),
                  SizedBox(
                    width: itemWidth,
                    child: MetricCard(
                      title: 'HRV (RMSSD)',
                      value: telemetry.hrv.toStringAsFixed(1),
                      unit: 'ms',
                      icon: Icons.graphic_eq_rounded,
                      accentColor: SomythraTheme.primaryCyan,
                      subtitle:
                          'Baseline: ${widget.profile.baselineHrv.toStringAsFixed(0)} ms',
                    ),
                  ),
                  SizedBox(
                    width: itemWidth,
                    child: MetricCard(
                      title: 'Mean RR Interval',
                      value: telemetry.meanRr.toStringAsFixed(0),
                      unit: 'ms',
                      icon: Icons.timer_outlined,
                      accentColor: SomythraTheme.cyanGlow,
                      subtitle:
                          'SDNN: ${telemetry.sdnn.toStringAsFixed(0)} ms',
                    ),
                  ),
                  SizedBox(
                    width: itemWidth,
                    child: MetricCard(
                      title: 'Autonomic Tone',
                      value: telemetry.stressIndex.toStringAsFixed(0),
                      unit: 'SI',
                      icon: Icons.psychology_outlined,
                      accentColor: telemetry.stressLevel.contains('High')
                          ? SomythraTheme.criticalRed
                          : SomythraTheme.safeGreen,
                      subtitle: telemetry.stressLevel,
                    ),
                  ),
                ],
              );
            },
          ),
          const SizedBox(height: 24),

          // Split Row: Live Pulse Waveform + CES Gauge
          LayoutBuilder(
            builder: (context, constraints) {
              final isWide = constraints.maxWidth > 800;
              if (isWide) {
                return Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      flex: 6,
                      child: LivePulseChart(
                        values: _hrHistory.isNotEmpty
                            ? _hrHistory
                            : [68.0, 70.0, 71.0, 69.0, 72.0],
                        label: 'Continuous Heart Rate Waveform (BPM)',
                        minVal: 45.0,
                        maxVal: 150.0,
                        lineColor: SomythraTheme.heartRose,
                      ),
                    ),
                    const SizedBox(width: 20),
                    Expanded(
                      flex: 4,
                      child: CesGauge(
                        score: telemetry.cesScore,
                        rating: telemetry.cesRating,
                      ),
                    ),
                  ],
                );
              } else {
                return Column(
                  children: [
                    LivePulseChart(
                      values: _hrHistory.isNotEmpty
                          ? _hrHistory
                          : [68.0, 70.0, 71.0, 69.0, 72.0],
                      label: 'Continuous Heart Rate Waveform (BPM)',
                      minVal: 45.0,
                      maxVal: 150.0,
                      lineColor: SomythraTheme.heartRose,
                    ),
                    const SizedBox(height: 20),
                    CesGauge(
                      score: telemetry.cesScore,
                      rating: telemetry.cesRating,
                    ),
                  ],
                );
              }
            },
          ),
        ],
      ),
    );
  }
}
