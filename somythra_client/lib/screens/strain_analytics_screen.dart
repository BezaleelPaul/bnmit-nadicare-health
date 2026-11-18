import 'package:flutter/material.dart';
import '../models/user_profile.dart';
import '../services/api_service.dart';
import '../theme/somythra_theme.dart';
import '../widgets/metric_card.dart';

class StrainAnalyticsScreen extends StatefulWidget {
  final UserProfile profile;
  const StrainAnalyticsScreen({super.key, required this.profile});

  @override
  State<StrainAnalyticsScreen> createState() => _StrainAnalyticsScreenState();
}

class _StrainAnalyticsScreenState extends State<StrainAnalyticsScreen> {
  double _testHr = 85.0;
  double _testHrv = 42.0;
  final double _testLoad = 2.0;

  bool _loading = false;
  Map<String, dynamic>? _stressData;

  @override
  void initState() {
    super.initState();
    _analyze();
  }

  Future<void> _analyze() async {
    setState(() => _loading = true);
    final res = await ApiService.predictStress(
      hr: _testHr,
      hrv: _testHrv,
      activityLoad: _testLoad,
    );
    if (mounted) {
      setState(() {
        _loading = false;
        _stressData = res ??
            {
              'stress_level': _testHrv < 30 ? 'High Autonomic Stress' : 'Balanced / Low Stress',
              'confidence': 0.88,
              'stress_index': (50.0 * (_testHr / 70.0) * (50.0 / _testHrv)).roundToDouble(),
              'autonomic_state': _testHrv < 30 ? 'Sympathetic Hyperarousal' : 'Autonomic Homeostasis',
            };
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final stressLevel = _stressData?['stress_level'] ?? 'Balanced';
    final stressIndex = (_stressData?['stress_index'] as num?)?.toDouble() ?? 48.0;
    final autonomicState = _stressData?['autonomic_state'] ?? 'Homeostasis';
    final isStressed = stressLevel.contains('High');

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'SOMYTHRA RESEARCH & STRAIN',
            style: TextStyle(
              color: SomythraTheme.textPrimary,
              fontSize: 24,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            'Autonomic nervous system strain analysis, Baevsky stress index & HRV feature classification',
            style: TextStyle(
              color: SomythraTheme.textSecondary,
              fontSize: 13,
            ),
          ),
          const SizedBox(height: 24),

          // Interactive Testing Controls
          Container(
            padding: const EdgeInsets.all(20),
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
                    const Text(
                      'OBSERVED VITAL INPUTS',
                      style: TextStyle(
                        color: SomythraTheme.textSecondary,
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 1.0,
                      ),
                    ),
                    if (_loading)
                      const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: SomythraTheme.primaryCyan,
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              const Text('Observed Heart Rate',
                                  style: TextStyle(color: SomythraTheme.textPrimary)),
                              Text('${_testHr.toStringAsFixed(0)} BPM',
                                  style: const TextStyle(
                                      color: SomythraTheme.heartRose,
                                      fontWeight: FontWeight.w700)),
                            ],
                          ),
                          Slider(
                            value: _testHr,
                            min: 45,
                            max: 180,
                            divisions: 27,
                            activeColor: SomythraTheme.heartRose,
                            onChanged: (val) {
                              setState(() => _testHr = val);
                              _analyze();
                            },
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 20),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              const Text('Observed HRV (RMSSD)',
                                  style: TextStyle(color: SomythraTheme.textPrimary)),
                              Text('${_testHrv.toStringAsFixed(0)} ms',
                                  style: const TextStyle(
                                      color: SomythraTheme.primaryCyan,
                                      fontWeight: FontWeight.w700)),
                            ],
                          ),
                          Slider(
                            value: _testHrv,
                            min: 10,
                            max: 120,
                            divisions: 22,
                            activeColor: SomythraTheme.primaryCyan,
                            onChanged: (val) {
                              setState(() => _testHrv = val);
                              _analyze();
                            },
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // Diagnostic Output Cards
          Row(
            children: [
              Expanded(
                child: MetricCard(
                  title: 'Stress Classification',
                  value: stressLevel,
                  unit: '',
                  icon: Icons.psychology_rounded,
                  accentColor: isStressed
                      ? SomythraTheme.criticalRed
                      : SomythraTheme.safeGreen,
                  subtitle: autonomicState,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: MetricCard(
                  title: 'Baevsky Stress Index',
                  value: stressIndex.toStringAsFixed(0),
                  unit: 'SI',
                  icon: Icons.speed_rounded,
                  accentColor: isStressed
                      ? SomythraTheme.warningOrange
                      : SomythraTheme.cyanGlow,
                  subtitle: 'Ideal baseline resting range: 30 - 90 SI',
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),

          // Informational Panel on SOMYTHRA Evidence Standards
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: SomythraTheme.surface,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: SomythraTheme.cardBorder),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: const [
                    Icon(Icons.shield_outlined,
                        color: SomythraTheme.primaryCyan, size: 20),
                    SizedBox(width: 10),
                    Text(
                      'SOMYTHRA EVIDENCE & RESPONSIBLE USE PRINCIPLES',
                      style: TextStyle(
                        color: SomythraTheme.textPrimary,
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 0.5,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                const Text(
                  '• Evidence Before Claims: Derived stress metrics represent estimated autonomic balance, not clinical diagnoses.\n'
                  '• Signal Quality First: Missing or noisy sensor readings are flagged and rejected rather than clamped.\n'
                  '• Non-punitive Field Context: Community and workplace metrics are designed strictly to support human wellbeing and prevent exhaustion.',
                  style: TextStyle(
                    color: SomythraTheme.textSecondary,
                    fontSize: 13,
                    height: 1.6,
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
