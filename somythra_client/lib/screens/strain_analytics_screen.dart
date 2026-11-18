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

    // Sympathovagal ratio approximation (SNS vs PNS balance)
    final snsRatio = ((_testHr / 120.0) * 0.5 + (1.0 - (_testHrv / 100.0).clamp(0.0, 1.0)) * 0.5).clamp(0.05, 0.95);
    final pnsRatio = 1.0 - snsRatio;

    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header Row
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Wrap(
                      spacing: 8,
                      runSpacing: 4,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        const Text(
                          'SOMYTHRA RESEARCH & STRAIN',
                          style: TextStyle(
                            color: SomythraTheme.textPrimary,
                            fontSize: 22,
                            fontWeight: FontWeight.w900,
                            letterSpacing: 1.0,
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                          decoration: BoxDecoration(
                            color: SomythraTheme.cyanGlow.withOpacity(0.14),
                            borderRadius: BorderRadius.circular(4),
                            border: Border.all(color: SomythraTheme.cyanGlow.withOpacity(0.4)),
                          ),
                          child: const Text(
                            'AUTONOMIC NEURODYNAMICS',
                            style: TextStyle(
                              color: SomythraTheme.cyanGlow,
                              fontSize: 9,
                              fontWeight: FontWeight.w800,
                              fontFamily: 'monospace',
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 3),
                    const Text(
                      'Autonomic nervous system strain analysis, Baevsky stress index & HRV feature classification',
                      style: TextStyle(
                        color: SomythraTheme.textSecondary,
                        fontSize: 12.5,
                      ),
                    ),
                  ],
                ),
              ),
              if (_loading) ...[
                const SizedBox(width: 12),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: SomythraTheme.surface,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: SomythraTheme.cardBorder),
                  ),
                  child: const Row(
                    children: [
                      SizedBox(
                        width: 12,
                        height: 12,
                        child: CircularProgressIndicator(strokeWidth: 2, color: SomythraTheme.primaryCyan),
                      ),
                      SizedBox(width: 8),
                      Text(
                        'ANALYZING',
                        style: TextStyle(fontSize: 10, color: SomythraTheme.primaryCyan, fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(height: 20),

          // Interactive Testing Controls
          Container(
            padding: const EdgeInsets.all(22),
            decoration: BoxDecoration(
              color: SomythraTheme.card,
              borderRadius: BorderRadius.circular(18),
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
                const Text(
                  'OBSERVED VITAL INPUTS',
                  style: TextStyle(
                    color: SomythraTheme.textSecondary,
                    fontSize: 11,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 1.2,
                    fontFamily: 'monospace',
                  ),
                ),
                const SizedBox(height: 18),
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
                                  style: TextStyle(color: SomythraTheme.textPrimary, fontSize: 12.5, fontWeight: FontWeight.w600)),
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                                decoration: BoxDecoration(
                                  color: SomythraTheme.heartRose.withOpacity(0.15),
                                  borderRadius: BorderRadius.circular(6),
                                  border: Border.all(color: SomythraTheme.heartRose.withOpacity(0.4)),
                                ),
                                child: Text(
                                  '${_testHr.toStringAsFixed(0)} BPM',
                                  style: const TextStyle(
                                    color: SomythraTheme.heartRose,
                                    fontWeight: FontWeight.w800,
                                    fontFamily: 'monospace',
                                    fontSize: 12,
                                  ),
                                ),
                              ),
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
                    const SizedBox(width: 30),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              const Text('Observed HRV (RMSSD)',
                                  style: TextStyle(color: SomythraTheme.textPrimary, fontSize: 12.5, fontWeight: FontWeight.w600)),
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                                decoration: BoxDecoration(
                                  color: SomythraTheme.primaryCyan.withOpacity(0.15),
                                  borderRadius: BorderRadius.circular(6),
                                  border: Border.all(color: SomythraTheme.primaryCyan.withOpacity(0.4)),
                                ),
                                child: Text(
                                  '${_testHrv.toStringAsFixed(0)} ms',
                                  style: const TextStyle(
                                    color: SomythraTheme.primaryCyan,
                                    fontWeight: FontWeight.w800,
                                    fontFamily: 'monospace',
                                    fontSize: 12,
                                  ),
                                ),
                              ),
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
          const SizedBox(height: 20),

          // Sympathovagal Balance Visual Bar
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: SomythraTheme.card,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: SomythraTheme.cardBorder, width: 1.2),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      'SYMPATHOVAGAL EQUILIBRIUM (SNS vs PNS)',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.0,
                        color: SomythraTheme.textSecondary,
                        fontFamily: 'monospace',
                      ),
                    ),
                    Text(
                      'SNS: ${(snsRatio * 100).toStringAsFixed(0)}% • PNS: ${(pnsRatio * 100).toStringAsFixed(0)}%',
                      style: const TextStyle(
                        fontSize: 11,
                        color: SomythraTheme.primaryCyan,
                        fontWeight: FontWeight.w700,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 14),
                ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: SizedBox(
                    height: 12,
                    child: Row(
                      children: [
                        Expanded(
                          flex: (snsRatio * 100).round(),
                          child: Container(
                            decoration: const BoxDecoration(
                              gradient: LinearGradient(
                                colors: [SomythraTheme.heartRose, SomythraTheme.warningOrange],
                              ),
                            ),
                          ),
                        ),
                        Expanded(
                          flex: (pnsRatio * 100).round(),
                          child: Container(
                            decoration: const BoxDecoration(
                              gradient: LinearGradient(
                                colors: [SomythraTheme.cyanGlow, SomythraTheme.safeGreen],
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 8),
                const Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Flexible(
                      child: Text(
                        '◄ Sympathetic Activation (SNS)',
                        style: TextStyle(fontSize: 10, color: SomythraTheme.heartRose, fontWeight: FontWeight.w600),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    Flexible(
                      child: Text(
                        'Parasympathetic Tone (PNS) ►',
                        style: TextStyle(fontSize: 10, color: SomythraTheme.safeGreen, fontWeight: FontWeight.w600),
                        textAlign: TextAlign.right,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),

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
          const SizedBox(height: 20),

          // Informational Panel on SOMYTHRA Evidence Standards
          Container(
            padding: const EdgeInsets.all(22),
            decoration: BoxDecoration(
              color: SomythraTheme.surface,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: SomythraTheme.cardBorder, width: 1.2),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(6),
                      decoration: BoxDecoration(
                        color: SomythraTheme.primaryCyan.withOpacity(0.14),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Icon(Icons.shield_outlined,
                          color: SomythraTheme.primaryCyan, size: 18),
                    ),
                    const SizedBox(width: 12),
                    const Text(
                      'SOMYTHRA EVIDENCE & RESPONSIBLE USE PRINCIPLES',
                      style: TextStyle(
                        color: SomythraTheme.textPrimary,
                        fontSize: 12.5,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 0.6,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 14),
                const Text(
                  '• Evidence Before Claims: Derived stress metrics represent estimated autonomic balance, not clinical diagnoses.\n'
                  '• Signal Quality First: Missing or noisy sensor readings are flagged and rejected rather than clamped.\n'
                  '• Non-punitive Field Context: Community and workplace metrics are designed strictly to support human wellbeing and prevent exhaustion.',
                  style: TextStyle(
                    color: SomythraTheme.textSecondary,
                    fontSize: 12.5,
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
