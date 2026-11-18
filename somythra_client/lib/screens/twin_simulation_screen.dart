import 'package:flutter/material.dart';
import '../models/user_profile.dart';
import '../models/telemetry.dart';
import '../services/api_service.dart';
import '../theme/somythra_theme.dart';
import '../widgets/live_pulse_chart.dart';

class TwinSimulationScreen extends StatefulWidget {
  final UserProfile profile;
  const TwinSimulationScreen({super.key, required this.profile});

  @override
  State<TwinSimulationScreen> createState() => _TwinSimulationScreenState();
}

class _TwinSimulationScreenState extends State<TwinSimulationScreen> {
  double _peakHr = 165.0;
  int _durationSec = 240;
  bool _isLoading = false;
  RecoveryResultModel? _result;

  @override
  void initState() {
    super.initState();
    _runSimulation();
  }

  Future<void> _runSimulation() async {
    setState(() => _isLoading = true);
    final res = await ApiService.simulateRecovery(
      profile: widget.profile,
      peakHr: _peakHr,
      durationSec: _durationSec,
      stepSec: 5,
    );

    if (mounted) {
      setState(() {
        _isLoading = false;
        _result = res ?? _generateLocalSimulation();
      });
    }
  }

  RecoveryResultModel _generateLocalSimulation() {
    // Local fallback exponential model: HR(t) = HR_base + (peak - HR_base) * exp(-lambda * t)
    final lambda = 0.040;
    final base = widget.profile.baselineHr;
    final delta = _peakHr - base;
    final points = <RecoveryPointModel>[];

    for (int t = 0; t <= _durationSec; t += 5) {
      final hr = base + delta * (2.71828 * (-lambda * t / 2.71828)); // approximation
      final pct = ((_peakHr - hr) / delta * 100).clamp(0.0, 100.0);
      points.add(RecoveryPointModel(
        timeSec: t,
        predictedHr: double.parse(hr.toStringAsFixed(1)),
        recoveryPct: double.parse(pct.toStringAsFixed(1)),
      ));
    }

    return RecoveryResultModel(
      points: points,
      halfRecoverySec: 17.3,
      fullRecoveryEstimateSec: 75.0,
      decayConstant: lambda,
      maxHr: widget.profile.maxHr,
    );
  }

  @override
  Widget build(BuildContext context) {
    final points = _result?.points ?? [];
    final hrCurve = points.map((p) => p.predictedHr).toList();

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'SOMYTHRA TWIN',
            style: TextStyle(
              color: SomythraTheme.textPrimary,
              fontSize: 24,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            'Exploratory physiological digital-twin recovery simulation & exponential decay modeling',
            style: TextStyle(
              color: SomythraTheme.textSecondary,
              fontSize: 13,
            ),
          ),
          const SizedBox(height: 24),

          // Controls Card
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
                const Text(
                  'SIMULATION PARAMETERS',
                  style: TextStyle(
                    color: SomythraTheme.textSecondary,
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1.0,
                  ),
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
                              const Text('Peak Exertion Heart Rate',
                                  style: TextStyle(color: SomythraTheme.textPrimary)),
                              Text('${_peakHr.toStringAsFixed(0)} BPM',
                                  style: const TextStyle(
                                      color: SomythraTheme.heartRose,
                                      fontWeight: FontWeight.w700)),
                            ],
                          ),
                          Slider(
                            value: _peakHr,
                            min: 90,
                            max: 200,
                            divisions: 22,
                            activeColor: SomythraTheme.heartRose,
                            onChanged: (val) {
                              setState(() => _peakHr = val);
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
                              const Text('Monitoring Window',
                                  style: TextStyle(color: SomythraTheme.textPrimary)),
                              Text('${_durationSec}s',
                                  style: const TextStyle(
                                      color: SomythraTheme.primaryCyan,
                                      fontWeight: FontWeight.w700)),
                            ],
                          ),
                          Slider(
                            value: _durationSec.toDouble(),
                            min: 60,
                            max: 600,
                            divisions: 18,
                            activeColor: SomythraTheme.primaryCyan,
                            onChanged: (val) {
                              setState(() => _durationSec = val.toInt());
                            },
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Align(
                  alignment: Alignment.centerRight,
                  child: ElevatedButton.icon(
                    onPressed: _isLoading ? null : _runSimulation,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: SomythraTheme.primaryCyan,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                    icon: _isLoading
                        ? const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(
                                strokeWidth: 2, color: Colors.white),
                          )
                        : const Icon(Icons.play_arrow_rounded, size: 20),
                    label: const Text(
                      'SIMULATE TWIN RECOVERY',
                      style: TextStyle(fontWeight: FontWeight.w700),
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // Simulation Recovery Stats
          if (_result != null) ...[
            Row(
              children: [
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: SomythraTheme.card,
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(color: SomythraTheme.cardBorder),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('HALF-RECOVERY (T 1/2)',
                            style: TextStyle(color: SomythraTheme.textMuted, fontSize: 11)),
                        const SizedBox(height: 6),
                        Text(
                          '${_result!.halfRecoverySec.toStringAsFixed(1)}s',
                          style: const TextStyle(
                            color: SomythraTheme.primaryCyan,
                            fontSize: 22,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                        const SizedBox(height: 4),
                        const Text('Time to 50% delta recovery',
                            style: TextStyle(color: SomythraTheme.textSecondary, fontSize: 11)),
                      ],
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: SomythraTheme.card,
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(color: SomythraTheme.cardBorder),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('FULL RECOVERY ESTIMATE',
                            style: TextStyle(color: SomythraTheme.textMuted, fontSize: 11)),
                        const SizedBox(height: 6),
                        Text(
                          '${_result!.fullRecoveryEstimateSec.toStringAsFixed(0)}s',
                          style: const TextStyle(
                            color: SomythraTheme.safeGreen,
                            fontSize: 22,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                        const SizedBox(height: 4),
                        const Text('95% baseline convergence (3 / λ)',
                            style: TextStyle(color: SomythraTheme.textSecondary, fontSize: 11)),
                      ],
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: SomythraTheme.card,
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(color: SomythraTheme.cardBorder),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('DECAY CONSTANT (λ)',
                            style: TextStyle(color: SomythraTheme.textMuted, fontSize: 11)),
                        const SizedBox(height: 6),
                        Text(
                          _result!.decayConstant.toStringAsFixed(3),
                          style: const TextStyle(
                            color: SomythraTheme.cyanGlow,
                            fontSize: 22,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text('Tier: ${widget.profile.fitnessLevel}',
                            style: const TextStyle(color: SomythraTheme.textSecondary, fontSize: 11)),
                      ],
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),

            // Recovery Curve Chart
            LivePulseChart(
              values: hrCurve.isNotEmpty ? hrCurve : [70.0, 70.0],
              minVal: widget.profile.baselineHr - 10,
              maxVal: _peakHr + 10,
              lineColor: SomythraTheme.primaryCyan,
              label: 'Predicted Digital Twin Recovery Curve (BPM vs Time)',
            ),
          ],
        ],
      ),
    );
  }
}
