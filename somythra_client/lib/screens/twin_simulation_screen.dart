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
    final lambda = 0.040;
    final base = widget.profile.baselineHr;
    final delta = _peakHr - base;
    final points = <RecoveryPointModel>[];

    for (int t = 0; t <= _durationSec; t += 5) {
      final decay = (2.718281828 * (-lambda * t / 2.718281828));
      final hr = base + delta * decay.clamp(0.0, 1.0);
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
                          'SOMYTHRA TWIN',
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
                            color: SomythraTheme.neuralViolet.withOpacity(0.14),
                            borderRadius: BorderRadius.circular(4),
                            border: Border.all(color: SomythraTheme.neuralViolet.withOpacity(0.4)),
                          ),
                          child: const Text(
                            'IN SILICO MODELING',
                            style: TextStyle(
                              color: SomythraTheme.neuralViolet,
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
                      'Exploratory physiological digital-twin recovery simulation & exponential decay dynamics',
                      style: TextStyle(
                        color: SomythraTheme.textSecondary,
                        fontSize: 12.5,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 12),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: SomythraTheme.surface,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: SomythraTheme.cardBorder),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.fitness_center_rounded, size: 14, color: SomythraTheme.primaryCyan),
                    const SizedBox(width: 6),
                    Text(
                      'Fitness Tier: ${widget.profile.fitnessLevel.toUpperCase()}',
                      style: const TextStyle(
                        fontSize: 10.5,
                        fontWeight: FontWeight.w700,
                        color: SomythraTheme.textPrimary,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),

          // Controls Card (Simulation Console)
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
                Wrap(
                  alignment: WrapAlignment.spaceBetween,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  spacing: 12,
                  runSpacing: 6,
                  children: const [
                    Text(
                      'SIMULATION PARAMETERS',
                      style: TextStyle(
                        color: SomythraTheme.textSecondary,
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.2,
                        fontFamily: 'monospace',
                      ),
                    ),
                    Text(
                      'EXPONENTIAL DECAY ODE: dHR/dt = -λ(HR - HR_base)',
                      style: TextStyle(
                        color: SomythraTheme.textMuted,
                        fontSize: 10,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 18),
                LayoutBuilder(
                  builder: (context, constraints) {
                    final isWide = constraints.maxWidth > 650;
                    if (isWide) {
                      return Row(
                        children: [
                          Expanded(child: _buildPeakHrSlider()),
                          const SizedBox(width: 32),
                          Expanded(child: _buildDurationSlider()),
                        ],
                      );
                    } else {
                      return Column(
                        children: [
                          _buildPeakHrSlider(),
                          const SizedBox(height: 16),
                          _buildDurationSlider(),
                        ],
                      );
                    }
                  },
                ),
                const SizedBox(height: 16),
                Align(
                  alignment: Alignment.centerRight,
                  child: ElevatedButton.icon(
                    onPressed: _isLoading ? null : _runSimulation,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: SomythraTheme.primaryCyan,
                      foregroundColor: Colors.black,
                      elevation: 0,
                      padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    icon: _isLoading
                        ? const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.black,
                            ),
                          )
                        : const Icon(Icons.play_arrow_rounded, size: 20),
                    label: const Text(
                      'SIMULATE TWIN RECOVERY',
                      style: TextStyle(fontWeight: FontWeight.w900, letterSpacing: 0.8),
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),

          // Simulation Recovery Stats
          if (_result != null) ...[
            Row(
              children: [
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: SomythraTheme.card,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: SomythraTheme.primaryCyan.withOpacity(0.3), width: 1.2),
                      boxShadow: [
                        BoxShadow(
                          color: SomythraTheme.primaryCyan.withOpacity(0.06),
                          blurRadius: 14,
                        ),
                      ],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'HALF-RECOVERY (T 1/2)',
                          style: TextStyle(
                            color: SomythraTheme.textMuted,
                            fontSize: 10,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 0.8,
                            fontFamily: 'monospace',
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          '${_result!.halfRecoverySec.toStringAsFixed(1)}s',
                          style: const TextStyle(
                            color: SomythraTheme.primaryCyan,
                            fontSize: 26,
                            fontWeight: FontWeight.w900,
                            fontFamily: 'monospace',
                          ),
                        ),
                        const SizedBox(height: 4),
                        const Text(
                          'Time to 50% delta recovery',
                          style: TextStyle(color: SomythraTheme.textSecondary, fontSize: 11),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: SomythraTheme.card,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: SomythraTheme.safeGreen.withOpacity(0.3), width: 1.2),
                      boxShadow: [
                        BoxShadow(
                          color: SomythraTheme.safeGreen.withOpacity(0.06),
                          blurRadius: 14,
                        ),
                      ],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'FULL RECOVERY ESTIMATE',
                          style: TextStyle(
                            color: SomythraTheme.textMuted,
                            fontSize: 10,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 0.8,
                            fontFamily: 'monospace',
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          '${_result!.fullRecoveryEstimateSec.toStringAsFixed(0)}s',
                          style: const TextStyle(
                            color: SomythraTheme.safeGreen,
                            fontSize: 26,
                            fontWeight: FontWeight.w900,
                            fontFamily: 'monospace',
                          ),
                        ),
                        const SizedBox(height: 4),
                        const Text(
                          '95% baseline convergence (3 / λ)',
                          style: TextStyle(color: SomythraTheme.textSecondary, fontSize: 11),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: SomythraTheme.card,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: SomythraTheme.neuralViolet.withOpacity(0.3), width: 1.2),
                      boxShadow: [
                        BoxShadow(
                          color: SomythraTheme.neuralViolet.withOpacity(0.06),
                          blurRadius: 14,
                        ),
                      ],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'DECAY CONSTANT (λ)',
                          style: TextStyle(
                            color: SomythraTheme.textMuted,
                            fontSize: 10,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 0.8,
                            fontFamily: 'monospace',
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          _result!.decayConstant.toStringAsFixed(3),
                          style: const TextStyle(
                            color: SomythraTheme.neuralViolet,
                            fontSize: 26,
                            fontWeight: FontWeight.w900,
                            fontFamily: 'monospace',
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Tanaka Max: ${widget.profile.maxHr.toStringAsFixed(0)} BPM',
                          style: const TextStyle(color: SomythraTheme.textSecondary, fontSize: 11),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),

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

  Widget _buildPeakHrSlider() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Flexible(
              child: Text(
                'Peak Exertion HR',
                style: TextStyle(color: SomythraTheme.textPrimary, fontSize: 12.5, fontWeight: FontWeight.w600),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: SomythraTheme.heartRose.withOpacity(0.15),
                borderRadius: BorderRadius.circular(6),
                border: Border.all(color: SomythraTheme.heartRose.withOpacity(0.4)),
              ),
              child: Text(
                '${_peakHr.toStringAsFixed(0)} BPM',
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
    );
  }

  Widget _buildDurationSlider() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Flexible(
              child: Text(
                'Monitoring Window',
                style: TextStyle(color: SomythraTheme.textPrimary, fontSize: 12.5, fontWeight: FontWeight.w600),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: SomythraTheme.primaryCyan.withOpacity(0.15),
                borderRadius: BorderRadius.circular(6),
                border: Border.all(color: SomythraTheme.primaryCyan.withOpacity(0.4)),
              ),
              child: Text(
                '${_durationSec}s',
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
    );
  }
}
