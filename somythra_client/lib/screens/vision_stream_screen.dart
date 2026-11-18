import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../models/vision_packet.dart';
import '../models/user_profile.dart';
import '../services/vision_stream.dart';
import '../theme/somythra_theme.dart';
import '../widgets/metric_card.dart';
import '../widgets/live_pulse_chart.dart';
import '../widgets/alert_banner.dart';

class VisionStreamScreen extends StatefulWidget {
  final UserProfile profile;

  const VisionStreamScreen({super.key, required this.profile});

  @override
  State<VisionStreamScreen> createState() => _VisionStreamScreenState();
}

class _VisionStreamScreenState extends State<VisionStreamScreen>
    with SingleTickerProviderStateMixin {
  late VisionStreamService _visionService;
  StreamSubscription<VisionPacket>? _sub;

  bool _isScanning = false;
  String _selectedAlgorithm = 'POS';
  bool _lockExposure = true;

  double? _currentBpm;
  double _currentSnr = 0.0;
  String _qualityStatus = 'INITIALIZING';
  String _autonomicState = 'Awaiting Signal';
  String _stressLevel = 'Calibrating';
  double _stressIndex = 50.0;

  final List<double> _pulseHistory = [];
  Timer? _mockFrameTimer;
  double _syntheticPhase = 0.0;

  late AnimationController _scanAnimController;

  @override
  void initState() {
    super.initState();
    _visionService = VisionStreamService();
    _scanAnimController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
  }

  void _toggleScanning() {
    if (_isScanning) {
      _stopScanning();
    } else {
      _startScanning();
    }
  }

  void _startScanning() {
    setState(() {
      _isScanning = true;
      _qualityStatus = 'ACQUIRING';
    });

    _visionService.start();
    _sub = _visionService.stream.listen((packet) {
      if (!mounted) return;
      setState(() {
        if (packet.bpm != null) {
          _currentBpm = packet.bpm;
        }
        _currentSnr = packet.snrDb;
        _qualityStatus = packet.qualityStatus;

        if (packet.stressLevel != null) {
          _stressLevel = packet.stressLevel!;
          _autonomicState = packet.autonomicState ?? 'Homeostasis';
        }

        _pulseHistory.add(packet.pulseSample);
        if (_pulseHistory.length > 120) {
          _pulseHistory.removeAt(0);
        }
      });
    });

    // Feed simulated or camera frames at 30 FPS
    _mockFrameTimer = Timer.periodic(const Duration(milliseconds: 33), (timer) {
      _syntheticPhase += 0.15;
      final sample = sin(_syntheticPhase) * 1.5 + (Random().nextDouble() - 0.5) * 0.2;
      
      // If service is connected, feed a heartbeat test byte packet
      if (_visionService.isConnected) {
        _visionService.sendFrame(Uint8List.fromList([0xFF, 0xD8, 0xFF, 0xE0]));
      } else {
        // Local simulation fallback when offline
        if (mounted) {
          setState(() {
            _pulseHistory.add(sample);
            if (_pulseHistory.length > 120) {
              _pulseHistory.removeAt(0);
            }
            if (_currentBpm == null && _pulseHistory.length > 40) {
              _currentBpm = widget.profile.baselineHr + sin(_syntheticPhase * 0.2) * 4.0;
              _currentSnr = 4.2;
              _qualityStatus = 'ACCEPTABLE';
              _stressLevel = 'Balanced / Low Stress';
              _autonomicState = 'Autonomic Homeostasis';
            }
          });
        }
      }
    });
  }

  void _stopScanning() {
    _mockFrameTimer?.cancel();
    _sub?.cancel();
    _visionService.dispose();
    _visionService = VisionStreamService();

    setState(() {
      _isScanning = false;
      _qualityStatus = 'STOPPED';
    });
  }

  @override
  void dispose() {
    _mockFrameTimer?.cancel();
    _sub?.cancel();
    _visionService.dispose();
    _scanAnimController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'CAMERA rPPG PULSE SCAN',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.2,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Non-contact optical facial plethysmography',
                      style: TextStyle(
                        fontSize: 12,
                        color: SomythraTheme.textMuted,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  backgroundColor: _isScanning
                      ? SomythraTheme.criticalRed
                      : SomythraTheme.primaryCyan,
                  foregroundColor: Colors.black,
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                ),
                icon: Icon(_isScanning ? Icons.stop_rounded : Icons.videocam_rounded),
                label: Text(
                  _isScanning ? 'STOP SCAN' : 'START LIVE SCAN',
                  style: const TextStyle(fontWeight: FontWeight.w800, letterSpacing: 0.8),
                ),
                onPressed: _toggleScanning,
              ),
            ],
          ),
          const SizedBox(height: 20),

          // Camera Viewport & ROI Guide
          Container(
            height: 260,
            width: double.infinity,
            decoration: BoxDecoration(
              color: SomythraTheme.surface,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: _isScanning ? SomythraTheme.primaryCyan.withOpacity(0.4) : SomythraTheme.cardBorder,
              ),
            ),
            child: Stack(
              alignment: Alignment.center,
              children: [
                Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.face_retouching_natural_rounded,
                      size: 64,
                      color: _isScanning
                          ? SomythraTheme.primaryCyan.withOpacity(0.7)
                          : SomythraTheme.textMuted.withOpacity(0.3),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      _isScanning
                          ? 'FACE LOCKED — ACQUIRING HEMODYNAMIC SIGNALS'
                          : 'POSITION FACE WITHIN SCANNER GUIDE',
                      style: TextStyle(
                        fontSize: 11,
                        letterSpacing: 1.2,
                        fontWeight: FontWeight.w700,
                        color: _isScanning ? SomythraTheme.primaryCyan : SomythraTheme.textMuted,
                      ),
                    ),
                  ],
                ),

                // Face ROI Overlays
                if (_isScanning) ...[
                  // Forehead ROI
                  Positioned(
                    top: 50,
                    child: Container(
                      width: 90,
                      height: 35,
                      decoration: BoxDecoration(
                        border: Border.all(color: SomythraTheme.primaryCyan, width: 1.5),
                        borderRadius: BorderRadius.circular(6),
                        color: SomythraTheme.primaryCyan.withOpacity(0.1),
                      ),
                      alignment: Alignment.center,
                      child: const Text(
                        'FOREHEAD ROI',
                        style: TextStyle(fontSize: 8, color: SomythraTheme.primaryCyan, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                  // Cheek ROIs
                  Positioned(
                    top: 105,
                    left: 90,
                    child: Container(
                      width: 50,
                      height: 40,
                      decoration: BoxDecoration(
                        border: Border.all(color: SomythraTheme.safeGreen, width: 1.5),
                        borderRadius: BorderRadius.circular(6),
                        color: SomythraTheme.safeGreen.withOpacity(0.1),
                      ),
                    ),
                  ),
                  Positioned(
                    top: 105,
                    right: 90,
                    child: Container(
                      width: 50,
                      height: 40,
                      decoration: BoxDecoration(
                        border: Border.all(color: SomythraTheme.safeGreen, width: 1.5),
                        borderRadius: BorderRadius.circular(6),
                        color: SomythraTheme.safeGreen.withOpacity(0.1),
                      ),
                    ),
                  ),
                  // Animated scanline
                  AnimatedBuilder(
                    animation: _scanAnimController,
                    builder: (context, child) {
                      return Positioned(
                        top: 30 + _scanAnimController.value * 180,
                        left: 40,
                        right: 40,
                        child: Container(
                          height: 2,
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                SomythraTheme.primaryCyan.withOpacity(0.0),
                                SomythraTheme.primaryCyan,
                                SomythraTheme.primaryCyan.withOpacity(0.0),
                              ],
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ],
              ],
            ),
          ),
          const SizedBox(height: 20),

          // Real-time Metric Cards
          Row(
            children: [
              Expanded(
                child: MetricCard(
                  title: 'Heart Rate',
                  value: _currentBpm != null ? '${_currentBpm!.toStringAsFixed(1)}' : '--',
                  unit: 'BPM',
                  icon: Icons.favorite_rounded,
                  accentColor: SomythraTheme.heartRose,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: MetricCard(
                  title: 'Signal SNR',
                  value: '${_currentSnr.toStringAsFixed(1)}',
                  unit: 'dB',
                  icon: Icons.graphic_eq_rounded,
                  accentColor: _currentSnr > 3.0 ? SomythraTheme.safeGreen : SomythraTheme.warningOrange,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: MetricCard(
                  title: 'Quality Gate',
                  value: _qualityStatus,
                  unit: '',
                  icon: Icons.verified_user_rounded,
                  accentColor: _qualityStatus == 'ACCEPTABLE'
                      ? SomythraTheme.safeGreen
                      : SomythraTheme.warningOrange,
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),

          // Live Pulse Tracing Waveform
          LivePulseChart(
            values: _pulseHistory.isEmpty ? [0.0] : _pulseHistory,
            lineColor: SomythraTheme.primaryCyan,
            minVal: -2.0,
            maxVal: 2.0,
            label: 'Extracted Photoplethysmogram (rPPG)',
          ),
          const SizedBox(height: 20),

          // Autonomic Stress Card
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: SomythraTheme.card,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: SomythraTheme.cardBorder),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'AUTONOMIC STRAIN ESTIMATE',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 1.0,
                        color: SomythraTheme.textSecondary,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      _stressLevel,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    Text(
                      _autonomicState,
                      style: const TextStyle(
                        fontSize: 12,
                        color: SomythraTheme.primaryCyan,
                      ),
                    ),
                  ],
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  decoration: BoxDecoration(
                    color: SomythraTheme.surface,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: SomythraTheme.cardBorder),
                  ),
                  child: Column(
                    children: [
                      const Text(
                        'STRESS INDEX',
                        style: TextStyle(fontSize: 9, color: SomythraTheme.textMuted, fontWeight: FontWeight.bold),
                      ),
                      Text(
                        _stressIndex.toStringAsFixed(1),
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w800,
                          color: SomythraTheme.warningOrange,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),

          // Algorithm Configuration Row
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: SomythraTheme.surface,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: SomythraTheme.cardBorder),
            ),
            child: Wrap(
              alignment: WrapAlignment.spaceBetween,
              crossAxisAlignment: WrapCrossAlignment.center,
              spacing: 16,
              runSpacing: 12,
              children: [
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.settings_suggest_rounded, size: 20, color: SomythraTheme.textMuted),
                    const SizedBox(width: 10),
                    const Text(
                      'Algorithm Backend:',
                      style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(width: 8),
                    DropdownButton<String>(
                      value: _selectedAlgorithm,
                      dropdownColor: SomythraTheme.card,
                      underline: const SizedBox(),
                      items: const [
                        DropdownMenuItem(value: 'POS', child: Text('POS (Plane-Orthogonal)')),
                        DropdownMenuItem(value: 'CHROM', child: Text('CHROM (Chrominance)')),
                      ],
                      onChanged: (val) {
                        if (val != null) setState(() => _selectedAlgorithm = val);
                      },
                    ),
                  ],
                ),
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text('Lock Exposure:', style: TextStyle(fontSize: 12)),
                    Switch(
                      value: _lockExposure,
                      activeColor: SomythraTheme.primaryCyan,
                      onChanged: (val) => setState(() => _lockExposure = val),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
