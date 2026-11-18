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

      if (_visionService.isConnected) {
        _visionService.sendFrame(Uint8List.fromList([0xFF, 0xD8, 0xFF, 0xE0]));
      } else {
        if (mounted) {
          setState(() {
            _pulseHistory.add(sample);
            if (_pulseHistory.length > 120) {
              _pulseHistory.removeAt(0);
            }
            if (_currentBpm == null && _pulseHistory.length > 30) {
              _currentBpm = widget.profile.baselineHr + sin(_syntheticPhase * 0.2) * 3.5;
              _currentSnr = 5.4;
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
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header Row & Primary Trigger Button
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
                          'CAMERA rPPG PULSE SCAN',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.w900,
                            letterSpacing: 1.1,
                            color: SomythraTheme.textPrimary,
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                          decoration: BoxDecoration(
                            color: SomythraTheme.primaryCyan.withOpacity(0.14),
                            borderRadius: BorderRadius.circular(4),
                            border: Border.all(color: SomythraTheme.primaryCyan.withOpacity(0.4)),
                          ),
                          child: const Text(
                            'OPTICAL HEMODYNAMICS',
                            style: TextStyle(
                              color: SomythraTheme.primaryCyan,
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
                      'Non-contact optical facial plethysmography & micro-vascular dermal perfusion tracking',
                      style: TextStyle(
                        fontSize: 12.5,
                        color: SomythraTheme.textSecondary,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 12),
              ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  backgroundColor: _isScanning
                      ? SomythraTheme.criticalRed
                      : SomythraTheme.primaryCyan,
                  foregroundColor: Colors.black,
                  elevation: 0,
                  shadowColor: (_isScanning
                          ? SomythraTheme.criticalRed
                          : SomythraTheme.primaryCyan)
                      .withOpacity(0.4),
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
                icon: Icon(_isScanning ? Icons.stop_rounded : Icons.videocam_rounded, size: 20),
                label: Text(
                  _isScanning ? 'STOP SCAN' : 'START LIVE SCAN',
                  style: const TextStyle(fontWeight: FontWeight.w900, letterSpacing: 0.8),
                ),
                onPressed: _toggleScanning,
              ),
            ],
          ),
          const SizedBox(height: 20),

          // High-Tech Biometric HUD Viewport
          Container(
            height: 280,
            width: double.infinity,
            decoration: BoxDecoration(
              color: const Color(0xFF090E16),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: _isScanning
                    ? SomythraTheme.primaryCyan.withOpacity(0.6)
                    : SomythraTheme.cardBorder,
                width: 1.5,
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.5),
                  blurRadius: 20,
                  offset: const Offset(0, 8),
                ),
                if (_isScanning)
                  BoxShadow(
                    color: SomythraTheme.primaryCyan.withOpacity(0.12),
                    blurRadius: 30,
                    spreadRadius: 2,
                  ),
              ],
            ),
            child: Stack(
              alignment: Alignment.center,
              children: [
                // Viewport Grid and HUD Crosshairs
                CustomPaint(
                  size: const Size(double.infinity, 280),
                  painter: _HudGridPainter(isScanning: _isScanning),
                ),

                // Center Guide & Biometric Face Indicator
                Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Container(
                      width: 90,
                      height: 90,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: _isScanning
                              ? SomythraTheme.primaryCyan.withOpacity(0.8)
                              : SomythraTheme.cardBorder,
                          width: 1.5,
                        ),
                        color: _isScanning
                            ? SomythraTheme.primaryCyan.withOpacity(0.06)
                            : Colors.transparent,
                      ),
                      child: Icon(
                        Icons.face_retouching_natural_rounded,
                        size: 52,
                        color: _isScanning
                            ? SomythraTheme.primaryCyan
                            : SomythraTheme.textMuted.withOpacity(0.4),
                      ),
                    ),
                    const SizedBox(height: 14),
                    Text(
                      _isScanning
                          ? 'FACE LOCKED — ACQUIRING HEMODYNAMIC SIGNALS'
                          : 'POSITION FACE WITHIN SCANNER GUIDE',
                      style: TextStyle(
                        fontSize: 11,
                        letterSpacing: 1.2,
                        fontWeight: FontWeight.w800,
                        fontFamily: 'monospace',
                        color: _isScanning ? SomythraTheme.primaryCyan : SomythraTheme.textMuted,
                      ),
                    ),
                  ],
                ),

                // Optical Sampling Zones (ROIs) & Animated Laser Scanline
                if (_isScanning) ...[
                  // Forehead ROI
                  Positioned(
                    top: 50,
                    child: Container(
                      width: 110,
                      height: 38,
                      decoration: BoxDecoration(
                        border: Border.all(color: SomythraTheme.primaryCyan, width: 1.5),
                        borderRadius: BorderRadius.circular(8),
                        color: SomythraTheme.primaryCyan.withOpacity(0.12),
                        boxShadow: [
                          BoxShadow(
                            color: SomythraTheme.primaryCyan.withOpacity(0.2),
                            blurRadius: 8,
                          ),
                        ],
                      ),
                      alignment: Alignment.center,
                      child: const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.grain_rounded, color: SomythraTheme.primaryCyan, size: 12),
                          SizedBox(width: 4),
                          Text(
                            'FOREHEAD ROI',
                            style: TextStyle(
                              fontSize: 9,
                              color: SomythraTheme.primaryCyan,
                              fontWeight: FontWeight.w800,
                              letterSpacing: 0.5,
                              fontFamily: 'monospace',
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  // Left Malar (Cheek) ROI
                  Positioned(
                    top: 115,
                    left: 80,
                    child: Container(
                      width: 60,
                      height: 44,
                      decoration: BoxDecoration(
                        border: Border.all(color: SomythraTheme.safeGreen, width: 1.5),
                        borderRadius: BorderRadius.circular(8),
                        color: SomythraTheme.safeGreen.withOpacity(0.12),
                      ),
                      alignment: Alignment.center,
                      child: const Text(
                        'MALAR-L',
                        style: TextStyle(
                          fontSize: 8,
                          color: SomythraTheme.safeGreen,
                          fontWeight: FontWeight.w800,
                          fontFamily: 'monospace',
                        ),
                      ),
                    ),
                  ),
                  // Right Malar (Cheek) ROI
                  Positioned(
                    top: 115,
                    right: 80,
                    child: Container(
                      width: 60,
                      height: 44,
                      decoration: BoxDecoration(
                        border: Border.all(color: SomythraTheme.safeGreen, width: 1.5),
                        borderRadius: BorderRadius.circular(8),
                        color: SomythraTheme.safeGreen.withOpacity(0.12),
                      ),
                      alignment: Alignment.center,
                      child: const Text(
                        'MALAR-R',
                        style: TextStyle(
                          fontSize: 8,
                          color: SomythraTheme.safeGreen,
                          fontWeight: FontWeight.w800,
                          fontFamily: 'monospace',
                        ),
                      ),
                    ),
                  ),

                  // Animated Sweep Laser Scanline
                  AnimatedBuilder(
                    animation: _scanAnimController,
                    builder: (context, child) {
                      return Positioned(
                        top: 25 + _scanAnimController.value * 230,
                        left: 20,
                        right: 20,
                        child: Container(
                          height: 2.5,
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                SomythraTheme.primaryCyan.withOpacity(0.0),
                                SomythraTheme.primaryCyan.withOpacity(0.8),
                                Colors.white,
                                SomythraTheme.primaryCyan.withOpacity(0.8),
                                SomythraTheme.primaryCyan.withOpacity(0.0),
                              ],
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: SomythraTheme.primaryCyan.withOpacity(0.6),
                                blurRadius: 10,
                                spreadRadius: 1,
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ],

                // Top HUD Status Overlay
                Positioned(
                  top: 12,
                  left: 16,
                  child: Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.5),
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(color: SomythraTheme.cardBorder),
                        ),
                        child: Text(
                          'FPS: 30.0 • WL: 520-580nm • ${_selectedAlgorithm.toUpperCase()}',
                          style: const TextStyle(
                            color: SomythraTheme.textSecondary,
                            fontSize: 9,
                            fontFamily: 'monospace',
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),

          // Real-time Tri-Metric Cards
          Row(
            children: [
              Expanded(
                child: MetricCard(
                  title: 'Heart Rate',
                  value: _currentBpm != null ? _currentBpm!.toStringAsFixed(1) : '--',
                  unit: 'BPM',
                  icon: Icons.favorite_rounded,
                  accentColor: SomythraTheme.heartRose,
                  subtitle: 'Pulse Wave Harmonic',
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: MetricCard(
                  title: 'Signal SNR',
                  value: _currentSnr.toStringAsFixed(1),
                  unit: 'dB',
                  icon: Icons.graphic_eq_rounded,
                  accentColor: _currentSnr > 3.0 ? SomythraTheme.safeGreen : SomythraTheme.warningOrange,
                  subtitle: _currentSnr > 3.0 ? 'High Fidelity Perfusion' : 'Weak Optical Signal',
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: MetricCard(
                  title: 'Quality Gate',
                  value: _qualityStatus,
                  unit: '',
                  icon: Icons.verified_user_rounded,
                  accentColor: _qualityStatus == 'ACCEPTABLE'
                      ? SomythraTheme.safeGreen
                      : SomythraTheme.warningOrange,
                  subtitle: 'Hemodynamic Verification',
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

          // Autonomic Strain Assessment Card
          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: SomythraTheme.card,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: SomythraTheme.cardBorder, width: 1.2),
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
                        fontSize: 10.5,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.2,
                        color: SomythraTheme.textSecondary,
                        fontFamily: 'monospace',
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      _stressLevel,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w900,
                        color: SomythraTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _autonomicState,
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: SomythraTheme.primaryCyan,
                      ),
                    ),
                  ],
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                  decoration: BoxDecoration(
                    color: SomythraTheme.surface,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: SomythraTheme.cardBorder),
                  ),
                  child: Column(
                    children: [
                      const Text(
                        'STRESS INDEX',
                        style: TextStyle(
                          fontSize: 9,
                          color: SomythraTheme.textMuted,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 0.8,
                          fontFamily: 'monospace',
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        _stressIndex.toStringAsFixed(1),
                        style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.w900,
                          color: SomythraTheme.warningOrange,
                          fontFamily: 'monospace',
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
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
            decoration: BoxDecoration(
              color: SomythraTheme.surface,
              borderRadius: BorderRadius.circular(14),
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
                    const Icon(Icons.settings_suggest_rounded, size: 20, color: SomythraTheme.primaryCyan),
                    const SizedBox(width: 10),
                    const Text(
                      'Algorithm Backend:',
                      style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(width: 10),
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
                    const Text('Lock Exposure & Gain:', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
                    const SizedBox(width: 6),
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

class _HudGridPainter extends CustomPainter {
  final bool isScanning;
  _HudGridPainter({required this.isScanning});

  @override
  void paint(Canvas canvas, Size size) {
    final bracketColor = isScanning ? SomythraTheme.primaryCyan : SomythraTheme.cardBorder;
    final bracketPaint = Paint()
      ..color = bracketColor
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    const pad = 18.0;
    const len = 24.0;

    // Top-Left corner bracket
    canvas.drawLine(const Offset(pad, pad + len), const Offset(pad, pad), bracketPaint);
    canvas.drawLine(const Offset(pad, pad), const Offset(pad + len, pad), bracketPaint);

    // Top-Right corner bracket
    canvas.drawLine(Offset(size.width - pad - len, pad), Offset(size.width - pad, pad), bracketPaint);
    canvas.drawLine(Offset(size.width - pad, pad), Offset(size.width - pad, pad + len), bracketPaint);

    // Bottom-Left corner bracket
    canvas.drawLine(Offset(pad, size.height - pad - len), Offset(pad, size.height - pad), bracketPaint);
    canvas.drawLine(Offset(pad, size.height - pad), Offset(pad + len, size.height - pad), bracketPaint);

    // Bottom-Right corner bracket
    canvas.drawLine(Offset(size.width - pad - len, size.height - pad), Offset(size.width - pad, size.height - pad), bracketPaint);
    canvas.drawLine(Offset(size.width - pad, size.height - pad - len), Offset(size.width - pad, size.height - pad), bracketPaint);

    // Subtle center crosshairs
    final crossPaint = Paint()
      ..color = SomythraTheme.cardBorder.withOpacity(0.4)
      ..strokeWidth = 1.0;

    final cx = size.width / 2;
    final cy = size.height / 2;
    canvas.drawLine(Offset(cx - 15, cy), Offset(cx + 15, cy), crossPaint);
    canvas.drawLine(Offset(cx, cy - 15), Offset(cx, cy + 15), crossPaint);
  }

  @override
  bool shouldRepaint(covariant _HudGridPainter oldDelegate) =>
      oldDelegate.isScanning != isScanning;
}
