import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../models/telemetry.dart';

class TelemetryStreamService {
  static const String wsUrl = 'ws://127.0.0.1:8000/ws/telemetry';

  final _controller = StreamController<TelemetryData>.broadcast();
  Stream<TelemetryData> get stream => _controller.stream;

  WebSocketChannel? _channel;
  Timer? _reconnectTimer;
  Timer? _mockTimer;
  bool _isConnected = false;
  bool get isConnected => _isConnected;

  final Random _rng = Random();
  double _mockHr = 72.0;
  double _mockHrv = 54.0;

  void start() {
    _connectWebSocket();
  }

  void _connectWebSocket() {
    try {
      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));
      _channel!.stream.listen(
        (data) {
          if (!_isConnected) {
            _isConnected = true;
            _stopMockMode();
          }
          final decoded = jsonDecode(data as String) as Map<String, dynamic>;
          _controller.add(TelemetryData.fromJson(decoded));
        },
        onError: (err) {
          _handleDisconnect();
        },
        onDone: () {
          _handleDisconnect();
        },
      );
    } catch (_) {
      _handleDisconnect();
    }
  }

  void _handleDisconnect() {
    _isConnected = false;
    _startMockMode();
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(const Duration(seconds: 4), () {
      _connectWebSocket();
    });
  }

  void _startMockMode() {
    if (_mockTimer != null && _mockTimer!.isActive) return;
    _mockTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_isConnected) return;
      _mockHr = (_mockHr + (_rng.nextDouble() * 4 - 2)).clamp(55.0, 130.0);
      _mockHrv = (_mockHrv + (_rng.nextDouble() * 6 - 3)).clamp(20.0, 95.0);

      _controller.add(
        TelemetryData(
          timestamp: DateTime.now(),
          hr: double.parse(_mockHr.toStringAsFixed(1)),
          hrv: double.parse(_mockHrv.toStringAsFixed(1)),
          meanRr: double.parse((60000.0 / _mockHr).toStringAsFixed(1)),
          sdnn: double.parse((_mockHrv * 1.15).toStringAsFixed(1)),
          activityLoad: 1.0,
          cesScore: 78.5,
          cesRating: 'Optimal Interconnected State (Local Mode)',
          safetyLevel: _mockHr > 120 ? 'WARNING' : 'SAFE',
          safetyMessage: _mockHr > 120
              ? 'Elevated Cardiac Demand (Simulated)'
              : 'Physiological vitals stable within baseline limits.',
          stressLevel: _mockHrv < 30 ? 'High Autonomic Stress' : 'Balanced / Low Stress',
          stressIndex: double.parse((50.0 * (70 / _mockHrv)).toStringAsFixed(1)),
        ),
      );
    });
  }

  void _stopMockMode() {
    _mockTimer?.cancel();
    _mockTimer = null;
  }

  void dispose() {
    _reconnectTimer?.cancel();
    _mockTimer?.cancel();
    _channel?.sink.close();
    _controller.close();
  }
}
