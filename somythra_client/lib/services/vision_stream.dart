import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../models/vision_packet.dart';

class VisionStreamService {
  static const String wsUrl = 'ws://127.0.0.1:8000/ws/vision/live';

  final _controller = StreamController<VisionPacket>.broadcast();
  Stream<VisionPacket> get stream => _controller.stream;

  WebSocketChannel? _channel;
  bool _isConnected = false;
  bool get isConnected => _isConnected;

  void start() {
    _connectWebSocket();
  }

  void _connectWebSocket() {
    try {
      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));
      _channel!.stream.listen(
        (data) {
          _isConnected = true;
          final decoded = jsonDecode(data as String) as Map<String, dynamic>;
          _controller.add(VisionPacket.fromJson(decoded));
        },
        onError: (_) {
          _isConnected = false;
        },
        onDone: () {
          _isConnected = false;
        },
      );
    } catch (_) {
      _isConnected = false;
    }
  }

  void sendFrame(Uint8List jpegBytes) {
    if (_isConnected && _channel != null) {
      _channel!.sink.add(jpegBytes);
    }
  }

  void dispose() {
    _channel?.sink.close();
    _controller.close();
  }
}
