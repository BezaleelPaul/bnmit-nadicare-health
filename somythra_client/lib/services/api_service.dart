import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/user_profile.dart';
import '../models/telemetry.dart';

class ApiService {
  static const String baseUrl = 'http://127.0.0.1:8000';

  static Future<bool> checkHealth() async {
    try {
      final res = await http.get(Uri.parse('$baseUrl/health')).timeout(
            const Duration(seconds: 3),
          );
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<RecoveryResultModel?> simulateRecovery({
    required UserProfile profile,
    required double peakHr,
    int durationSec = 300,
    int stepSec = 5,
  }) async {
    try {
      final res = await http.post(
        Uri.parse('$baseUrl/api/v1/twin/simulate'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'profile': profile.toJson(),
          'peak_hr': peakHr,
          'duration_seconds': durationSec,
          'time_step_seconds': stepSec,
        }),
      );
      if (res.statusCode == 200) {
        return RecoveryResultModel.fromJson(jsonDecode(res.body));
      }
    } catch (_) {}
    return null;
  }

  static Future<Map<String, dynamic>?> checkSafety({
    required double hr,
    required double hrv,
    UserProfile? profile,
  }) async {
    try {
      final res = await http.post(
        Uri.parse('$baseUrl/api/v1/safety/check'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'hr': hr,
          'hrv_rmssd': hrv,
          'profile': profile?.toJson(),
        }),
      );
      if (res.statusCode == 200) {
        return jsonDecode(res.body) as Map<String, dynamic>;
      }
    } catch (_) {}
    return null;
  }

  static Future<Map<String, dynamic>?> predictStress({
    required double hr,
    required double hrv,
    double activityLoad = 1.0,
  }) async {
    try {
      final res = await http.post(
        Uri.parse('$baseUrl/api/v1/stress/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'hr': hr,
          'hrv_rmssd': hrv,
          'activity_load': activityLoad,
        }),
      );
      if (res.statusCode == 200) {
        return jsonDecode(res.body) as Map<String, dynamic>;
      }
    } catch (_) {}
    return null;
  }
}
