class TelemetryData {
  final DateTime timestamp;
  final double hr;
  final double hrv;
  final double meanRr;
  final double sdnn;
  final double activityLoad;
  final double cesScore;
  final String cesRating;
  final String safetyLevel;
  final String safetyMessage;
  final String stressLevel;
  final double stressIndex;

  TelemetryData({
    required this.timestamp,
    required this.hr,
    required this.hrv,
    required this.meanRr,
    required this.sdnn,
    required this.activityLoad,
    required this.cesScore,
    required this.cesRating,
    required this.safetyLevel,
    required this.safetyMessage,
    required this.stressLevel,
    required this.stressIndex,
  });

  factory TelemetryData.fromJson(Map<String, dynamic> json) => TelemetryData(
    timestamp: json['timestamp'] != null
        ? DateTime.tryParse(json['timestamp']) ?? DateTime.now()
        : DateTime.now(),
    hr: (json['hr'] as num?)?.toDouble() ?? 70.0,
    hrv: (json['hrv'] as num?)?.toDouble() ?? 50.0,
    meanRr: (json['mean_rr'] as num?)?.toDouble() ?? 850.0,
    sdnn: (json['sdnn'] as num?)?.toDouble() ?? 60.0,
    activityLoad: (json['activity_load'] as num?)?.toDouble() ?? 1.0,
    cesScore: (json['ces_score'] as num?)?.toDouble() ?? 75.0,
    cesRating: json['ces_rating'] ?? 'Optimal Interconnected State',
    safetyLevel: json['safety_level'] ?? 'SAFE',
    safetyMessage: json['safety_message'] ?? 'Physiological vitals stable.',
    stressLevel: json['stress_level'] ?? 'Balanced / Low Stress',
    stressIndex: (json['stress_index'] as num?)?.toDouble() ?? 45.0,
  );
}

class RecoveryPointModel {
  final int timeSec;
  final double predictedHr;
  final double recoveryPct;

  RecoveryPointModel({
    required this.timeSec,
    required this.predictedHr,
    required this.recoveryPct,
  });

  factory RecoveryPointModel.fromJson(Map<String, dynamic> json) =>
      RecoveryPointModel(
        timeSec: (json['time_sec'] as num?)?.toInt() ?? 0,
        predictedHr: (json['predicted_hr'] as num?)?.toDouble() ?? 0.0,
        recoveryPct: (json['recovery_pct'] as num?)?.toDouble() ?? 0.0,
      );
}

class RecoveryResultModel {
  final List<RecoveryPointModel> points;
  final double halfRecoverySec;
  final double fullRecoveryEstimateSec;
  final double decayConstant;
  final double maxHr;

  RecoveryResultModel({
    required this.points,
    required this.halfRecoverySec,
    required this.fullRecoveryEstimateSec,
    required this.decayConstant,
    required this.maxHr,
  });

  factory RecoveryResultModel.fromJson(Map<String, dynamic> json) =>
      RecoveryResultModel(
        points: (json['points'] as List<dynamic>?)
                ?.map((e) => RecoveryPointModel.fromJson(e))
                .toList() ??
            [],
        halfRecoverySec: (json['half_recovery_sec'] as num?)?.toDouble() ?? 0.0,
        fullRecoveryEstimateSec:
            (json['full_recovery_estimate_sec'] as num?)?.toDouble() ?? 0.0,
        decayConstant: (json['decay_constant'] as num?)?.toDouble() ?? 0.04,
        maxHr: (json['max_hr'] as num?)?.toDouble() ?? 190.0,
      );
}
