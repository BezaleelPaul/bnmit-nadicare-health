class VisionPacket {
  final int frameIndex;
  final double timestamp;
  final double pulseSample;
  final double? bpm;
  final double confidence;
  final double snrDb;
  final String qualityStatus;
  final bool isCadenceUpdate;
  final bool faceDetected;
  final String? rejectionReason;
  final String? stressLevel;
  final String? autonomicState;

  const VisionPacket({
    required this.frameIndex,
    required this.timestamp,
    required this.pulseSample,
    this.bpm,
    required this.confidence,
    required this.snrDb,
    required this.qualityStatus,
    required this.isCadenceUpdate,
    required this.faceDetected,
    this.rejectionReason,
    this.stressLevel,
    this.autonomicState,
  });

  factory VisionPacket.fromJson(Map<String, dynamic> json) {
    final stressMap = json['stress'] as Map<String, dynamic>?;
    return VisionPacket(
      frameIndex: json['frame_index'] as int? ?? 0,
      timestamp: (json['timestamp'] as num?)?.toDouble() ?? 0.0,
      pulseSample: (json['pulse_sample'] as num?)?.toDouble() ?? 0.0,
      bpm: (json['bpm'] as num?)?.toDouble(),
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      snrDb: (json['snr_db'] as num?)?.toDouble() ?? 0.0,
      qualityStatus: json['quality_status'] as String? ?? 'UNPROCESSED',
      isCadenceUpdate: json['is_cadence_update'] as bool? ?? false,
      faceDetected: json['face_detected'] as bool? ?? true,
      rejectionReason: json['rejection_reason'] as String?,
      stressLevel: stressMap?['stress_level'] as String?,
      autonomicState: stressMap?['autonomic_state'] as String?,
    );
  }

  bool get isReliable =>
      qualityStatus == 'OPTIMAL' || qualityStatus == 'ACCEPTABLE';
}
