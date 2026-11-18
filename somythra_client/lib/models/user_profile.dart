class UserProfile {
  final String name;
  final int age;
  final double weightKg;
  final double baselineHr;
  final double baselineHrv;
  final String fitnessLevel;

  UserProfile({
    this.name = 'Somythra Explorer',
    this.age = 25,
    this.weightKg = 70.0,
    this.baselineHr = 65.0,
    this.baselineHrv = 55.0,
    this.fitnessLevel = 'intermediate',
  });

  double get maxHr => 208.0 - (0.7 * age);
  double get criticalHrThreshold => maxHr * 0.90;
  double get warningHrThreshold => maxHr * 0.75;

  Map<String, dynamic> toJson() => {
    'name': name,
    'age': age,
    'weight_kg': weightKg,
    'baseline_hr': baselineHr,
    'baseline_hrv': baselineHrv,
    'fitness_level': fitnessLevel,
  };

  factory UserProfile.fromJson(Map<String, dynamic> json) => UserProfile(
    name: json['name'] ?? 'Somythra Explorer',
    age: (json['age'] as num?)?.toInt() ?? 25,
    weightKg: (json['weight_kg'] as num?)?.toDouble() ?? 70.0,
    baselineHr: (json['baseline_hr'] as num?)?.toDouble() ?? 65.0,
    baselineHrv: (json['baseline_hrv'] as num?)?.toDouble() ?? 55.0,
    fitnessLevel: json['fitness_level'] ?? 'intermediate',
  );
}
