import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:somythra_client/theme/somythra_theme.dart';
import 'package:somythra_client/widgets/metric_card.dart';
import 'package:somythra_client/widgets/alert_banner.dart';
import 'package:somythra_client/widgets/ces_gauge.dart';
import 'package:somythra_client/widgets/autonomic_nexus.dart';
import 'package:somythra_client/screens/about_screen.dart';
import 'package:somythra_client/screens/vision_stream_screen.dart';
import 'package:somythra_client/screens/dashboard_screen.dart';
import 'package:somythra_client/screens/twin_simulation_screen.dart';
import 'package:somythra_client/screens/strain_analytics_screen.dart';
import 'package:somythra_client/models/user_profile.dart';
import 'package:somythra_client/models/telemetry.dart';

void main() {
  final testProfile = UserProfile(
    name: 'Tester',
    age: 25,
    weightKg: 70.0,
    baselineHr: 65.0,
    baselineHrv: 55.0,
    fitnessLevel: 'intermediate',
  );

  testWidgets('MetricCard and AlertBanner test', (WidgetTester tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: SomythraTheme.darkTheme,
        home: const Scaffold(
          body: Column(
            children: [
              MetricCard(
                title: 'Heart Rate',
                value: '72',
                unit: 'BPM',
                icon: Icons.favorite,
                accentColor: SomythraTheme.heartRose,
              ),
              AlertBanner(
                level: 'SAFE',
                message: 'Physiological vitals stable.',
              ),
            ],
          ),
        ),
      ),
    );

    expect(find.text('HEART RATE'), findsOneWidget);
    expect(find.text('72'), findsOneWidget);
    expect(find.text('HOMEOSTASIS SECURE'), findsOneWidget);
    expect(find.text('Physiological vitals stable.'), findsOneWidget);
  });

  testWidgets('AutonomicNexus and CesGauge test', (WidgetTester tester) async {
    final telemetry = TelemetryData(
      timestamp: DateTime.now(),
      hr: 74.0,
      hrv: 56.0,
      meanRr: 810.0,
      sdnn: 64.0,
      activityLoad: 1.0,
      cesScore: 82.0,
      cesRating: 'Optimal Interconnected State',
      safetyLevel: 'SAFE',
      safetyMessage: 'Physiological vitals stable.',
      stressLevel: 'Balanced',
      stressIndex: 38.0,
    );

    await tester.pumpWidget(
      MaterialApp(
        theme: SomythraTheme.darkTheme,
        home: Scaffold(
          body: SingleChildScrollView(
            child: Column(
              children: [
                AutonomicNexusWidget(
                  telemetry: telemetry,
                  profile: testProfile,
                ),
                CesGauge(
                  score: telemetry.cesScore,
                  rating: telemetry.cesRating,
                ),
              ],
            ),
          ),
        ),
      ),
    );

    expect(find.text('AUTONOMIC NEXUS'), findsOneWidget);
    expect(find.text('CARDIAC ENHANCEMENT SCORE'), findsOneWidget);
    expect(find.text('82.0'), findsOneWidget);
  });

  testWidgets('DashboardScreen test', (WidgetTester tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: SomythraTheme.darkTheme,
        home: Scaffold(
          body: DashboardScreen(profile: testProfile),
        ),
      ),
    );

    expect(find.text('SOMYTHRA CORE'), findsOneWidget);
    expect(find.text('HEART RATE'), findsOneWidget);
    expect(find.text('HRV (RMSSD)'), findsOneWidget);
  });

  testWidgets('TwinSimulationScreen test', (WidgetTester tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: SomythraTheme.darkTheme,
        home: Scaffold(
          body: TwinSimulationScreen(profile: testProfile),
        ),
      ),
    );

    expect(find.text('SOMYTHRA TWIN'), findsOneWidget);
    expect(find.text('SIMULATE TWIN RECOVERY'), findsOneWidget);
  });

  testWidgets('StrainAnalyticsScreen test', (WidgetTester tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: SomythraTheme.darkTheme,
        home: Scaffold(
          body: StrainAnalyticsScreen(profile: testProfile),
        ),
      ),
    );

    expect(find.text('SOMYTHRA RESEARCH & STRAIN'), findsOneWidget);
    expect(find.text('OBSERVED VITAL INPUTS'), findsOneWidget);
  });

  testWidgets('AboutScreen charter test', (WidgetTester tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: SomythraTheme.darkTheme,
        home: const Scaffold(
          body: AboutScreen(),
        ),
      ),
    );

    expect(find.text('SOMYTHRA'), findsOneWidget);
    expect(find.text('SOMYTHRA Core'), findsOneWidget);
    expect(find.text('SOMYTHRA Twin'), findsOneWidget);
    expect(find.text('SOMYTHRA Field'), findsOneWidget);
    expect(find.text('SOMYTHRA Research'), findsOneWidget);
  });

  testWidgets('VisionStreamScreen test', (WidgetTester tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: SomythraTheme.darkTheme,
        home: Scaffold(
          body: VisionStreamScreen(profile: testProfile),
        ),
      ),
    );

    expect(find.text('CAMERA rPPG PULSE SCAN'), findsOneWidget);
    expect(find.text('START LIVE SCAN'), findsOneWidget);
    expect(find.text('POSITION FACE WITHIN SCANNER GUIDE'), findsOneWidget);
  });
}
