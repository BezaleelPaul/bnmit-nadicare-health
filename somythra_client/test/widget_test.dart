import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:somythra_client/theme/somythra_theme.dart';
import 'package:somythra_client/widgets/metric_card.dart';
import 'package:somythra_client/widgets/alert_banner.dart';
import 'package:somythra_client/screens/about_screen.dart';
import 'package:somythra_client/screens/vision_stream_screen.dart';
import 'package:somythra_client/models/user_profile.dart';

void main() {
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
    final profile = UserProfile(
      name: 'Tester',
      age: 25,
      weightKg: 70.0,
      baselineHr: 65.0,
      baselineHrv: 55.0,
      fitnessLevel: 'intermediate',
    );

    await tester.pumpWidget(
      MaterialApp(
        theme: SomythraTheme.darkTheme,
        home: Scaffold(
          body: VisionStreamScreen(profile: profile),
        ),
      ),
    );

    expect(find.text('CAMERA rPPG PULSE SCAN'), findsOneWidget);
    expect(find.text('START LIVE SCAN'), findsOneWidget);
    expect(find.text('POSITION FACE WITHIN SCANNER GUIDE'), findsOneWidget);
  });
}
