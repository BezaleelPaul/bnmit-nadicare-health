import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:somythra_client/theme/somythra_theme.dart';
import 'package:somythra_client/widgets/metric_card.dart';
import 'package:somythra_client/widgets/alert_banner.dart';
import 'package:somythra_client/screens/about_screen.dart';

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
}
