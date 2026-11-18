import 'package:flutter/material.dart';
import 'models/user_profile.dart';
import 'theme/somythra_theme.dart';
import 'screens/dashboard_screen.dart';
import 'screens/twin_simulation_screen.dart';
import 'screens/strain_analytics_screen.dart';
import 'screens/about_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SomythraApp());
}

class SomythraApp extends StatelessWidget {
  const SomythraApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SOMYTHRA — Physiological Intelligence',
      debugShowCheckedModeBanner: false,
      theme: SomythraTheme.darkTheme,
      home: const SomythraMainShell(),
    );
  }
}

class SomythraMainShell extends StatefulWidget {
  const SomythraMainShell({super.key});

  @override
  State<SomythraMainShell> createState() => _SomythraMainShellState();
}

class _SomythraMainShellState extends State<SomythraMainShell> {
  int _selectedIndex = 0;
  UserProfile _profile = UserProfile(
    name: 'Bezaleel Paul',
    age: 22,
    weightKg: 68.0,
    baselineHr: 62.0,
    baselineHrv: 58.0,
    fitnessLevel: 'intermediate',
  );

  @override
  Widget build(BuildContext context) {
    final screens = [
      DashboardScreen(profile: _profile),
      TwinSimulationScreen(profile: _profile),
      StrainAnalyticsScreen(profile: _profile),
      const AboutScreen(),
    ];

    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: SomythraTheme.primaryCyan.withOpacity(0.15),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(
                Icons.all_inclusive_rounded,
                color: SomythraTheme.primaryCyan,
                size: 20,
              ),
            ),
            const SizedBox(width: 12),
            const Text(
              'SOMYTHRA',
              style: TextStyle(
                fontWeight: FontWeight.w900,
                letterSpacing: 1.5,
                fontSize: 18,
              ),
            ),
            const SizedBox(width: 8),
            const Text(
              'PHYSIOLOGICAL TWIN',
              style: TextStyle(
                color: SomythraTheme.textMuted,
                fontWeight: FontWeight.w600,
                letterSpacing: 1.0,
                fontSize: 11,
              ),
            ),
          ],
        ),
        actions: [
          IconButton(
            tooltip: 'User Physiological Profile',
            icon: const Icon(Icons.tune_rounded, color: SomythraTheme.textSecondary),
            onPressed: () => _showProfileDialog(context),
          ),
          const SizedBox(width: 12),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final isDesktop = constraints.maxWidth >= 720;
          if (isDesktop) {
            return Row(
              children: [
                NavigationRail(
                  selectedIndex: _selectedIndex,
                  onDestinationSelected: (index) {
                    setState(() => _selectedIndex = index);
                  },
                  labelType: NavigationRailLabelType.all,
                  leading: const SizedBox(height: 12),
                  destinations: const [
                    NavigationRailDestination(
                      icon: Icon(Icons.monitor_heart_outlined),
                      selectedIcon: Icon(Icons.monitor_heart_rounded),
                      label: Text('Monitor'),
                    ),
                    NavigationRailDestination(
                      icon: Icon(Icons.device_hub_outlined),
                      selectedIcon: Icon(Icons.device_hub_rounded),
                      label: Text('Twin'),
                    ),
                    NavigationRailDestination(
                      icon: Icon(Icons.psychology_outlined),
                      selectedIcon: Icon(Icons.psychology_rounded),
                      label: Text('Strain'),
                    ),
                    NavigationRailDestination(
                      icon: Icon(Icons.info_outline_rounded),
                      selectedIcon: Icon(Icons.info_rounded),
                      label: Text('Charter'),
                    ),
                  ],
                ),
                const VerticalDivider(
                  thickness: 1,
                  width: 1,
                  color: SomythraTheme.cardBorder,
                ),
                Expanded(child: screens[_selectedIndex]),
              ],
            );
          } else {
            return screens[_selectedIndex];
          }
        },
      ),
      bottomNavigationBar: MediaQuery.of(context).size.width < 720
          ? BottomNavigationBar(
              currentIndex: _selectedIndex,
              onTap: (index) => setState(() => _selectedIndex = index),
              backgroundColor: SomythraTheme.surface,
              selectedItemColor: SomythraTheme.primaryCyan,
              unselectedItemColor: SomythraTheme.textMuted,
              type: BottomNavigationBarType.fixed,
              items: const [
                BottomNavigationBarItem(
                  icon: Icon(Icons.monitor_heart_outlined),
                  activeIcon: Icon(Icons.monitor_heart_rounded),
                  label: 'Monitor',
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.device_hub_outlined),
                  activeIcon: Icon(Icons.device_hub_rounded),
                  label: 'Twin',
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.psychology_outlined),
                  activeIcon: Icon(Icons.psychology_rounded),
                  label: 'Strain',
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.info_outline_rounded),
                  activeIcon: Icon(Icons.info_rounded),
                  label: 'Charter',
                ),
              ],
            )
          : null,
    );
  }

  void _showProfileDialog(BuildContext context) {
    final nameCtrl = TextEditingController(text: _profile.name);
    final ageCtrl = TextEditingController(text: _profile.age.toString());
    final hrCtrl = TextEditingController(text: _profile.baselineHr.toStringAsFixed(0));
    final hrvCtrl = TextEditingController(text: _profile.baselineHrv.toStringAsFixed(0));

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: SomythraTheme.card,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: const BorderSide(color: SomythraTheme.cardBorder),
        ),
        title: const Text(
          'Personal Baseline Settings',
          style: TextStyle(color: SomythraTheme.textPrimary, fontWeight: FontWeight.w700),
        ),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: nameCtrl,
                decoration: const InputDecoration(labelText: 'Name'),
              ),
              TextField(
                controller: ageCtrl,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(labelText: 'Age (Years)'),
              ),
              TextField(
                controller: hrCtrl,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(labelText: 'Resting Baseline HR (BPM)'),
              ),
              TextField(
                controller: hrvCtrl,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(labelText: 'Resting Baseline HRV (ms)'),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel', style: TextStyle(color: SomythraTheme.textMuted)),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: SomythraTheme.primaryCyan,
              foregroundColor: Colors.white,
            ),
            onPressed: () {
              setState(() {
                _profile = UserProfile(
                  name: nameCtrl.text.isNotEmpty ? nameCtrl.text : _profile.name,
                  age: int.tryParse(ageCtrl.text) ?? _profile.age,
                  baselineHr: double.tryParse(hrCtrl.text) ?? _profile.baselineHr,
                  baselineHrv: double.tryParse(hrvCtrl.text) ?? _profile.baselineHrv,
                  weightKg: _profile.weightKg,
                  fitnessLevel: _profile.fitnessLevel,
                );
              });
              Navigator.pop(ctx);
            },
            child: const Text('Save Baseline'),
          ),
        ],
      ),
    );
  }
}
