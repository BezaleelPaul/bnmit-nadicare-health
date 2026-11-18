import 'package:flutter/material.dart';
import 'models/user_profile.dart';
import 'theme/somythra_theme.dart';
import 'screens/dashboard_screen.dart';
import 'screens/vision_stream_screen.dart';
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
      VisionStreamScreen(profile: _profile),
      TwinSimulationScreen(profile: _profile),
      StrainAnalyticsScreen(profile: _profile),
      const AboutScreen(),
    ];

    return Scaffold(
      appBar: AppBar(
        titleSpacing: 20,
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(7),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    SomythraTheme.primaryCyan.withOpacity(0.25),
                    SomythraTheme.cyanDim.withOpacity(0.1),
                  ],
                ),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(
                  color: SomythraTheme.primaryCyan.withOpacity(0.5),
                  width: 1.2,
                ),
                boxShadow: [
                  BoxShadow(
                    color: SomythraTheme.primaryCyan.withOpacity(0.25),
                    blurRadius: 10,
                    spreadRadius: 1,
                  ),
                ],
              ),
              child: const Icon(
                Icons.all_inclusive_rounded,
                color: SomythraTheme.primaryCyan,
                size: 20,
              ),
            ),
            const SizedBox(width: 14),
            const Text(
              'SOMYTHRA',
              style: TextStyle(
                fontWeight: FontWeight.w900,
                letterSpacing: 2.2,
                fontSize: 19,
                color: SomythraTheme.textPrimary,
              ),
            ),
            const SizedBox(width: 10),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
              decoration: BoxDecoration(
                color: SomythraTheme.cardElevated,
                borderRadius: BorderRadius.circular(4),
                border: Border.all(color: SomythraTheme.cardBorder),
              ),
              child: const Text(
                'BIO-INTELLIGENCE',
                style: TextStyle(
                  color: SomythraTheme.cyanGlow,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 1.0,
                  fontSize: 9.5,
                  fontFamily: 'monospace',
                ),
              ),
            ),
          ],
        ),
        actions: [
          // Host daemon status indicator
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: SomythraTheme.surface,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: SomythraTheme.cardBorder),
            ),
            child: Row(
              children: [
                Container(
                  width: 6,
                  height: 6,
                  decoration: const BoxDecoration(
                    color: SomythraTheme.safeGreen,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 6),
                const Text(
                  'ENGINE 127.0.0.1:8000',
                  style: TextStyle(
                    color: SomythraTheme.textSecondary,
                    fontSize: 9.5,
                    fontWeight: FontWeight.w700,
                    fontFamily: 'monospace',
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 12),
          IconButton(
            tooltip: 'User Physiological Profile & Baselines',
            icon: Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: SomythraTheme.card,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: SomythraTheme.cardBorder),
              ),
              child: const Icon(Icons.tune_rounded, color: SomythraTheme.primaryCyan, size: 18),
            ),
            onPressed: () => _showProfileDialog(context),
          ),
          const SizedBox(width: 16),
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
                      icon: Icon(Icons.videocam_outlined),
                      selectedIcon: Icon(Icons.videocam_rounded),
                      label: Text('Vision Scan'),
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
                  icon: Icon(Icons.videocam_outlined),
                  activeIcon: Icon(Icons.videocam_rounded),
                  label: 'Vision Scan',
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
          borderRadius: BorderRadius.circular(18),
          side: const BorderSide(color: SomythraTheme.cardBorder, width: 1.2),
        ),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: SomythraTheme.primaryCyan.withOpacity(0.15),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(Icons.tune_rounded, color: SomythraTheme.primaryCyan, size: 18),
            ),
            const SizedBox(width: 10),
            const Text(
              'Personal Baseline Settings',
              style: TextStyle(color: SomythraTheme.textPrimary, fontWeight: FontWeight.w800, fontSize: 16),
            ),
          ],
        ),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: nameCtrl,
                decoration: const InputDecoration(labelText: 'Name', labelStyle: TextStyle(color: SomythraTheme.textSecondary)),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: ageCtrl,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(labelText: 'Age (Years)', labelStyle: TextStyle(color: SomythraTheme.textSecondary)),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: hrCtrl,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(labelText: 'Resting Baseline HR (BPM)', labelStyle: TextStyle(color: SomythraTheme.textSecondary)),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: hrvCtrl,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(labelText: 'Resting Baseline HRV (ms)', labelStyle: TextStyle(color: SomythraTheme.textSecondary)),
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
              foregroundColor: Colors.black,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
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
            child: const Text('Save Baseline', style: TextStyle(fontWeight: FontWeight.w800)),
          ),
        ],
      ),
    );
  }
}
