import 'package:flutter/material.dart';
import '../theme/somythra_theme.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Banner Card
          Container(
            padding: const EdgeInsets.all(28),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  SomythraTheme.card,
                  SomythraTheme.surface,
                ],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: SomythraTheme.primaryCyan.withOpacity(0.35),
                width: 1.5,
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      'SOMYTHRA',
                      style: TextStyle(
                        color: SomythraTheme.textPrimary,
                        fontSize: 28,
                        fontWeight: FontWeight.w900,
                        letterSpacing: 2.0,
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: SomythraTheme.primaryCyan.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: SomythraTheme.primaryCyan,
                        ),
                      ),
                      child: const Text(
                        'SOMYTHRA-181126-001',
                        style: TextStyle(
                          color: SomythraTheme.primaryCyan,
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          fontFamily: 'monospace',
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                const Text(
                  'The science of understanding the human body as an interconnected living system.',
                  style: TextStyle(
                    color: SomythraTheme.cyanGlow,
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 14),
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 14, vertical: 10),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.25),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                      color: SomythraTheme.cardBorder,
                    ),
                  ),
                  child: const Text(
                    '“Life is not a collection of parts. It is a system of relationships.”',
                    style: TextStyle(
                      color: SomythraTheme.textPrimary,
                      fontStyle: FontStyle.italic,
                      fontSize: 14,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 28),

          // Etymology Section
          const Text(
            'INVENTED ETYMOLOGY',
            style: TextStyle(
              color: SomythraTheme.textSecondary,
              fontSize: 12,
              fontWeight: FontWeight.w700,
              letterSpacing: 1.2,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              _buildEtymologyCard('SOM', 'The Living Vessel', 'The biological human body and organic infrastructure.'),
              const SizedBox(width: 14),
              _buildEtymologyCard('YTH', 'Interdependence', 'The dynamic relationships and feedback loops across bodily systems.'),
              const SizedBox(width: 14),
              _buildEtymologyCard('RA', 'Pattern Intelligence', 'Holistic understanding and synthesis of interconnected signals.'),
            ],
          ),
          const SizedBox(height: 28),

          // Four Pillars Architecture
          const Text(
            'PRODUCT & RESEARCH ARCHITECTURE',
            style: TextStyle(
              color: SomythraTheme.textSecondary,
              fontSize: 12,
              fontWeight: FontWeight.w700,
              letterSpacing: 1.2,
            ),
          ),
          const SizedBox(height: 12),
          _buildPillarItem(
            'SOMYTHRA Core',
            'Physiological telemetry ingestion, Tanaka threshold safety validation, signal quality assessment, and personal baselines.',
            Icons.hub_rounded,
            SomythraTheme.primaryCyan,
          ),
          const SizedBox(height: 12),
          _buildPillarItem(
            'SOMYTHRA Twin',
            'Exploratory physiological digital-twin recovery models, exponential decay dynamics, and Cardiac Enhancement Scoring (CES).',
            Icons.device_hub_rounded,
            SomythraTheme.heartRose,
          ),
          const SizedBox(height: 12),
          _buildPillarItem(
            'SOMYTHRA Field',
            'Cross-platform mobile and edge tools for community wellbeing awareness and environmental strain without punitive surveillance.',
            Icons.nature_people_rounded,
            SomythraTheme.safeGreen,
          ),
          const SizedBox(height: 12),
          _buildPillarItem(
            'SOMYTHRA Research',
            'Autonomic strain classification, HRV feature engineering, reproducible experiments, and physiological dataset curation.',
            Icons.science_rounded,
            SomythraTheme.cyanGlow,
          ),
          const SizedBox(height: 28),

          // Provenance Note
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: SomythraTheme.card,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: SomythraTheme.cardBorder),
            ),
            child: const Text(
              'SOMYTHRA continues technical work originally developed under the prototype name NadiCare. '
              'The rebrand establishes a broader scientific identity; it represents research and wellbeing awareness, '
              'not a substitute for professional medical assessment, diagnosis, or treatment.',
              style: TextStyle(
                color: SomythraTheme.textMuted,
                fontSize: 12,
                height: 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEtymologyCard(String title, String subtitle, String desc) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: SomythraTheme.card,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: SomythraTheme.cardBorder),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: const TextStyle(
                color: SomythraTheme.primaryCyan,
                fontSize: 18,
                fontWeight: FontWeight.w800,
                letterSpacing: 1.0,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              subtitle,
              style: const TextStyle(
                color: SomythraTheme.textPrimary,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              desc,
              style: const TextStyle(
                color: SomythraTheme.textSecondary,
                fontSize: 11,
                height: 1.4,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPillarItem(
      String title, String desc, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: SomythraTheme.card,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: SomythraTheme.cardBorder),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: color.withOpacity(0.12),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: color, size: 22),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    color: SomythraTheme.textPrimary,
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  desc,
                  style: const TextStyle(
                    color: SomythraTheme.textSecondary,
                    fontSize: 12,
                    height: 1.4,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
