import 'package:flutter/material.dart';
import '../theme/somythra_theme.dart';

class AlertBanner extends StatelessWidget {
  final String level;
  final String message;

  const AlertBanner({
    super.key,
    required this.level,
    required this.message,
  });

  @override
  Widget build(BuildContext context) {
    Color bannerColor;
    IconData icon;
    String statusTitle;

    switch (level.toUpperCase()) {
      case 'CRITICAL':
        bannerColor = SomythraTheme.criticalRed;
        icon = Icons.warning_rounded;
        statusTitle = 'CRITICAL ALERT';
        break;
      case 'WARNING':
        bannerColor = SomythraTheme.warningOrange;
        icon = Icons.info_outline_rounded;
        statusTitle = 'PHYSIOLOGICAL WARNING';
        break;
      default:
        bannerColor = SomythraTheme.safeGreen;
        icon = Icons.verified_user_rounded;
        statusTitle = 'HOMEOSTASIS SECURE';
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: bannerColor.withOpacity(0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: bannerColor.withOpacity(0.35),
          width: 1.5,
        ),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(6),
            decoration: BoxDecoration(
              color: bannerColor.withOpacity(0.18),
              shape: BoxShape.circle,
            ),
            child: Icon(icon, color: bannerColor, size: 20),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  statusTitle,
                  style: TextStyle(
                    color: bannerColor,
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.8,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  message,
                  style: const TextStyle(
                    color: SomythraTheme.textPrimary,
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
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
