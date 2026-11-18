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
    String badgeCode;

    switch (level.toUpperCase()) {
      case 'CRITICAL':
        bannerColor = SomythraTheme.criticalRed;
        icon = Icons.warning_rounded;
        statusTitle = 'CRITICAL ALERT';
        badgeCode = 'LVL-3 CRITICAL';
        break;
      case 'WARNING':
        bannerColor = SomythraTheme.warningOrange;
        icon = Icons.info_outline_rounded;
        statusTitle = 'PHYSIOLOGICAL WARNING';
        badgeCode = 'LVL-2 ELEVATED';
        break;
      default:
        bannerColor = SomythraTheme.safeGreen;
        icon = Icons.verified_user_rounded;
        statusTitle = 'HOMEOSTASIS SECURE';
        badgeCode = 'LVL-1 NOMINAL';
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: bannerColor.withOpacity(0.07),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: bannerColor.withOpacity(0.35),
          width: 1.2,
        ),
        boxShadow: [
          BoxShadow(
            color: bannerColor.withOpacity(0.08),
            blurRadius: 16,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(7),
            decoration: BoxDecoration(
              color: bannerColor.withOpacity(0.16),
              shape: BoxShape.circle,
              border: Border.all(color: bannerColor.withOpacity(0.4), width: 1),
            ),
            child: Icon(icon, color: bannerColor, size: 18),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      statusTitle,
                      style: TextStyle(
                        color: bannerColor,
                        fontSize: 11.5,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.0,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: bannerColor.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        badgeCode,
                        style: TextStyle(
                          color: bannerColor,
                          fontSize: 9,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 0.5,
                          fontFamily: 'monospace',
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 3),
                Text(
                  message,
                  style: const TextStyle(
                    color: SomythraTheme.textPrimary,
                    fontSize: 12.5,
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
