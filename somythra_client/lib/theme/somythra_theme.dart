import 'package:flutter/material.dart';

class SomythraTheme {
  // Deep Cyber-Biological Obsidian Palette
  static const Color background = Color(0xFF070B10);
  static const Color backgroundAlt = Color(0xFF0B1017);
  static const Color surface = Color(0xFF0F1722);
  static const Color surfaceGlass = Color(0xCC131D2B);
  static const Color card = Color(0xFF141F2D);
  static const Color cardElevated = Color(0xFF1A2738);
  static const Color cardBorder = Color(0xFF22344A);
  static const Color cardBorderBright = Color(0xFF2F4866);

  // Futuristic Laser & Bio-Luminescent Accents
  static const Color primaryCyan = Color(0xFF00E5FF);
  static const Color cyanGlow = Color(0xFF38BDF8);
  static const Color cyanDim = Color(0xFF0284C7);
  static const Color heartRose = Color(0xFFFF2A5F);
  static const Color heartRoseDim = Color(0xFFBE123C);
  static const Color safeGreen = Color(0xFF00F59B);
  static const Color safeGreenDim = Color(0xFF059669);
  static const Color warningOrange = Color(0xFFFFB020);
  static const Color criticalRed = Color(0xFFFF334B);
  static const Color neuralViolet = Color(0xFFA855F7);
  static const Color electricAmber = Color(0xFFFBBF24);

  // High-Contrast Scientific Typography
  static const Color textPrimary = Color(0xFFF8FAFC);
  static const Color textSecondary = Color(0xFF94A3B8);
  static const Color textMuted = Color(0xFF64748B);
  static const Color textDark = Color(0xFF334155);

  // Glassmorphic Card Decoration Helper
  static BoxDecoration glassCardDecoration({
    Color borderColor = cardBorder,
    Color glowColor = Colors.transparent,
    double radius = 16.0,
    bool elevated = false,
  }) {
    return BoxDecoration(
      color: elevated ? cardElevated : card,
      borderRadius: BorderRadius.circular(radius),
      border: Border.all(color: borderColor, width: 1.2),
      boxShadow: [
        BoxShadow(
          color: Colors.black.withOpacity(0.4),
          blurRadius: 18,
          offset: const Offset(0, 6),
        ),
        if (glowColor != Colors.transparent)
          BoxShadow(
            color: glowColor.withOpacity(0.12),
            blurRadius: 22,
            spreadRadius: 1,
            offset: const Offset(0, 2),
          ),
      ],
    );
  }

  static ThemeData get darkTheme {
    return ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: background,
      primaryColor: primaryCyan,
      canvasColor: background,
      colorScheme: const ColorScheme.dark(
        primary: primaryCyan,
        secondary: heartRose,
        surface: surface,
        error: criticalRed,
      ),
      cardTheme: CardThemeData(
        color: card,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: const BorderSide(color: cardBorder, width: 1.2),
        ),
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: background,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: TextStyle(
          color: textPrimary,
          fontSize: 20,
          fontWeight: FontWeight.w800,
          letterSpacing: 1.0,
        ),
      ),
      navigationRailTheme: const NavigationRailThemeData(
        backgroundColor: surface,
        selectedIconTheme: IconThemeData(color: primaryCyan, size: 24),
        unselectedIconTheme: IconThemeData(color: textMuted, size: 22),
        selectedLabelTextStyle: TextStyle(
          color: primaryCyan,
          fontWeight: FontWeight.w700,
          fontSize: 11,
          letterSpacing: 0.8,
        ),
        unselectedLabelTextStyle: TextStyle(
          color: textMuted,
          fontWeight: FontWeight.w500,
          fontSize: 11,
          letterSpacing: 0.5,
        ),
      ),
      sliderTheme: const SliderThemeData(
        activeTrackColor: primaryCyan,
        inactiveTrackColor: cardBorder,
        thumbColor: primaryCyan,
        overlayColor: Color(0x3300E5FF),
        trackHeight: 4.0,
      ),
      tooltipTheme: TooltipThemeData(
        decoration: BoxDecoration(
          color: surface,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: cardBorder),
        ),
        textStyle: const TextStyle(color: textPrimary, fontSize: 11),
      ),
    );
  }
}
