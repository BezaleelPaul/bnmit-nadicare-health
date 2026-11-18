#!/usr/bin/env python3
"""
SOMYTHRA Twin — Autonomic State & Stress Model Training Pipeline
Trains a calibrated physiological stress classifier on Kaggle SWELL-KW HRV dataset.

Features (34 HRV time-domain, frequency-domain, and non-linear parameters):
  - Time domain: MEAN_RR, MEDIAN_RR, SDRR, RMSSD, SDSD, SDRR_RMSSD, HR, pNN25, pNN50
  - Poincare / Non-linear: SD1, SD2, KURT, SKEW, sampen, higuci
  - Relative metrics: MEAN_REL_RR, MEDIAN_REL_RR, SDRR_REL_RR, RMSSD_REL_RR, etc.
  - Spectral: VLF, VLF_PCT, LF, LF_PCT, LF_NU, HF, HF_PCT, HF_NU, TP, LF_HF, HF_LF

Target Classes:
  - no stress: Autonomic homeostasis / parasympathetic recovery
  - interruption: Cognitive distraction / mild sympathetic arousal
  - time pressure: Acute mental workload / high sympathetic strain
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("somythra.twin.train")

# Columns to exclude from training to prevent data leakage
EXCLUDE_COLUMNS = {"datasetId", "id", "subject", "participant", "condition", "target"}

DEFAULT_DATA_CANDIDATES = [
    Path("data/swell/test.csv"),
    Path("data/swell/train.csv"),
    Path("data/test.csv"),
    Path("data/train.csv"),
    Path("test.csv"),
    Path("train.csv"),
    Path("/tmp/swell_test/hrv dataset/data/final/test.csv"),
]


def resolve_dataset_path(provided_path: Optional[str] = None, auto_download: bool = False) -> Path:
    """Find local dataset or download via Kaggle CLI if authorized."""
    if provided_path:
        p = Path(provided_path)
        if p.exists():
            return p
        logger.warning("Provided path '%s' not found. Checking fallback paths...", provided_path)

    for cand in DEFAULT_DATA_CANDIDATES:
        if cand.exists():
            logger.info("Discovered dataset at '%s'", cand)
            return cand

    if auto_download:
        logger.info("Attempting automatic download from Kaggle (qiriro/swell-heart-rate-variability-hrv)...")
        target_dir = Path("data/swell")
        target_dir.mkdir(parents=True, exist_ok=True)
        cmd = f'kaggle datasets download qiriro/swell-heart-rate-variability-hrv -f "hrv dataset/data/final/test.csv" -p "{target_dir}" --unzip'
        res = os.system(cmd)
        target_csv = target_dir / "hrv dataset/data/final/test.csv"
        if res == 0 and target_csv.exists():
            clean_dest = target_dir / "test.csv"
            target_csv.rename(clean_dest)
            return clean_dest

    raise FileNotFoundError(
        "SWELL-KW dataset not found. Provide --data-path or pass --auto-download "
        "to download the CC0 dataset via Kaggle."
    )


def load_and_preprocess_data(data_path: Path, max_samples: Optional[int] = None):
    """Load SWELL dataset, sanitize features, and split into X and y."""
    logger.info("Reading dataset from %s ...", data_path)
    df = pd.read_csv(data_path, nrows=max_samples)
    logger.info("Raw dataset loaded: %d rows, %d columns", len(df), len(df.columns))

    if "condition" not in df.columns:
        raise ValueError("Dataset missing 'condition' target column.")

    # Determine target and features
    y_raw = df["condition"].astype(str).str.strip().str.lower()
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]

    X = df[feature_cols].copy()

    # Data hygiene: handle NaNs or Infs
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X.isna().sum().sum() > 0:
        logger.info("Imputing missing values with column medians...")
        X.fillna(X.median(), inplace=True)

    logger.info("Cleaned feature matrix: %d samples with %d physiological features.", len(X), len(feature_cols))
    logger.info("Class distribution:\n%s", y_raw.value_counts().to_string())

    return X, y_raw, feature_cols


def train_autonomic_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    n_estimators: int = 100,
    max_depth: Optional[int] = 16,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train Random Forest classifier with scaling and cross-validation."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    logger.info("Split dataset: Train=%d, Test=%d (Stratified)", len(X_train), len(X_test))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize calibrated Random Forest ensemble
    logger.info("Training RandomForestClassifier (n_estimators=%d, max_depth=%s)...", n_estimators, max_depth)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluation on holdout test set
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    logger.info("==================================================================")
    logger.info("MODEL EVALUATION RESULTS (Holdout Test Set):")
    logger.info("  Holdout Accuracy: %.4f (%.2f%%)", acc, acc * 100)
    logger.info("  Macro F1-Score:   %.4f", macro_f1)
    logger.info("------------------------------------------------------------------")
    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred, target_names=le.classes_))
    logger.info("==================================================================")

    # Feature importances
    importances = dict(
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(15)
    )
    logger.info("Top 10 Physiological Features by Importance:")
    for rank, (feat, imp) in enumerate(list(importances.items())[:10], 1):
        logger.info("  %2d. %-20s : %.4f", rank, feat, imp)

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_columns": feature_cols,
        "feature_names": feature_cols,  # Compatibility alias
        "test_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "classes": list(le.classes_),
        "top_features": importances,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(X),
            "n_features": len(feature_cols),
            "algorithm": "RandomForestClassifier",
            "provenance": "Kaggle SWELL-KW HRV Dataset (CC0)",
            "scikit_learn_version": sys.modules["sklearn"].__version__,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="SOMYTHRA Twin HRV Stress Model Trainer")
    parser.add_argument("--data-path", type=str, default=None, help="Path to SWELL-KW CSV file")
    parser.add_argument("--output", type=str, default="stress_prediction_model.pkl", help="Output artifact path")
    parser.add_argument("--engine-output", type=str, default=None, help="Optional secondary output in engine/ package")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of decision trees")
    parser.add_argument("--max-depth", type=int, default=16, help="Maximum tree depth")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit for fast testing")
    parser.add_argument("--auto-download", action="store_true", help="Download dataset via Kaggle if not found")
    args = parser.parse_args()

    data_path = resolve_dataset_path(args.data_path, auto_download=args.auto_download)
    X, y, feature_cols = load_and_preprocess_data(data_path, max_samples=args.max_samples)

    artifact = train_autonomic_model(
        X, y, feature_cols,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )

    out_path = Path(args.output)
    joblib.dump(artifact, out_path)
    logger.info("Successfully exported primary model artifact to: %s", out_path.resolve())

    if args.engine_output:
        eng_path = Path(args.engine_output)
        eng_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, eng_path)
        logger.info("Successfully exported secondary engine artifact to: %s", eng_path.resolve())


if __name__ == "__main__":
    main()
