#!/usr/bin/env python3
"""
SOMYTHRA Twin — Optical PPG Autonomic Stress Model Training Pipeline
Trained on the WESAD Wearable Optical PPG Benchmark Dataset (Empatica E4 wristband).

Unlike chest-electrode ECG models, this model is trained strictly on optical
photoplethysmography (PPG) features directly accessible via remote rPPG / wearables:
  1. mean_hr  : Mean pulse rate in beats per minute (BPM)
  2. rmssd    : Root mean square of successive pulse-to-pulse differences (ms)
  3. pnn50    : Percentage of successive intervals differing by > 50ms (%)
  4. lf_hf    : Ratio of low-frequency (0.04-0.15 Hz) to high-frequency (0.15-0.4 Hz) power

Cross-Validation:
  Evaluated using rigorous 15-fold Leave-One-Subject-Out (LOSO) cross-validation
  across all 15 human subjects (S2 through S17) to guarantee zero data leakage.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("somythra.twin.wesad")

OPTICAL_FEATURES = ["mean_hr", "rmssd", "pnn50", "lf_hf"]

DEFAULT_WESAD_PATHS = [
    Path("data/wesad/features.csv"),
    Path("/tmp/wesad_wrist/features.csv"),
    Path("features.csv"),
]


def resolve_wesad_dataset(path: Optional[str] = None, auto_download: bool = False) -> Path:
    """Find local WESAD wrist features dataset or download via Kaggle CLI."""
    if path:
        p = Path(path)
        if p.exists():
            return p
        logger.warning("Provided path '%s' not found. Checking standard locations...", path)

    for cand in DEFAULT_WESAD_PATHS:
        if cand.exists():
            logger.info("Found WESAD optical dataset at '%s'", cand)
            return cand

    if auto_download:
        logger.info("Downloading WESAD engineered wrist features from Kaggle...")
        target_dir = Path("data/wesad")
        target_dir.mkdir(parents=True, exist_ok=True)
        cmd = f'kaggle datasets download vladafed/wesad-engineered-wrist-features -p "{target_dir}" --unzip'
        res = os.system(cmd)
        target_csv = target_dir / "features.csv"
        if res == 0 and target_csv.exists():
            return target_csv

    raise FileNotFoundError(
        "WESAD optical wrist dataset not found. Pass --data-path or --auto-download."
    )


def evaluate_loso(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 6,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run 15-fold Leave-One-Subject-Out (LOSO) cross-validation.
    Each human subject is tested on a model trained entirely without their data.
    """
    logo = LeaveOneGroupOut()
    subject_results = []
    y_true_all, y_pred_all = [], []

    subjects = groups.unique()
    logger.info("Starting Leave-One-Subject-Out CV across %d human subjects...", len(subjects))

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        test_subject = groups.iloc[test_idx].iloc[0]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        probs = clf.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan

        subject_results.append({
            "subject": test_subject,
            "accuracy": acc,
            "f1": f1,
            "roc_auc": auc,
            "n_samples": len(y_test),
            "stress_ratio": float(np.mean(y_test)),
        })

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    df_res = pd.DataFrame(subject_results)
    mean_acc = float(df_res["accuracy"].mean())
    mean_f1 = float(df_res["f1"].mean())
    mean_auc = float(df_res["roc_auc"].dropna().mean())

    logger.info("==================================================================")
    logger.info("LEAVE-ONE-SUBJECT-OUT (LOSO) VALIDATION RESULTS:")
    logger.info("  Mean Accuracy Across New Humans: %.2f%%", mean_acc * 100)
    logger.info("  Mean F1-Score:                   %.4f", mean_f1)
    logger.info("  Mean ROC-AUC:                    %.4f", mean_auc)
    logger.info("  Overall Confusion Matrix:\n%s", confusion_matrix(y_true_all, y_pred_all))
    logger.info("==================================================================")

    return {
        "mean_accuracy": mean_acc,
        "mean_f1": mean_f1,
        "mean_roc_auc": mean_auc,
        "per_subject": df_res.to_dict(orient="records"),
    }


def train_production_optical_model(
    data_path: Path,
    output_path: Path,
    n_estimators: int = 100,
    max_depth: int = 6,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train and export production-ready optical stress classifier."""
    df = pd.read_csv(data_path)
    X = df[OPTICAL_FEATURES].copy()
    y = (df["label_name"].astype(str).str.lower() == "stress").astype(int).values
    groups = df["subject"]

    # 1. Run rigorous LOSO validation
    loso_eval = evaluate_loso(
        X, y, groups, n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )

    # 2. Fit final production model on all 15 subjects
    logger.info("Fitting final production model on full 15-subject cohort (%d samples)...", len(X))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    prod_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    prod_model.fit(X_scaled, y)

    importances = dict(zip(OPTICAL_FEATURES, [float(v) for v in prod_model.feature_importances_]))
    logger.info("Feature Importances: %s", importances)

    artifact = {
        "model": prod_model,
        "scaler": scaler,
        "feature_columns": OPTICAL_FEATURES,
        "feature_names": OPTICAL_FEATURES,
        "classes": ["baseline", "stress"],
        "class_labels": {0: "baseline", 1: "stress"},
        "loso_validation": loso_eval,
        "loso_mean_accuracy": loso_eval["mean_accuracy"],
        "loso_mean_f1": loso_eval["mean_f1"],
        "loso_mean_roc_auc": loso_eval["mean_roc_auc"],
        "feature_importances": importances,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(X),
            "n_subjects": int(groups.nunique()),
            "sensor_modality": "Optical Photoplethysmography (BVP/PPG)",
            "validation_protocol": "15-Fold Leave-One-Subject-Out (LOSO)",
            "scikit_learn_version": sys.modules["sklearn"].__version__,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    logger.info("Successfully exported optical model artifact to: %s", output_path.resolve())

    return artifact


def main():
    parser = argparse.ArgumentParser(description="SOMYTHRA Twin Optical PPG Stress Model Trainer")
    parser.add_argument("--data-path", type=str, default=None, help="Path to WESAD wrist features.csv")
    parser.add_argument("--output", type=str, default="somythra_optical_stress_model.pkl", help="Output artifact path")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth")
    parser.add_argument("--auto-download", action="store_true", help="Download from Kaggle if missing")
    args = parser.parse_args()

    data_path = resolve_wesad_dataset(args.data_path, auto_download=args.auto_download)
    out_path = Path(args.output)

    train_production_optical_model(
        data_path=data_path,
        output_path=out_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main()
