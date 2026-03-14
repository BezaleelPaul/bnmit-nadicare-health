import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def main():
    # Load training data
    data_path = "train.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path, nrows=10000)  # Use subset for faster training
    print(f"Loaded {len(df)} samples with {len(df.columns)} features.")

    # Assume last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split into train (75%) and test (25%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model on train set
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_scaled, y_train_encoded)

    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

    # Save artifact
    artifact = {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_columns": list(X.columns),
        "test_accuracy": accuracy
    }

    model_path = "stress_prediction_model.pkl"
    joblib.dump(artifact, model_path)
    print(f"Model saved to {model_path}")
    print(f"Classes: {le.classes_}")
    print(f"Features: {list(X.columns)}")

if __name__ == "__main__":
    main()