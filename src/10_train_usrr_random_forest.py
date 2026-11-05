import csv
import joblib
import numpy as np

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import LeaveOneOut, cross_val_predict

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

INPUT_CSV = Path("results/features/recording_features.csv")

MODEL_PATH = Path("results/models/usrr_random_forest.joblib")

CATEGORY = "usrr"

FEATURES = [
    "active_bin_temporal_cv",
    "mean_frame_difference",
    "mean_temporal_std"
]


# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_dataset():

    recordings = []

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Feature CSV not found: {INPUT_CSV}"
        )

    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["category"] != CATEGORY:
                continue

            values = [
                float(row[name])
                for name in FEATURES
            ]

            if not np.all(np.isfinite(values)):
                raise ValueError("Invalid feature values found.")

            recordings.append({
                "file": row["file"],
                "label": row["label"],
                "features": values
            })

    return recordings


# ==========================================
# 3. CREATE RANDOM FOREST MODEL
# ==========================================

def create_model():

    model = Pipeline([

        ("scaler", StandardScaler()),

        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))

    ])

    return model


# ==========================================
# 4. MAIN FUNCTION
# ==========================================

def main():

    print("\n======================================")
    print("USRR RANDOM FOREST TRAINING")
    print("======================================")

    recordings = load_dataset()

    X = np.array([
        record["features"]
        for record in recordings
    ], dtype=np.float64)

    y = np.array([
        record["label"]
        for record in recordings
    ])

    print("\nDATASET INFORMATION")

    print("Total recordings:", len(recordings))
    print("Human:", np.sum(y == "human"))
    print("Without Human:", np.sum(y == "without_human"))

    if len(recordings) < 4:
        raise ValueError("Insufficient training recordings.")

    if set(y) != {"human", "without_human"}:
        raise ValueError("Both classes are required.")

    # ======================================
    # 5. LEAVE-ONE-OUT VALIDATION
    # ======================================

    print("\nStarting Leave-One-Recording-Out Validation...")

    model = create_model()

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=LeaveOneOut()
    )

    accuracy = accuracy_score(y, predictions)

    balanced_accuracy = balanced_accuracy_score(y, predictions)

    matrix = confusion_matrix(
        y,
        predictions,
        labels=["without_human", "human"]
    )

    print("\nVALIDATION RESULTS")

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(
        f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%"
    )

    print("\nCONFUSION MATRIX")

    print("Class order: without_human, human")
    print(matrix)

    # ======================================
    # 6. TRAIN FINAL MODEL
    # ======================================

    print("\nTraining final Random Forest model...")

    model.fit(X, y)

    # ======================================
    # 7. SAVE TRAINED MODEL
    # ======================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model_bundle = {
        "model": model,
        "features": FEATURES,
        "category": CATEGORY
    }

    joblib.dump(model_bundle, MODEL_PATH)

    print("\n======================================")
    print("MODEL TRAINING COMPLETED")
    print("======================================")

    print("Saved model:", MODEL_PATH)
    print("Training recordings:", len(recordings))
    print("Features:", FEATURES)


# ==========================================
# 8. EXECUTION
# ==========================================

if __name__ == "__main__":
    main()