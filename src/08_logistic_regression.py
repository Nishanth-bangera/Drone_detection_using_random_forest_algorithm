import csv
import numpy as np

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import LeaveOneOut, cross_val_predict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

INPUT_CSV = Path(
    "results/features/recording_features.csv"
)

OUTPUT_DIR = Path("results/model")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CATEGORIES = [
    "srr",
    "usrr",
    "indoor"
]

CLASSES = [
    "without_human",
    "human"
]

FEATURES = [
    "active_bin_temporal_cv",
    "mean_frame_difference",
    "mean_temporal_std"
]


# ==================================================
# 2. LOAD FEATURE DATASET
# ==================================================

def load_dataset():

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Feature CSV not found: {INPUT_CSV}"
        )

    recordings = []

    with open(
        INPUT_CSV,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row.get("category") not in CATEGORIES:
                continue

            if row.get("label") not in CLASSES:
                continue

            if row.get("status", "SUCCESS") == "FAILED":
                continue

            try:

                feature_values = [
                    float(row[name])
                    for name in FEATURES
                ]

                if not np.all(
                    np.isfinite(feature_values)
                ):
                    raise ValueError(
                        "Non-finite feature value"
                    )

                recordings.append({

                    "file": row["file"],

                    "category": row["category"],

                    "label": row["label"],

                    "features": feature_values

                })

            except (ValueError, KeyError) as error:

                print(
                    f"Skipping invalid row: {error}"
                )

    return recordings


# ==================================================
# 3. BUILD MACHINE LEARNING MODEL
# ==================================================

def create_model():

    model = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                C=0.1,
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )

    ])

    return model


# ==================================================
# 4. TRAIN AND VALIDATE EACH CATEGORY
# ==================================================

def evaluate_category(recordings, category):

    selected = [

        recording

        for recording in recordings

        if recording["category"] == category

    ]

    print("\n" + "=" * 65)

    print(
        f"{category.upper()} CLASSIFICATION"
    )

    print("=" * 65)

    if len(selected) < 4:

        print(
            "Insufficient recordings for evaluation."
        )

        return None, []

    X = np.array(
        [
            recording["features"]
            for recording in selected
        ],
        dtype=np.float64
    )

    y = np.array(
        [
            recording["label"]
            for recording in selected
        ]
    )

    # Check class distribution

    unique_classes, counts = np.unique(
        y,
        return_counts=True
    )

    print("\nDATASET INFORMATION")

    print("Total recordings:", len(y))

    for label, count in zip(
        unique_classes,
        counts
    ):

        print(
            f"{label}: {count}"
        )

    if len(unique_classes) != 2:

        print(
            "Both classes are required."
        )

        return None, []

    # --------------------------------------------------
    # Leave-One-Recording-Out Validation
    # --------------------------------------------------

    print(
        "\nStarting Leave-One-Recording-Out Validation..."
    )

    model = create_model()

    loo = LeaveOneOut()

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=loo
    )

    # --------------------------------------------------
    # Evaluation Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        pos_label="human",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        pos_label="human",
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        pos_label="human",
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=CLASSES
    )

    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------

    print("\nCLASSIFICATION RESULTS")

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced_accuracy:.4f}"
    )

    print(
        f"Precision         : {precision:.4f}"
    )

    print(
        f"Recall            : {recall:.4f}"
    )

    print(
        f"F1 Score          : {f1:.4f}"
    )

    print("\nCONFUSION MATRIX")

    print("Rows = Actual class")
    print("Columns = Predicted class")

    print("Class order:", CLASSES)

    print(cm)

    # --------------------------------------------------
    # Save Individual Predictions
    # --------------------------------------------------

    prediction_rows = []

    for index, recording in enumerate(selected):

        prediction_rows.append({

            "file": recording["file"],

            "category": category,

            "actual": y[index],

            "predicted": predictions[index],

            "correct": (
                y[index] == predictions[index]
            )

        })

    metrics = {

        "category": category,

        "total_recordings": len(y),

        "accuracy": accuracy,

        "balanced_accuracy": balanced_accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "true_negative": int(cm[0, 0]),

        "false_positive": int(cm[0, 1]),

        "false_negative": int(cm[1, 0]),

        "true_positive": int(cm[1, 1])

    }

    return metrics, prediction_rows


# ==================================================
# 5. MAIN FUNCTION
# ==================================================

def main():

    print("\n======================================")

    print("RADAR HUMAN DETECTION")

    print("BASELINE MACHINE LEARNING")

    print("======================================")

    # Load features

    recordings = load_dataset()

    print(
        "\nTotal valid recordings:",
        len(recordings)
    )

    all_metrics = []

    all_predictions = []

    # Evaluate each radar category separately

    for category in CATEGORIES:

        metrics, predictions = evaluate_category(
            recordings,
            category
        )

        if metrics is not None:

            all_metrics.append(metrics)

            all_predictions.extend(
                predictions
            )

    # --------------------------------------------------
    # Save Metrics
    # --------------------------------------------------

    metrics_file = (
        OUTPUT_DIR /
        "baseline_metrics.csv"
    )

    if all_metrics:

        with open(
            metrics_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    all_metrics[0].keys()
                )
            )

            writer.writeheader()

            writer.writerows(
                all_metrics
            )

    # --------------------------------------------------
    # Save Predictions
    # --------------------------------------------------

    prediction_file = (
        OUTPUT_DIR /
        "baseline_predictions.csv"
    )

    if all_predictions:

        with open(
            prediction_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    all_predictions[0].keys()
                )
            )

            writer.writeheader()

            writer.writerows(
                all_predictions
            )

    # --------------------------------------------------
    # Final Summary
    # --------------------------------------------------

    print("\n" + "=" * 65)

    print("BASELINE CLASSIFICATION COMPLETED")

    print("=" * 65)

    for result in all_metrics:

        print(
            f"{result['category'].upper()} "
            f"Accuracy: {result['accuracy']:.4f}"
        )

    print("\nSaved files:")

    print(metrics_file)

    print(prediction_file)

    print(
        "\nNEXT STEP: Evaluate the model results."
    )


# ==================================================
# 6. EXECUTION
# ==================================================

if __name__ == "__main__":

    main()