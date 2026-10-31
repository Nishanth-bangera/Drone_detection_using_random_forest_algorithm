import csv
import numpy as np

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import LeaveOneOut, cross_val_predict

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

INPUT_CSV = Path(
    "results/features/recording_features.csv"
)

OUTPUT_DIR = Path(
    "results/model_comparison"
)

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
# 2. LOAD DATASET
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

            category = row.get("category")
            label = row.get("label")

            if category not in CATEGORIES:
                continue

            if label not in CLASSES:
                continue

            try:

                features = [
                    float(row[name])
                    for name in FEATURES
                ]

                if not np.all(np.isfinite(features)):
                    raise ValueError(
                        "Invalid feature values"
                    )

                recordings.append({

                    "file": row["file"],

                    "category": category,

                    "label": label,

                    "features": features

                })

            except (ValueError, KeyError) as error:

                print(
                    f"Skipping invalid row: {error}"
                )

    return recordings


# ==================================================
# 3. DEFINE THREE ML MODELS
# ==================================================

def create_models():

    models = {

        "Logistic Regression": Pipeline([

            ("scaler", StandardScaler()),

            ("classifier", LogisticRegression(
                C=0.1,
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))

        ]),

        "Random Forest": Pipeline([

            ("scaler", StandardScaler()),

            ("classifier", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))

        ]),

        "SVM": Pipeline([

            ("scaler", StandardScaler()),

            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced"
            ))

        ])

    }

    return models


# ==================================================
# 4. EVALUATE ONE MODEL
# ==================================================

def evaluate_model(model, X, y):

    loo = LeaveOneOut()

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=loo
    )

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

    matrix = confusion_matrix(
        y,
        predictions,
        labels=CLASSES
    )

    metrics = {

        "accuracy": accuracy,

        "balanced_accuracy": balanced_accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "true_negative": int(matrix[0, 0]),

        "false_positive": int(matrix[0, 1]),

        "false_negative": int(matrix[1, 0]),

        "true_positive": int(matrix[1, 1])

    }

    return metrics, predictions, matrix


# ==================================================
# 5. EVALUATE EACH CATEGORY
# ==================================================

def compare_algorithms(recordings):

    models = create_models()

    all_metrics = []

    all_predictions = []

    for category in CATEGORIES:

        selected = [

            record for record in recordings

            if record["category"] == category

        ]

        print("\n" + "=" * 70)

        print(
            f"{category.upper()} - ALGORITHM COMPARISON"
        )

        print("=" * 70)

        if len(selected) < 4:

            print("Insufficient recordings.")

            continue

        X = np.array([

            record["features"]

            for record in selected

        ], dtype=np.float64)

        y = np.array([

            record["label"]

            for record in selected

        ])

        if set(y) != set(CLASSES):

            print(
                "Both Human and Without Human classes are required."
            )

            continue

        print(
            f"Total recordings: {len(selected)}"
        )

        print(
            "Human:",
            np.sum(y == "human")
        )

        print(
            "Without Human:",
            np.sum(y == "without_human")
        )

        # Compare all three algorithms

        for model_name, model in models.items():

            print("\n--------------------------------------")

            print("Algorithm:", model_name)

            metrics, predictions, matrix = evaluate_model(
                model,
                X,
                y
            )

            print("\nRESULTS")

            for name, value in metrics.items():

                if isinstance(value, float):

                    print(
                        f"{name}: {value:.4f}"
                    )

            print("\nCONFUSION MATRIX")

            print(
                "Class order:",
                CLASSES
            )

            print(matrix)

            # Save metrics

            metric_row = {

                "category": category,

                "algorithm": model_name,

                "total_recordings": len(selected)

            }

            metric_row.update(metrics)

            all_metrics.append(metric_row)

            # Save individual predictions

            for index, record in enumerate(selected):

                all_predictions.append({

                    "category": category,

                    "algorithm": model_name,

                    "file": record["file"],

                    "actual": y[index],

                    "predicted": predictions[index],

                    "correct": bool(
                        y[index] == predictions[index]
                    )

                })

    return all_metrics, all_predictions


# ==================================================
# 6. SAVE CSV RESULTS
# ==================================================

def save_csv(filename, rows):

    if not rows:

        print(
            f"No results to save: {filename}"
        )

        return

    columns = sorted({

        key

        for row in rows

        for key in row

    })

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

        writer.writerows(rows)

    print(
        "\nSaved:",
        filename
    )


# ==================================================
# 7. DISPLAY FINAL COMPARISON
# ==================================================

def display_comparison(metrics):

    print("\n" + "=" * 85)

    print("FINAL ALGORITHM COMPARISON")

    print("=" * 85)

    print(
        f"{'Category':<12}"
        f"{'Algorithm':<23}"
        f"{'Accuracy':<13}"
        f"{'Precision':<13}"
        f"{'Recall':<13}"
        f"{'F1 Score':<13}"
    )

    print("-" * 85)

    for row in metrics:

        print(

            f"{row['category']:<12}"

            f"{row['algorithm']:<23}"

            f"{row['accuracy']:<13.4f}"

            f"{row['precision']:<13.4f}"

            f"{row['recall']:<13.4f}"

            f"{row['f1_score']:<13.4f}"

        )


# ==================================================
# 8. MAIN FUNCTION
# ==================================================

def main():

    print("\n======================================")

    print("RADAR HUMAN DETECTION")

    print("ML ALGORITHM COMPARISON")

    print("======================================")

    recordings = load_dataset()

    print(
        "\nTotal valid recordings:",
        len(recordings)
    )

    all_metrics, all_predictions = (
        compare_algorithms(recordings)
    )

    # Save metrics

    metrics_file = (
        OUTPUT_DIR /
        "algorithm_comparison_metrics.csv"
    )

    save_csv(
        metrics_file,
        all_metrics
    )

    # Save predictions

    predictions_file = (
        OUTPUT_DIR /
        "algorithm_comparison_predictions.csv"
    )

    save_csv(
        predictions_file,
        all_predictions
    )

    # Display final table

    display_comparison(
        all_metrics
    )

    print("\n======================================")

    print("ALGORITHM COMPARISON COMPLETED")

    print("======================================")

    print("Models tested: 3")

    print("Categories tested: 3")

    print("Validation: Leave-One-Recording-Out")

    print("\nOutput folder:")

    print(OUTPUT_DIR)


# ==================================================
# 9. EXECUTION
# ==================================================

if __name__ == "__main__":

    main()