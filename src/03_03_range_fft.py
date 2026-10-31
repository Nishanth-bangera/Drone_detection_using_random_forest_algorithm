import numpy as np
import matplotlib.pyplot as plt
import csv

from pathlib import Path
from collections import defaultdict


# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_ROOT = Path("results/decoded")

OUTPUT_ROOT = Path("results/features")

GRAPH_ROOT = OUTPUT_ROOT / "graphs"

RADAR_CONFIGS = {
    "srr": {
        "chirps_per_frame": 128
    },
    "usrr": {
        "chirps_per_frame": 64
    },
    "indoor": {
        "chirps_per_frame": 128
    }
}

CATEGORIES = ["srr", "usrr", "indoor"]

CLASSES = ["human", "without_human"]

EPSILON = 1e-10


# ==================================================
# 2. COMPUTE RANGE-TIME MATRIX
# ==================================================

def calculate_range_time(file_path, category):

    data = np.load(
        file_path,
        mmap_mode="r"
    )

    if data.ndim != 3:
        raise ValueError(
            f"Invalid radar shape: {data.shape}"
        )

    if not np.iscomplexobj(data):
        raise ValueError(
            "Expected complex radar data."
        )

    num_chirps, num_rx, num_samples = data.shape

    chirps_per_frame = (
        RADAR_CONFIGS[category]["chirps_per_frame"]
    )

    num_frames = (
        num_chirps + chirps_per_frame - 1
    ) // chirps_per_frame

    half_bins = num_samples // 2

    window = np.hanning(
        num_samples
    ).astype(np.float32)

    range_time = np.zeros(
        (num_frames, half_bins),
        dtype=np.float32
    )

    for frame in range(num_frames):

        start = frame * chirps_per_frame

        end = min(
            start + chirps_per_frame,
            num_chirps
        )

        frame_data = np.asarray(
            data[start:end],
            dtype=np.complex64
        )

        windowed = (
            frame_data *
            window[None, None, :]
        )

        fft_result = np.fft.fft(
            windowed,
            axis=2
        )

        magnitude = np.abs(
            fft_result[:, :, :half_bins]
        )

        range_time[frame] = np.mean(
            magnitude,
            axis=(0, 1)
        )

    return range_time


# ==================================================
# 3. EXTRACT FEATURES
# ==================================================

def extract_features(range_time):

    # Average reflection strength at each range bin

    bin_mean = np.mean(
        range_time,
        axis=0
    )

    # Standard deviation over time

    bin_std = np.std(
        range_time,
        axis=0
    )

    # Coefficient of variation for every range bin

    bin_cv = (
        bin_std /
        np.maximum(bin_mean, EPSILON)
    )

    # Frame-to-frame absolute difference

    if range_time.shape[0] > 1:

        frame_difference = np.abs(
            np.diff(
                range_time,
                axis=0
            )
        )

        bin_frame_difference = np.mean(
            frame_difference,
            axis=0
        )

        mean_frame_difference = float(
            np.mean(frame_difference)
        )

    else:

        bin_frame_difference = np.zeros(
            range_time.shape[1]
        )

        mean_frame_difference = 0.0

    # Mean squared magnitude

    bin_energy = np.mean(
        range_time ** 2,
        axis=0
    )

    # Ignore DC bin for the primary range analysis

    analysis_mean = bin_mean[1:]

    analysis_std = bin_std[1:]

    analysis_cv = bin_cv[1:]

    analysis_difference = (
        bin_frame_difference[1:]
    )

    analysis_energy = bin_energy[1:]

    # Identify strongest range bin

    peak_bin = int(
        np.argmax(analysis_mean) + 1
    )

    peak_magnitude = float(
        bin_mean[peak_bin]
    )

    # Identify active range bins

    maximum_mean = np.max(
        analysis_mean
    )

    active_threshold = (
        0.10 * maximum_mean
    )

    active_mask = (
        analysis_mean >= active_threshold
    )

    if np.any(active_mask):

        active_cv = analysis_cv[active_mask]

        active_bins = int(
            np.sum(active_mask)
        )

        mean_active_cv = float(
            np.mean(active_cv)
        )

    else:

        active_bins = 0

        mean_active_cv = 0.0

    features = {

        "mean_magnitude_all_bins": float(
            np.mean(range_time)
        ),

        "std_magnitude_all_bins": float(
            np.std(range_time)
        ),

        "peak_range_bin": peak_bin,

        "peak_magnitude": peak_magnitude,

        "mean_temporal_std": float(
            np.mean(analysis_std)
        ),

        "max_temporal_std": float(
            np.max(analysis_std)
        ),

        "mean_temporal_cv": float(
            np.mean(analysis_cv)
        ),

        "active_bin_temporal_cv": mean_active_cv,

        "mean_frame_difference": mean_frame_difference,

        "mean_square_energy": float(
            np.mean(analysis_energy)
        ),

        "active_range_bin_count": active_bins,

        "total_frames": range_time.shape[0],

        "total_range_bins": range_time.shape[1]

    }

    # Per-range-bin feature information

    per_bin = []

    for bin_index in range(
        range_time.shape[1]
    ):

        per_bin.append({

            "range_bin": bin_index,

            "mean_magnitude": float(
                bin_mean[bin_index]
            ),

            "temporal_std": float(
                bin_std[bin_index]
            ),

            "temporal_cv": float(
                bin_cv[bin_index]
            ),

            "mean_frame_difference": float(
                bin_frame_difference[bin_index]
            ),

            "mean_square_energy": float(
                bin_energy[bin_index]
            )

        })

    return features, per_bin


# ==================================================
# 4. SAVE CSV FILE
# ==================================================

def save_csv(file_path, rows):

    if not rows:
        return

    columns = sorted({

        key

        for row in rows

        for key in row.keys()

    })

    with open(
        file_path,
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


# ==================================================
# 5. AGGREGATE FEATURE VALUES
# ==================================================

def aggregate_features(rows, group_keys, metric_keys):

    groups = defaultdict(list)

    for row in rows:

        key = tuple(
            row[k] for k in group_keys
        )

        groups[key].append(row)

    aggregated = []

    for key, group in groups.items():

        result = dict(
            zip(group_keys, key)
        )

        result["recording_count"] = len(group)

        for metric in metric_keys:

            values = np.array(
                [
                    float(row[metric])
                    for row in group
                    if metric in row
                    and row[metric] != ""
                ],
                dtype=float
            )

            if len(values):

                result[f"{metric}_mean"] = float(
                    np.mean(values)
                )

                result[f"{metric}_std"] = float(
                    np.std(values)
                )

        aggregated.append(result)

    return aggregated


# ==================================================
# 6. PLOT RANGE-BIN TEMPORAL CV
# ==================================================

def plot_range_bin_comparison(per_bin_rows, category):

    plt.figure(figsize=(12, 5))

    for label in CLASSES:

        selected = [

            row for row in per_bin_rows

            if row["category"] == category

            and row["label"] == label

        ]

        if not selected:
            continue

        bin_groups = defaultdict(list)

        for row in selected:

            bin_groups[
                row["range_bin"]
            ].append(
                row["temporal_cv"]
            )

        bins = sorted(bin_groups)

        averages = [

            np.mean(bin_groups[b])

            for b in bins

        ]

        plt.plot(
            bins,
            averages,
            label=label
        )

    plt.title(
        f"{category.upper()} - Temporal CV by Range Bin"
    )

    plt.xlabel("Range Bin")

    plt.ylabel("Average Temporal CV")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    output_file = (
        GRAPH_ROOT /
        f"{category}_range_bin_temporal_cv.png"
    )

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    print(
        "Saved feature graph:",
        output_file
    )


# ==================================================
# 7. PLOT FEATURE COMPARISONS
# ==================================================

def plot_feature_comparison(recording_rows, feature):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    for index, category in enumerate(CATEGORIES):

        ax = axes[index]

        for class_index, label in enumerate(CLASSES):

            values = [

                row[feature]

                for row in recording_rows

                if row["category"] == category

                and row["label"] == label

            ]

            if not values:
                continue

            x_position = (
                class_index + 1
            )

            # Show every recording separately

            x_values = np.full(
                len(values),
                x_position,
                dtype=float
            )

            x_values += np.linspace(
                -0.06,
                0.06,
                len(values)
            )

            ax.scatter(
                x_values,
                values
            )

            ax.plot(
                [x_position - 0.15, x_position + 0.15],
                [np.mean(values), np.mean(values)]
            )

        ax.set_title(
            category.upper()
        )

        ax.set_xticks([1, 2])

        ax.set_xticklabels(
            ["Human", "Without Human"]
        )

        ax.set_ylabel(feature)

        ax.grid(True)

    fig.suptitle(
        f"Human vs Without Human: {feature}"
    )

    fig.tight_layout()

    output_file = (
        GRAPH_ROOT /
        f"comparison_{feature}.png"
    )

    fig.savefig(
        output_file,
        dpi=150
    )

    plt.close(fig)

    print(
        "Saved comparison:",
        output_file
    )


# ==================================================
# 8. MAIN FUNCTION
# ==================================================

def main():

    if not DATA_ROOT.exists():

        raise FileNotFoundError(
            "Decoded radar data not found. "
            "Run 02_adc_decode.py first."
        )

    files = sorted(
        DATA_ROOT.rglob("*.npy")
    )

    print("\n======================================")

    print("RANGE-TIME FEATURE EXTRACTION")

    print("Total files:", len(files))

    print("======================================")

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    GRAPH_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    recording_rows = []

    per_bin_rows = []

    successful = 0

    failed = 0

    # --------------------------------------------------
    # Process every recording
    # --------------------------------------------------

    for file_path in files:

        try:

            parts = [

                part.lower()

                for part in file_path.relative_to(
                    DATA_ROOT
                ).parts

            ]

            category = next(

                (
                    part for part in parts
                    if part in CATEGORIES
                ),

                None

            )

            label = next(

                (
                    part for part in parts
                    if part in CLASSES
                ),

                None

            )

            if category is None or label is None:

                print(
                    "Skipping unrecognized file:",
                    file_path
                )

                continue

            print("\n--------------------------------------")

            print("File:", file_path.name)

            print("Category:", category.upper())

            print("Class:", label)

            # Calculate Range-Time

            range_time = calculate_range_time(
                file_path,
                category
            )

            print(
                "Range-Time shape:",
                range_time.shape
            )

            # Extract numerical features

            features, bin_features = extract_features(
                range_time
            )

            print("\nFEATURE RESULTS")

            for name, value in features.items():

                print(
                    f"{name}: {value}"
                )

            # Store recording-level features

            row = {

                "file": str(file_path),

                "category": category,

                "label": label

            }

            row.update(features)

            recording_rows.append(row)

            # Store every range-bin feature

            for bin_row in bin_features:

                bin_row.update({

                    "file": str(file_path),

                    "category": category,

                    "label": label

                })

                per_bin_rows.append(bin_row)

            successful += 1

        except Exception as error:

            failed += 1

            print(
                "[ERROR]",
                file_path,
                error
            )

    # --------------------------------------------------
    # Save recording feature CSV
    # --------------------------------------------------

    recording_csv = (
        OUTPUT_ROOT /
        "recording_features.csv"
    )

    save_csv(
        recording_csv,
        recording_rows
    )

    # --------------------------------------------------
    # Save range-bin feature CSV
    # --------------------------------------------------

    bin_csv = (
        OUTPUT_ROOT /
        "range_bin_features.csv"
    )

    save_csv(
        bin_csv,
        per_bin_rows
    )

    # --------------------------------------------------
    # Aggregate class-level features
    # --------------------------------------------------

    metric_keys = [

        "mean_magnitude_all_bins",

        "std_magnitude_all_bins",

        "peak_magnitude",

        "mean_temporal_std",

        "max_temporal_std",

        "mean_temporal_cv",

        "active_bin_temporal_cv",

        "mean_frame_difference",

        "mean_square_energy",

        "active_range_bin_count"

    ]

    class_summary = aggregate_features(

        recording_rows,

        ["category", "label"],

        metric_keys

    )

    class_csv = (
        OUTPUT_ROOT /
        "class_feature_summary.csv"
    )

    save_csv(
        class_csv,
        class_summary
    )

    # --------------------------------------------------
    # Aggregate range-bin features by class
    # --------------------------------------------------

    range_bin_summary = aggregate_features(

        per_bin_rows,

        ["category", "label", "range_bin"],

        [

            "mean_magnitude",

            "temporal_std",

            "temporal_cv",

            "mean_frame_difference",

            "mean_square_energy"

        ]

    )

    range_bin_csv = (
        OUTPUT_ROOT /
        "range_bin_class_summary.csv"
    )

    save_csv(
        range_bin_csv,
        range_bin_summary
    )

    # --------------------------------------------------
    # Generate graphs
    # --------------------------------------------------

    print("\n======================================")

    print("GENERATING FEATURE COMPARISON GRAPHS")

    print("======================================")

    for category in CATEGORIES:

        plot_range_bin_comparison(
            per_bin_rows,
            category
        )

    important_features = [

        "active_bin_temporal_cv",

        "mean_frame_difference",

        "mean_temporal_std",

        "mean_square_energy"

    ]

    for feature in important_features:

        plot_feature_comparison(
            recording_rows,
            feature
        )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n======================================")

    print("FEATURE EXTRACTION COMPLETED")

    print("======================================")

    print("Successful files:", successful)

    print("Failed files:", failed)

    print(
        "\nRecording features:",
        recording_csv
    )

    print(
        "Range-bin features:",
        bin_csv
    )

    print(
        "Class summary:",
        class_csv
    )

    print(
        "Range-bin class summary:",
        range_bin_csv
    )

    print(
        "Graphs:",
        GRAPH_ROOT
    )


# ==================================================
# 9. EXECUTION
# ==================================================

if __name__ == "__main__":

    main()