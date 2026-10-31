import numpy as np
import matplotlib.pyplot as plt
import csv

from pathlib import Path


# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_ROOT = Path("results/decoded")

OUTPUT_ROOT = Path("results/graphs/range_time")

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


# ==================================================
# 2. RANGE-TIME CALCULATION
# ==================================================

def calculate_range_time(npy_file, category):

    data = np.load(
        npy_file,
        mmap_mode="r"
    )

    if data.ndim != 3:

        raise ValueError(
            f"Invalid radar data shape: {data.shape}"
        )

    if not np.iscomplexobj(data):

        raise ValueError(
            "Expected complex I/Q radar data."
        )

    num_chirps, num_rx, num_samples = data.shape

    chirps_per_frame = (
        RADAR_CONFIGS[category]["chirps_per_frame"]
    )

    half_bins = num_samples // 2

    num_frames = (
        num_chirps + chirps_per_frame - 1
    ) // chirps_per_frame

    print("\nLoaded:", npy_file)

    print("Radar shape:", data.shape)

    print("Number of chirps:", num_chirps)

    print("Number of RX antennas:", num_rx)

    print("ADC samples:", num_samples)

    print("Chirps per frame:", chirps_per_frame)

    # Apply Hann window

    window = np.hanning(
        num_samples
    ).astype(np.float32)

    # Create output matrix:
    # Rows = Frames
    # Columns = Range bins

    range_time = np.zeros(
        (num_frames, half_bins),
        dtype=np.float32
    )

    # Process each frame separately

    for frame_index in range(num_frames):

        start = (
            frame_index * chirps_per_frame
        )

        end = min(
            start + chirps_per_frame,
            num_chirps
        )

        # Load current frame

        frame_data = np.asarray(
            data[start:end],
            dtype=np.complex64
        )

        # Apply window

        windowed_data = (
            frame_data * window[None, None, :]
        )

        # Perform Range FFT

        range_fft = np.fft.fft(
            windowed_data,
            axis=2
        )

        # Select positive range bins

        positive_fft = range_fft[
            :, :, :half_bins
        ]

        # Calculate magnitude

        magnitude = np.abs(
            positive_fft
        )

        # Average across chirps and RX antennas

        frame_profile = np.mean(
            magnitude,
            axis=(0, 1)
        )

        range_time[
            frame_index
        ] = frame_profile

    print(
        "Range-Time matrix shape:",
        range_time.shape
    )

    return range_time, {

        "num_chirps": num_chirps,

        "num_rx": num_rx,

        "num_adc_samples": num_samples,

        "num_frames": num_frames,

        "range_bins": half_bins

    }


# ==================================================
# 3. SAVE RANGE-TIME HEATMAP
# ==================================================

def save_range_time_graph(
    range_time,
    category,
    label,
    filename
):

    output_dir = (
        OUTPUT_ROOT
        / category
        / label
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / f"{filename.stem}_range_time.png"
    )

    # Convert magnitude to dB

    range_time_db = (
        20 * np.log10(
            np.maximum(range_time, 1e-12)
        )
    )

    plt.figure(figsize=(13, 6))

    plt.imshow(

        range_time_db.T,

        aspect="auto",

        origin="lower",

        interpolation="nearest",

        extent=[
            0,
            range_time.shape[0],
            0,
            range_time.shape[1]
        ]

    )

    plt.colorbar(
        label="Magnitude (dB)"
    )

    plt.title(
        f"{category.upper()} - {label} - Range-Time Map"
    )

    plt.xlabel(
        "Frame Number"
    )

    plt.ylabel(
        "Range Bin"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    return output_file


# ==================================================
# 4. HUMAN VS WITHOUT HUMAN COMPARISON
# ==================================================

def compare_human_without_human(
    category,
    profiles
):

    human_profiles = profiles["human"]

    nohuman_profiles = profiles["without_human"]

    if not human_profiles or not nohuman_profiles:

        print(
            f"Insufficient data for {category.upper()}"
        )

        return

    # All recordings in a category should have matching dimensions

    shapes = {
        profile.shape
        for profile in (
            human_profiles + nohuman_profiles
        )
    }

    if len(shapes) != 1:

        print(
            f"Cannot compare {category}: "
            "recordings have different frame/range dimensions."
        )

        return

    # Average across recordings

    human_average = np.mean(
        np.stack(human_profiles),
        axis=0
    )

    nohuman_average = np.mean(
        np.stack(nohuman_profiles),
        axis=0
    )

    # Convert to dB

    human_db = 20 * np.log10(
        np.maximum(human_average, 1e-12)
    )

    nohuman_db = 20 * np.log10(
        np.maximum(nohuman_average, 1e-12)
    )

    output_dir = OUTPUT_ROOT / category

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Use the same colour scale for both class plots

    combined = np.concatenate([
        human_db.ravel(),
        nohuman_db.ravel()
    ])

    vmin, vmax = np.percentile(
        combined,
        [2, 99]
    )

    # --------------------------------------------------
    # Human vs Without Human
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6)
    )

    maps = [
        (human_db, "Human"),
        (nohuman_db, "Without Human")
    ]

    for ax, (image_data, title) in zip(
        axes,
        maps
    ):

        plot = ax.imshow(

            image_data.T,

            aspect="auto",

            origin="lower",

            interpolation="nearest",

            extent=[
                0,
                image_data.shape[0],
                0,
                image_data.shape[1]
            ],

            vmin=vmin,

            vmax=vmax

        )

        ax.set_title(title)

        ax.set_xlabel("Frame Number")

        ax.set_ylabel("Range Bin")

    fig.colorbar(
        plot,
        ax=axes,
        label="Magnitude (dB)"
    )

    fig.suptitle(
        f"{category.upper()} - Human vs Without Human"
    )

    plt.tight_layout()

    comparison_file = (
        output_dir
        / f"{category}_human_vs_without_human.png"
    )

    plt.savefig(
        comparison_file,
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # Difference Map
    # --------------------------------------------------

    difference = (
        human_db - nohuman_db
    )

    limit = np.percentile(
        np.abs(difference),
        99
    )

    if limit <= 0:
        limit = 1e-6

    plt.figure(figsize=(12, 6))

    plt.imshow(

        difference.T,

        aspect="auto",

        origin="lower",

        interpolation="nearest",

        extent=[
            0,
            difference.shape[0],
            0,
            difference.shape[1]
        ],

        cmap="RdBu_r",

        vmin=-limit,

        vmax=limit

    )

    plt.colorbar(
        label="Human - Without Human (dB)"
    )

    plt.title(
        f"{category.upper()} - Difference Map"
    )

    plt.xlabel(
        "Frame Number"
    )

    plt.ylabel(
        "Range Bin"
    )

    plt.tight_layout()

    difference_file = (
        output_dir
        / f"{category}_difference_map.png"
    )

    plt.savefig(
        difference_file,
        dpi=150
    )

    plt.close()

    print(
        "\nComparison saved:",
        comparison_file
    )

    print(
        "Difference map saved:",
        difference_file
    )


# ==================================================
# 5. TEMPORAL ANALYSIS
# ==================================================

def calculate_temporal_metrics(range_time):

    # Average range profile across time

    average_profile = np.mean(
        range_time,
        axis=0
    )

    # Ignore DC bin when finding the strongest bin

    if len(average_profile) < 2:
        raise ValueError("Not enough range bins.")

    peak_bin = int(
        np.argmax(
            average_profile[1:]
        ) + 1
    )

    # Magnitude variation at strongest range bin

    temporal_signal = range_time[:, peak_bin]

    temporal_mean = np.mean(
        temporal_signal
    )

    temporal_std = np.std(
        temporal_signal
    )

    temporal_cv = (
        temporal_std / temporal_mean
        if temporal_mean > 0
        else 0
    )

    return {

        "strongest_range_bin": peak_bin,

        "mean_magnitude": float(
            temporal_mean
        ),

        "minimum_magnitude": float(
            np.min(temporal_signal)
        ),

        "maximum_magnitude": float(
            np.max(temporal_signal)
        ),

        "temporal_std": float(
            temporal_std
        ),

        "temporal_cv": float(
            temporal_cv
        )

    }


# ==================================================
# 6. MAIN FUNCTION
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

    print("RANGE-TIME ANALYSIS STARTED")

    print(
        "Total decoded files:",
        len(files)
    )

    print("======================================")

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = []

    all_profiles = {

        category: {

            "human": [],

            "without_human": []

        }

        for category in CATEGORIES

    }

    successful = 0

    failed = 0

    # ==================================================
    # PROCESS ALL RECORDINGS
    # ==================================================

    for file in files:

        try:

            parts = [
                part.lower()
                for part in file.relative_to(
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
                    file
                )

                continue

            print("\n--------------------------------------")

            print("Category:", category.upper())

            print("Class:", label)

            # Calculate Range-Time matrix

            range_time, info = calculate_range_time(
                file,
                category
            )

            # Generate individual heatmap

            graph_file = save_range_time_graph(
                range_time,
                category,
                label,
                file
            )

            # Store for class comparison

            all_profiles[category][label].append(
                range_time
            )

            # Calculate temporal information

            metrics = calculate_temporal_metrics(
                range_time
            )

            print("\nTEMPORAL ANALYSIS")

            print(
                "Strongest range bin:",
                metrics["strongest_range_bin"]
            )

            print(
                "Mean magnitude:",
                metrics["mean_magnitude"]
            )

            print(
                "Temporal standard deviation:",
                metrics["temporal_std"]
            )

            print(
                "Temporal coefficient of variation:",
                metrics["temporal_cv"]
            )

            print(
                "Graph saved:",
                graph_file
            )

            row = {

                "file": str(file),

                "category": category,

                "label": label,

                **info,

                **metrics,

                "graph": str(graph_file),

                "status": "SUCCESS"

            }

            summary.append(row)

            successful += 1

        except Exception as error:

            failed += 1

            print(
                "\nERROR:",
                file,
                error
            )

            summary.append({

                "file": str(file),

                "status": "FAILED",

                "error": str(error)

            })

    # ==================================================
    # CLASS COMPARISONS
    # ==================================================

    print("\n======================================")

    print("GENERATING CLASS COMPARISONS")

    print("======================================")

    for category in CATEGORIES:

        compare_human_without_human(
            category,
            all_profiles[category]
        )

    # ==================================================
    # SAVE CSV SUMMARY
    # ==================================================

    summary_file = (
        OUTPUT_ROOT
        / "range_time_summary.csv"
    )

    if summary:

        columns = sorted({

            key

            for row in summary

            for key in row

        })

        with summary_file.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=columns
            )

            writer.writeheader()

            writer.writerows(summary)

    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    print("\n======================================")

    print("RANGE-TIME ANALYSIS COMPLETED")

    print("======================================")

    print("Successful files:", successful)

    print("Failed files:", failed)

    print("Total files found:", len(files))

    print("Graphs folder:", OUTPUT_ROOT)

    print("Summary CSV:", summary_file)

    print("\nNEXT STEP: Validate temporal signal variation.")


# ==================================================
# 7. EXECUTION
# ==================================================

if __name__ == "__main__":

    main()