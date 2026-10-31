import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from pathlib import Path

# ==================================================
# 1. CONFIGURATION
# ==================================================

DECODED_ROOT = Path("results/decoded")

GRAPH_ROOT = Path("results/graphs/range_fft")

CHIRPS_PER_BATCH = 128

CATEGORIES = ["srr", "usrr", "indoor"]

CLASSES = ["human", "without_human"]


# ==================================================
# 2. CALCULATE AVERAGE RANGE PROFILE
# ==================================================

def calculate_range_fft(npy_file):

    data = np.load(npy_file, mmap_mode="r")

    if data.ndim != 3:
        raise ValueError(f"Invalid radar data shape: {data.shape}")

    num_chirps, num_rx, num_samples = data.shape

    if not np.iscomplexobj(data):
        raise ValueError("Expected complex I/Q radar data.")

    print("\nLoaded:", npy_file)
    print("Radar shape:", data.shape)

    # Apply Hann window
    window = np.hanning(num_samples).astype(np.float32)

    half = num_samples // 2

    magnitude_sum = np.zeros(half, dtype=np.float64)

    total_samples = 0

    # Process chirps in batches to reduce memory usage
    for start in range(0, num_chirps, CHIRPS_PER_BATCH):

        end = min(
            start + CHIRPS_PER_BATCH,
            num_chirps
        )

        batch = np.asarray(
            data[start:end],
            dtype=np.complex64
        )

        # Apply window
        windowed = batch * window[None, None, :]

        # Perform Range FFT along ADC sample axis
        range_fft = np.fft.fft(
            windowed,
            axis=2
        )

        magnitude = np.abs(
            range_fft[:, :, :half]
        )

        magnitude_sum += magnitude.sum(
            axis=(0, 1),
            dtype=np.float64
        )

        total_samples += (
            magnitude.shape[0] * magnitude.shape[1]
        )

    # Average across all chirps and RX antennas
    average_profile = magnitude_sum / total_samples

    # Ignore DC bin when finding strongest non-zero range bin
    peak_bin = int(
        np.argmax(average_profile[1:]) + 1
    )

    print("\nRANGE FFT RESULTS")

    print("FFT size:", num_samples)

    print("Peak range bin:", peak_bin)

    print(
        "Peak magnitude:",
        average_profile[peak_bin]
    )

    return average_profile, {
        "num_chirps": num_chirps,
        "num_rx": num_rx,
        "num_adc_samples": num_samples,
        "fft_size": num_samples,
        "peak_range_bin": peak_bin,
        "peak_magnitude": float(average_profile[peak_bin]),
        "mean_magnitude": float(np.mean(average_profile))
    }


# ==================================================
# 3. SAVE INDIVIDUAL RANGE PROFILE
# ==================================================

def save_range_plot(profile, category, label, filename):

    output_dir = (
        GRAPH_ROOT / category / label
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir /
        f"{filename.stem}_range_fft.png"
    )

    plt.figure(figsize=(12, 5))

    plt.plot(profile)

    plt.title(
        f"{category.upper()} - {label} - {filename.stem}"
    )

    plt.xlabel("Range Bin")

    plt.ylabel("Average FFT Magnitude")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    return output_file


# ==================================================
# 4. COMPARE HUMAN VS WITHOUT HUMAN
# ==================================================

def compare_classes(category, profiles):

    human_profiles = profiles["human"]

    nohuman_profiles = profiles["without_human"]

    if not human_profiles or not nohuman_profiles:

        print(
            f"Not enough data for {category.upper()} comparison."
        )

        return

    # Average profiles across recordings
    human_average = np.mean(
        np.stack(human_profiles),
        axis=0
    )

    nohuman_average = np.mean(
        np.stack(nohuman_profiles),
        axis=0
    )

    output_dir = GRAPH_ROOT / category

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir /
        f"{category}_human_vs_without_human.png"
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        human_average,
        label="Human"
    )

    plt.plot(
        nohuman_average,
        label="Without Human"
    )

    plt.title(
        f"{category.upper()} - Human vs Without Human"
    )

    plt.xlabel("Range Bin")

    plt.ylabel("Average FFT Magnitude")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    print(
        "\nComparison graph saved:",
        output_file
    )


# ==================================================
# 5. MAIN FUNCTION
# ==================================================

def main():

    if not DECODED_ROOT.exists():

        raise FileNotFoundError(
            "Decoded radar data not found. "
            "Run 02_adc_decode.py first."
        )

    files = sorted(
        DECODED_ROOT.rglob("*.npy")
    )

    print("\n======================================")

    print("RANGE FFT ANALYSIS STARTED")

    print("Total decoded files:", len(files))

    print("======================================")

    GRAPH_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = []

    # Store profiles for class comparison
    all_profiles = {

        category: {
            "human": [],
            "without_human": []
        }

        for category in CATEGORIES
    }

    success = 0

    failed = 0

    # ==================================================
    # 6. PROCESS EVERY NPY FILE
    # ==================================================

    for file in files:

        try:

            parts = [
                p.lower()
                for p in file.relative_to(
                    DECODED_ROOT
                ).parts
            ]

            category = next(
                (p for p in parts if p in CATEGORIES),
                None
            )

            label = next(
                (p for p in parts if p in CLASSES),
                None
            )

            if category is None or label is None:

                print("Skipping unrecognized file:", file)

                continue

            print("\n--------------------------------------")

            print("Category:", category.upper())

            print("Class:", label)

            # Calculate FFT
            profile, metrics = calculate_range_fft(file)

            # Save graph
            graph_file = save_range_plot(
                profile,
                category,
                label,
                file
            )

            # Save profile for comparison
            all_profiles[category][label].append(
                profile
            )

            row = {

                "file": str(file),

                "category": category,

                "label": label,

                **metrics,

                "graph": str(graph_file),

                "status": "SUCCESS"
            }

            summary.append(row)

            success += 1

            print("Graph saved:", graph_file)

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
    # 7. GENERATE CATEGORY COMPARISONS
    # ==================================================

    print("\n======================================")

    print("GENERATING CLASS COMPARISONS")

    print("======================================")

    for category in CATEGORIES:

        compare_classes(
            category,
            all_profiles[category]
        )

    # ==================================================
    # 8. SAVE CSV SUMMARY
    # ==================================================

    summary_file = (
        GRAPH_ROOT / "range_fft_summary.csv"
    )

    columns = sorted({
        key
        for row in summary
        for key in row
    })

    with open(
        summary_file,
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
    # 9. FINAL RESULTS
    # ==================================================

    print("\n======================================")

    print("RANGE FFT ANALYSIS COMPLETED")

    print("======================================")

    print("Successful files:", success)

    print("Failed files:", failed)

    print("Total files found:", len(files))

    print("Graphs folder:", GRAPH_ROOT)

    print("Summary CSV:", summary_file)

    print("\nNEXT STEP: Range-Time Analysis")


# ==================================================
# 10. EXECUTION
# ==================================================

if __name__ == "__main__":

    main()