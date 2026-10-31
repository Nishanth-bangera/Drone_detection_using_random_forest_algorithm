"""
TI IWR1642 + DCA1000 ADC Decoder

Dataset:
C:\\INTERNS\\data

Supported folders:
srr/human
srr/nohuman
usrr/human
usrr/nohuman
indoor/human
indoor/nohuman

Output:
results/decoded/
"""

from pathlib import Path
import csv
import numpy as np


# ==================================================
# 1. DATASET PATHS
# ==================================================

DATASET_ROOT = Path(r"C:\INTERNS\data")
OUTPUT_ROOT = Path("results") / "decoded"


# ==================================================
# 2. RADAR CONFIGURATION
# ==================================================

RADAR_CONFIGS = {

    "srr": {
        "num_rx": 4,
        "num_adc_samples": 256,
        "sampling_rate_ksps": 5000,
        "chirps_per_frame": 128
    },

    "usrr": {
        "num_rx": 4,
        "num_adc_samples": 512,
        "sampling_rate_ksps": 6250,
        "chirps_per_frame": 64
    },

    "indoor": {
        "num_rx": 4,
        "num_adc_samples": 256,
        "sampling_rate_ksps": 5000,
        "chirps_per_frame": 128
    }
}


# ==================================================
# 3. CLASS FOLDER NAMES
# ==================================================

CLASS_FOLDER_ALIASES = {

    "human": "human",

    "without_human": "without_human",

    "nohuman": "without_human",

    "no_human": "without_human"
}


# ==================================================
# 4. IDENTIFY CATEGORY AND CLASS
# ==================================================

def get_category_and_label(bin_path):

    relative_parts = [
        part.lower()
        for part in bin_path.relative_to(DATASET_ROOT).parts
    ]

    category = next(
        (
            part for part in relative_parts
            if part in RADAR_CONFIGS
        ),
        None
    )

    label = next(
        (
            CLASS_FOLDER_ALIASES[part]
            for part in relative_parts
            if part in CLASS_FOLDER_ALIASES
        ),
        None
    )

    if category is None or label is None:

        raise ValueError(
            "Could not identify category/class. "
            "Expected folders such as srr/human or srr/nohuman."
        )

    return category, label


# ==================================================
# 5. ADC BINARY DECODING
# ==================================================

def decode_iwr1642_dca1000(bin_path, config):

    """
    Output shape:

    (num_chirps, num_rx, num_adc_samples)

    Assumes 4 RX antennas and complex 16-bit ADC data.

    I/Q unpacking follows the xWR16xx DCA1000
    complex-data arrangement.
    """

    num_rx = config["num_rx"]

    num_samples = config["num_adc_samples"]

    chirps_per_frame = config["chirps_per_frame"]

    if num_rx != 4:

        raise ValueError(
            "This decoder currently expects 4 RX antennas."
        )

    if num_samples % 2 != 0:

        raise ValueError(
            "ADC sample count must be even."
        )

    # Read little-endian 16-bit ADC values

    raw = np.fromfile(
        bin_path,
        dtype="<i2"
    )

    if raw.size == 0:

        raise ValueError(
            "BIN file is empty."
        )

    # Calculate values per chirp

    values_per_chirp = (
        num_rx * num_samples * 2
    )

    if raw.size % values_per_chirp != 0:

        raise ValueError(
            f"{raw.size} int16 values are not divisible "
            f"by {values_per_chirp} values/chirp. "
            "Check configuration, file completeness, "
            "and DCA1000 packet reordering."
        )

    num_chirps = (
        raw.size // values_per_chirp
    )

    # Organize raw ADC data

    organized = raw.reshape(
        num_chirps,
        num_rx,
        num_samples // 2,
        4
    )

    # Extract I and Q components

    i_data = organized[:, :, :, 0:2].reshape(
        num_chirps,
        num_rx,
        num_samples
    )

    q_data = organized[:, :, :, 2:4].reshape(
        num_chirps,
        num_rx,
        num_samples
    )

    # Create complex radar signal

    radar_cube = (
        i_data.astype(np.float32)
        + 1j * q_data.astype(np.float32)
    ).astype(np.complex64)

    # Calculate frames

    full_frames, extra_chirps = divmod(
        num_chirps,
        chirps_per_frame
    )

    details = {

        "num_chirps": int(num_chirps),

        "full_frames": int(full_frames),

        "extra_chirps": int(extra_chirps),

        "decoded_shape": str(
            tuple(radar_cube.shape)
        ),

        "i_min": int(i_data.min()),

        "i_max": int(i_data.max()),

        "q_min": int(q_data.min()),

        "q_max": int(q_data.max())
    }

    return radar_cube, details


# ==================================================
# 6. MAIN PROCESSING FUNCTION
# ==================================================

def main():

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_ROOT}"
        )

    # Find all BIN files automatically

    bin_files = sorted(

        path for path in DATASET_ROOT.rglob("*")

        if path.is_file()
        and path.suffix.lower() == ".bin"
    )

    if not bin_files:

        print("No BIN files found.")

        return

    print(
        f"Found {len(bin_files)} BIN file(s)."
    )

    print(
        "Existing decoded files will be skipped."
    )

    rows = []

    newly_decoded = 0

    already_decoded = 0

    failed = 0

    # Process each BIN file

    for source in bin_files:

        try:

            # Identify category and class

            category, label = get_category_and_label(
                source
            )

            config = RADAR_CONFIGS[category]

            # Create output directory

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
                / f"{source.stem}.npy"
            )

            print("\n" + "-" * 70)

            print(f"File: {source}")

            print(f"Category: {category.upper()}")

            print(f"Class: {label}")

            print(
                f"RX: {config['num_rx']}"
            )

            print(
                f"ADC samples: {config['num_adc_samples']}"
            )

            print(
                f"Sampling rate: "
                f"{config['sampling_rate_ksps']} KSPS"
            )

            print(
                f"Chirps/frame: "
                f"{config['chirps_per_frame']}"
            )

            # Skip previously decoded files

            if output_file.exists():

                existing = np.load(
                    output_file,
                    mmap_mode="r"
                )

                if existing.ndim != 3:

                    raise ValueError(
                        f"Invalid existing NPY shape: "
                        f"{existing.shape}"
                    )

                num_chirps = existing.shape[0]

                full_frames, extra_chirps = divmod(
                    num_chirps,
                    config["chirps_per_frame"]
                )

                rows.append({

                    "source_file": str(source),

                    "category": category,

                    "label": label,

                    "decoded_shape": str(
                        tuple(existing.shape)
                    ),

                    "num_chirps": num_chirps,

                    "full_frames": full_frames,

                    "extra_chirps": extra_chirps,

                    "output_file": str(output_file),

                    "status": "ALREADY_DECODED",

                    "error": ""

                })

                already_decoded += 1

                print(
                    f"[SKIP] Already decoded: {output_file}"
                )

                continue

            # Decode ADC data

            cube, details = decode_iwr1642_dca1000(
                source,
                config
            )

            # Save decoded radar cube

            np.save(
                output_file,
                cube
            )

            # Store processing information

            row = {

                "source_file": str(source),

                "category": category,

                "label": label,

                "num_rx": config["num_rx"],

                "num_adc_samples": config["num_adc_samples"],

                "sampling_rate_ksps":
                    config["sampling_rate_ksps"],

                "chirps_per_frame":
                    config["chirps_per_frame"],

                "output_file": str(output_file),

                "status": "SUCCESS",

                "error": ""
            }

            row.update(details)

            rows.append(row)

            newly_decoded += 1

            print(
                f"[SUCCESS] Decoded shape: {cube.shape}"
            )

            print(
                f"Saved to: {output_file}"
            )

            print(
                f"Full frames: {details['full_frames']}"
            )

            print(
                f"Extra chirps: {details['extra_chirps']}"
            )

        except Exception as exc:

            failed += 1

            print(
                f"[FAILED] {source}: {exc}"
            )

            rows.append({

                "source_file": str(source),

                "status": "FAILED",

                "error": str(exc)
            })

    # ==================================================
    # 7. SAVE DECODING MANIFEST
    # ==================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_file = (
        OUTPUT_ROOT
        / "decode_manifest.csv"
    )

    columns = sorted({

        key

        for row in rows

        for key in row.keys()

    })

    with manifest_file.open(
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
    # 8. FINAL SUMMARY
    # ==================================================

    print("\n" + "=" * 70)

    print("ADC DECODING COMPLETED")

    print(
        f"Newly decoded: {newly_decoded}"
    )

    print(
        f"Already decoded: {already_decoded}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Manifest: {manifest_file}"
    )

    print("\nNext step: Validate the decoded radar signal.")


# ==================================================
# 9. EXECUTION
# ==================================================

if __name__ == "__main__":

    main()