import os
import re
from collections import Counter

DATASET_PATH = r"C:\INTERNS\data"

PARAMETERS = [
    "ChannelConfig",
    "AdcOutConfig",
    "ProfileConfig",
    "ChirpConfig",
    "FrameConfig",
    "AdvancedFrameConfig",
    "EnableTestSource",
]

print("=" * 85)
print("RADAR RECORDING METADATA COMPARISON")
print("=" * 85)

binary_files = []
for root, dirs, files in os.walk(DATASET_PATH):
    for f in sorted(files):
        if f.lower().endswith(".bin"):
            binary_files.append((os.path.join(root, f), f))

for bin_path, binary in sorted(binary_files, key=lambda x: x[1]):

    # Special filename mapping
    if binary == "test_data_ten_Raw_0.bin":
        base = "test_data_ten"
    else:
        base = os.path.splitext(binary)[0]

    log_file = os.path.join(
        DATASET_PATH, base + "_LogFile.txt"
    )

    rel_folder = os.path.relpath(bin_path, DATASET_PATH)
    print(f"\nRECORDING: {binary} ({rel_folder})")
    print("-" * 70)

    print("Binary size:",
          os.path.getsize(bin_path),
          "bytes")

    if not os.path.exists(log_file):
        print("WARNING: Log file not found at", log_file)
        continue

    with open(log_file, "r", errors="ignore") as f:
        content = f.read()

    for parameter in PARAMETERS:

        matches = re.findall(
            rf"API:{parameter}[^\r\n]*",
            content
        )

        if not matches:
            print(f"{parameter}: NOT FOUND")
            continue

        unique = list(dict.fromkeys(matches))

        print(f"{parameter}: {len(matches)} occurrence(s), "
              f"{len(unique)} unique value(s)")

        for value in unique:
            print("  ", value)

        if len(unique) > 1:
            print("  WARNING: Multiple configurations detected")

print("\n" + "=" * 85)
print("COMPARISON COMPLETED")
print("=" * 85)
