
import os
import re

DATASET_PATH = r"C:\INTERNS\data"

print("=" * 75)
print("RADAR EXPERIMENTAL PARAMETER EXTRACTION")
print("=" * 75)

for filename in sorted(os.listdir(DATASET_PATH)):

    if filename.endswith("_LogFile.txt"):

        path = os.path.join(DATASET_PATH, filename)

        print(f"\n\nFILE: {filename}")
        print("-" * 60)

        with open(path, "r", errors="ignore") as f:
            content = f.read()

        parameters = [
            "ChannelConfig",
            "AdcOutConfig",
            "ProfileConfig",
            "ChirpConfig",
            "FrameConfig",
            "AdvancedFrameConfig",
            "EnableTestSource",
            "SensorStart"
        ]

        for parameter in parameters:

            matches = re.findall(
                rf"API:{parameter}[^\r\n]*",
                content
            )

            for match in matches:
                print(match)

print("\n\nEXTRACTION COMPLETED")