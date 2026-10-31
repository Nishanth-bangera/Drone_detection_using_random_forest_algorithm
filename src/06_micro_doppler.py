
import os

DATASET_PATH = r"C:\INTERNS\data"

print("=" * 65)
print("RADAR RECORDING METADATA INVENTORY")
print("=" * 65)

for root, dirs, files in os.walk(DATASET_PATH):

    for file in sorted(files):

        extension = os.path.splitext(file)[1].lower()

        if extension in [".bin", ".txt", ".csv"]:

            path = os.path.join(root, file)
            size = os.path.getsize(path)

            print(f"{file:<40} {size:>12} bytes")

print("\nMetadata inventory completed!")