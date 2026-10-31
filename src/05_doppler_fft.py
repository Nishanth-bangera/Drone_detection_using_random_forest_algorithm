import os

DATASET_PATH = r"C:\INTERNS\data"

print("RADAR DATASET INVENTORY")
print("=" * 60)

for root, dirs, files in os.walk(DATASET_PATH):

    for file in sorted(files):

        if file.endswith(".bin"):
            path = os.path.join(root, file)

            size = os.path.getsize(path)

            print("\nFile:", file)
            print("Size:", size, "bytes")
            print("Size:", round(size / (1024 * 1024), 2), "MB")

print("\nInventory completed!")