import os
from pathlib import Path

DATA_DIR = r"C:\INTERNS\data"

print("=" * 80)
print(f"RADAR DATASET INSPECTION: {DATA_DIR}")
print("=" * 80)

if not os.path.exists(DATA_DIR):
    print(f"DATA DIRECTORY NOT FOUND: {DATA_DIR}")
else:
    print(f"Dataset root found: {DATA_DIR}\n")
    
    # Categories: usrr (6420 ksps), srr (5000 ksps), indoor (5000 ksps)
    for category in ["usrr", "srr", "indoor"]:
        cat_path = os.path.join(DATA_DIR, category)
        print(f"\n--- Category: {category.upper()} ---")
        if not os.path.exists(cat_path):
            print("  Category directory not found.")
            continue
        
        for root, dirs, files in os.walk(cat_path):
            for file in sorted(files):
                if file.endswith(".bin"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, DATA_DIR)
                    file_size = os.path.getsize(full_path)
                    div2 = file_size % 2 == 0
                    div4 = file_size % 4 == 0
                    print(f"File: {rel_path:<35} | Size: {file_size:>9} B ({file_size/(1024*1024):.1f} MB) | Div2: {str(div2):<5} | Div4: {str(div4):<5}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETED")
print("=" * 80)
