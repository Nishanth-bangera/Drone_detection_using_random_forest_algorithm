import numpy as np
import matplotlib.pyplot as plt
import os

# ==================================
# 1. LOAD RADAR DATA
# ==================================

npy_path = "results/radar_cube_sample.npy" if os.path.exists("results/radar_cube_sample.npy") else "radar_cube_adc11.npy"
if not os.path.exists(npy_path):
    raise FileNotFoundError("Decoded radar data not found. Please run src/02_adc_decode.py first.")

data = np.load(npy_path)
print(f"Loaded radar data from {npy_path}")
print("Original radar shape:", data.shape)

num_chirps = data.shape[0]
num_samples = data.shape[2]
half_bins = num_samples // 2

# ==================================
# 2. APPLY RANGE FFT
# ==================================

window = np.hanning(num_samples)

range_fft = np.fft.fft(
    data * window[None, None, :],
    axis=2
)

# Keep positive range bins
range_fft = range_fft[:, :, :half_bins]
print("Range FFT shape:", range_fft.shape)

# ==================================
# 3. STATIC CLUTTER REMOVAL
# ==================================

mean_clutter = np.mean(
    range_fft,
    axis=0,
    keepdims=True
)

cleaned_data = range_fft - mean_clutter

# ==================================
# 4. CALCULATE MAGNITUDE
# ==================================

magnitude = np.abs(cleaned_data)

# Average across RX channels
range_time_clean = np.mean(magnitude, axis=1)
print("Cleaned data shape:", cleaned_data.shape)

# ==================================
# 5. DISPLAY CLUTTER-REDUCED MAP
# ==================================

plt.figure(figsize=(13, 6))

plt.imshow(
    20 * np.log10(range_time_clean.T + 1e-6),
    aspect="auto",
    origin="lower",
    extent=[0, num_chirps, 0, half_bins]
)

plt.colorbar(label="Magnitude (dB)")
plt.title("Range-Time Map After Static Clutter Removal")
plt.xlabel("Chirp Number")
plt.ylabel("Range Bin")
plt.tight_layout()

os.makedirs("results/graphs", exist_ok=True)
plt.savefig("results/graphs/range_time_clean.png")
plt.close()

# ==================================
# 6. SAVE CLEANED DATA
# ==================================

np.save("results/cleaned_range_data.npy", cleaned_data)
print("\nStatic clutter removal completed! Saved results/cleaned_range_data.npy")
