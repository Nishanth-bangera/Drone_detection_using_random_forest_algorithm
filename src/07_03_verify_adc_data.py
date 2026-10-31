import numpy as np
import os
import matplotlib.pyplot as plt

# ==================================
# 1. FILE CONFIGURATION
# ==================================

FILE_PATH = r"C:\INTERNS\data\usrr\human\test_data_1.bin"

NUM_RX = 4
# USRR uses 512 samples (6420/6250 ksps), SRR and Indoor use 256 samples (5000 ksps)
if "usrr" in FILE_PATH.lower():
    NUM_ADC_SAMPLES = 512
elif "srr" in FILE_PATH.lower() or "indoor" in FILE_PATH.lower():
    NUM_ADC_SAMPLES = 256
else:
    NUM_ADC_SAMPLES = 512

# ==================================
# 2. READ RAW ADC DATA
# ==================================

print("\nRADAR ADC DECODING VERIFICATION")
print("Target file:", FILE_PATH)

raw = np.fromfile(FILE_PATH, dtype=np.int16)

print("\nFile information")
print("Total int16 values:", len(raw))
print("Minimum:", raw.min())
print("Maximum:", raw.max())

# ==================================
# 3. CALCULATE CHIRPS
# ==================================

values_per_chirp = (
    NUM_RX * NUM_ADC_SAMPLES * 2
)

if len(raw) % values_per_chirp != 0:
    raise ValueError(
        f"File size does not match the assumed {NUM_RX}-RX, {NUM_ADC_SAMPLES}-sample complex format."
    )

NUM_CHIRPS = len(raw) // values_per_chirp

print("Number of RX:", NUM_RX)
print("ADC samples:", NUM_ADC_SAMPLES)
print("Calculated chirps:", NUM_CHIRPS)

# ==================================
# 4. RECONSTRUCT COMPLEX SAMPLES
# ==================================

# TI IWR16xx DCA1000 complex format:
# I0, I1, Q0, Q1, I2, I3, Q2, Q3 ...

data = raw.reshape(
    NUM_CHIRPS,
    NUM_RX,
    NUM_ADC_SAMPLES // 2,
    4
)

I = data[..., 0:2].reshape(
    NUM_CHIRPS, NUM_RX, NUM_ADC_SAMPLES
)

Q = data[..., 2:4].reshape(
    NUM_CHIRPS, NUM_RX, NUM_ADC_SAMPLES
)

radar_cube = I.astype(np.float32) + 1j * Q.astype(np.float32)

print("\nDecoded radar cube:", radar_cube.shape)
print("I range:", I.min(), I.max())
print("Q range:", Q.min(), Q.max())

# ==================================
# 5. SAVE DECODED DATA
# ==================================

os.makedirs("results", exist_ok=True)

np.save(
    "results/verified_candidate.npy",
    radar_cube
)

print("\nDecoded candidate saved.")

# ==================================
# 6. TIME DOMAIN SIGNAL
# ==================================

plt.figure(figsize=(12, 4))

plt.plot(
    radar_cube[0, 0, :].real,
    label="I component"
)

plt.plot(
    radar_cube[0, 0, :].imag,
    label="Q component"
)

plt.title("Raw ADC Signal - Chirp 1, RX0")
plt.xlabel("ADC Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("results/09_time_signal.png")
plt.close()

# ==================================
# 7. RANGE FFT
# ==================================

window = np.hanning(NUM_ADC_SAMPLES)

range_fft = np.fft.fft(
    radar_cube[0, 0, :] * window
)

magnitude = np.abs(range_fft)

plt.figure(figsize=(12, 4))

plt.plot(magnitude[:NUM_ADC_SAMPLES // 2])

plt.title("Range FFT - Chirp 1, RX0")
plt.xlabel("Range Bin")
plt.ylabel("Magnitude")
plt.grid()

plt.tight_layout()
plt.savefig("results/09_range_fft.png")
plt.close()

# ==================================
# 8. AVERAGE RANGE PROFILE
# ==================================

range_fft_all = np.fft.fft(
    radar_cube * window,
    axis=2
)

average_profile = np.mean(
    np.abs(range_fft_all[:, :, :NUM_ADC_SAMPLES // 2]),
    axis=(0, 1)
)

plt.figure(figsize=(12, 4))

plt.plot(average_profile)

plt.title("Average Range Profile")
plt.xlabel("Range Bin")
plt.ylabel("Average Magnitude")
plt.grid()

plt.tight_layout()
plt.savefig("results/09_average_range_profile.png")
plt.close()

print("\nVERIFICATION SCRIPT COMPLETED")
