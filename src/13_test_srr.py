import numpy as np
import joblib
from pathlib import Path


# ==================================================
# 1. CONFIGURATION
# ==================================================

TEST_BIN = Path("testing/test1.bin")

MODEL_PATH = Path(
    "results/models/srr_random_forest.joblib"
)

NUM_RX = 4
NUM_ADC_SAMPLES = 256
CHIRPS_PER_FRAME = 128

FEATURES = [
    "active_bin_temporal_cv",
    "mean_frame_difference",
    "mean_temporal_std"
]

EPSILON = 1e-10


# ==================================================
# 2. DECODE ADC DATA
# ==================================================

def decode_adc(bin_file):

    print("\nReading ADC data...")

    raw_data = np.fromfile(
        bin_file,
        dtype=np.int16
    )

    if len(raw_data) == 0:
        raise ValueError("BIN file is empty.")

    # Each complex sample contains I and Q values.
    # ADC data is stored as I1, Q1, I2, Q2...

    if len(raw_data) % 2 != 0:
        raise ValueError(
            "Invalid ADC data: odd number of int16 values."
        )

    complex_data = (
        raw_data[0::2].astype(np.float32)
        + 1j * raw_data[1::2].astype(np.float32)
    )

    samples_per_chirp = (
        NUM_RX * NUM_ADC_SAMPLES
    )

    if len(complex_data) % samples_per_chirp != 0:
        raise ValueError(
            "ADC data size is not divisible by "
            "RX antennas × ADC samples."
        )

    num_chirps = (
        len(complex_data) // samples_per_chirp
    )

    radar_cube = complex_data.reshape(
        num_chirps,
        NUM_RX,
        NUM_ADC_SAMPLES
    )

    radar_cube = radar_cube.astype(
        np.complex64
    )

    print("Radar cube shape:", radar_cube.shape)
    print("Total chirps:", num_chirps)

    return radar_cube


# ==================================================
# 3. CALCULATE RANGE-TIME MATRIX
# ==================================================

def calculate_range_time(radar_cube):

    print("\nCalculating Range-Time matrix...")

    num_chirps, num_rx, num_samples = (
        radar_cube.shape
    )

    num_frames = (
        num_chirps + CHIRPS_PER_FRAME - 1
    ) // CHIRPS_PER_FRAME

    half_bins = num_samples // 2

    window = np.hanning(
        num_samples
    ).astype(np.float32)

    range_time = np.zeros(
        (num_frames, half_bins),
        dtype=np.float32
    )

    for frame in range(num_frames):

        start = (
            frame * CHIRPS_PER_FRAME
        )

        end = min(
            start + CHIRPS_PER_FRAME,
            num_chirps
        )

        frame_data = radar_cube[
            start:end
        ]

        windowed = (
            frame_data
            * window[None, None, :]
        )

        fft_result = np.fft.fft(
            windowed,
            axis=2
        )

        magnitude = np.abs(
            fft_result[:, :, :half_bins]
        )

        range_time[frame] = np.mean(
            magnitude,
            axis=(0, 1)
        )

    print(
        "Range-Time matrix shape:",
        range_time.shape
    )

    return range_time


# ==================================================
# 4. EXTRACT FEATURES
# ==================================================

def extract_features(range_time):

    print("\nExtracting features...")

    # Average reflection strength
    bin_mean = np.mean(
        range_time,
        axis=0
    )

    # Temporal standard deviation
    bin_std = np.std(
        range_time,
        axis=0
    )

    # Coefficient of variation
    bin_cv = (
        bin_std
        / np.maximum(
            bin_mean,
            EPSILON
        )
    )

    # Frame-to-frame absolute difference
    if range_time.shape[0] > 1:

        frame_difference = np.abs(
            np.diff(
                range_time,
                axis=0
            )
        )

        mean_frame_difference = float(
            np.mean(frame_difference)
        )

    else:

        mean_frame_difference = 0.0

    # Ignore DC bin
    analysis_mean = bin_mean[1:]

    analysis_std = bin_std[1:]

    analysis_cv = bin_cv[1:]

    # Active range bins
    maximum_mean = np.max(
        analysis_mean
    )

    active_threshold = (
        0.10 * maximum_mean
    )

    active_mask = (
        analysis_mean >= active_threshold
    )

    if np.any(active_mask):

        active_cv = analysis_cv[
            active_mask
        ]

        active_bin_temporal_cv = float(
            np.mean(active_cv)
        )

    else:

        active_bin_temporal_cv = 0.0

    # Required Random Forest features
    features = {

        "active_bin_temporal_cv":
            active_bin_temporal_cv,

        "mean_frame_difference":
            mean_frame_difference,

        "mean_temporal_std":
            float(
                np.mean(analysis_std)
            )
    }

    print("\nFEATURE VALUES")

    for name, value in features.items():

        print(
            f"{name}: {value:.6f}"
        )

    return features


# ==================================================
# 5. LOAD MODEL AND PREDICT
# ==================================================

def predict_human(features):

    print("\nLoading SRR Random Forest model...")

    model_bundle = joblib.load(
        MODEL_PATH
    )

    model = model_bundle["model"]

    trained_features = (
        model_bundle["features"]
    )

    trained_category = (
        model_bundle["category"]
    )

    if trained_category != "srr":

        raise ValueError(
            "Loaded model is not an SRR model."
        )

    if trained_features != FEATURES:

        raise ValueError(
            "Feature order does not match "
            "the trained model."
        )

    feature_vector = np.array(
        [
            features[name]
            for name in FEATURES
        ],
        dtype=np.float64
    ).reshape(1, -1)

    prediction = model.predict(
        feature_vector
    )[0]

    probabilities = (
        model.predict_proba(
            feature_vector
        )[0]
    )

    classes = model.classes_

    confidence = float(
        probabilities[
            list(classes).index(prediction)
        ]
    )

    print("\n======================================")

    print("SRR HUMAN DETECTION RESULT")

    print("======================================")

    if prediction == "human":

        print("HUMAN DETECTED")

    elif prediction == "without_human":

        print("NO HUMAN DETECTED")

    else:

        print("UNKNOWN CLASS:", prediction)

    print(
        f"Prediction probability: "
        f"{confidence * 100:.2f}%"
    )

    print("======================================")

    return prediction


# ==================================================
# 6. MAIN FUNCTION
# ==================================================

def main():

    if not TEST_BIN.exists():

        raise FileNotFoundError(
            f"Testing BIN file not found: {TEST_BIN}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"SRR model not found: {MODEL_PATH}"
        )

    print("\n======================================")

    print("SRR INDEPENDENT TESTING")

    print("======================================")

    # Decode ADC data
    radar_cube = decode_adc(
        TEST_BIN
    )

    # Calculate Range-Time matrix
    range_time = calculate_range_time(
        radar_cube
    )

    # Extract features
    features = extract_features(
        range_time
    )

    # Predict
    predict_human(
        features
    )


# ==================================================
# 7. EXECUTION
# ==================================================

if __name__ == "__main__":

    main()