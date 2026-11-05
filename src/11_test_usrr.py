import numpy as np
import joblib

from pathlib import Path


# ==========================================
# 1. CONFIGURATION
# ==========================================

TEST_BIN = Path("testing/test1.bin")

MODEL_PATH = Path(
    "results/models/usrr_random_forest.joblib"
)

OUTPUT_DIR = Path("results/testing")

NUM_RX = 4
NUM_ADC_SAMPLES = 512
CHIRPS_PER_FRAME = 64

FEATURES = [
    "active_bin_temporal_cv",
    "mean_frame_difference",
    "mean_temporal_std"
]

EPSILON = 1e-10


# ==========================================
# 2. ADC DECODING
# ==========================================

def decode_adc(bin_path):

    print("\nSTEP 1: ADC DECODING")

    raw = np.fromfile(
        bin_path,
        dtype="<i2"
    )

    if raw.size == 0:
        raise ValueError("Empty BIN file.")

    values_per_chirp = (
        NUM_RX * NUM_ADC_SAMPLES * 2
    )

    if raw.size % values_per_chirp != 0:
        raise ValueError(
            "Invalid BIN size. Check ADC configuration."
        )

    num_chirps = (
        raw.size // values_per_chirp
    )

    organized = raw.reshape(
        num_chirps,
        NUM_RX,
        NUM_ADC_SAMPLES // 2,
        4
    )

    i_data = organized[:, :, :, 0:2].reshape(
        num_chirps,
        NUM_RX,
        NUM_ADC_SAMPLES
    )

    q_data = organized[:, :, :, 2:4].reshape(
        num_chirps,
        NUM_RX,
        NUM_ADC_SAMPLES
    )

    radar_cube = (
        i_data.astype(np.float32)
        + 1j * q_data.astype(np.float32)
    ).astype(np.complex64)

    full_frames, extra_chirps = divmod(
        num_chirps,
        CHIRPS_PER_FRAME
    )

    print("ADC decoding successful")

    print("Total chirps:", num_chirps)

    print("Full frames:", full_frames)

    print("Extra chirps:", extra_chirps)

    print("IQ shape:", radar_cube.shape)

    return radar_cube


# ==========================================
# 3. RANGE-TIME ANALYSIS
# ==========================================

def calculate_range_time(radar_cube):

    print("\nSTEP 2: RANGE-TIME ANALYSIS")

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

        start = frame * CHIRPS_PER_FRAME

        end = min(
            start + CHIRPS_PER_FRAME,
            num_chirps
        )

        frame_data = np.asarray(
            radar_cube[start:end],
            dtype=np.complex64
        )

        windowed = (
            frame_data *
            window[None, None, :]
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

    print("Range-Time shape:", range_time.shape)

    return range_time


# ==========================================
# 4. FEATURE EXTRACTION
# ==========================================

def extract_features(range_time):

    print("\nSTEP 3: FEATURE EXTRACTION")

    bin_mean = np.mean(
        range_time,
        axis=0
    )

    bin_std = np.std(
        range_time,
        axis=0
    )

    bin_cv = (
        bin_std /
        np.maximum(bin_mean, EPSILON)
    )

    if range_time.shape[0] > 1:

        frame_difference = np.abs(
            np.diff(
                range_time,
                axis=0
            )
        )

        bin_frame_difference = np.mean(
            frame_difference,
            axis=0
        )

        mean_frame_difference = float(
            np.mean(frame_difference)
        )

    else:

        bin_frame_difference = np.zeros(
            range_time.shape[1]
        )

        mean_frame_difference = 0.0

    analysis_std = bin_std[1:]

    analysis_cv = bin_cv[1:]

    analysis_mean = bin_mean[1:]

    maximum_mean = np.max(analysis_mean)

    active_threshold = 0.10 * maximum_mean

    active_mask = (
        analysis_mean >= active_threshold
    )

    if np.any(active_mask):

        active_cv = analysis_cv[active_mask]

        active_bin_temporal_cv = float(
            np.mean(active_cv)
        )

    else:

        active_bin_temporal_cv = 0.0

    features = {

        "active_bin_temporal_cv":
            active_bin_temporal_cv,

        "mean_frame_difference":
            mean_frame_difference,

        "mean_temporal_std":
            float(np.mean(analysis_std))
    }

    print("\nEXTRACTED FEATURES")

    for name, value in features.items():

        print(f"{name}: {value}")

    return features


# ==========================================
# 5. RANDOM FOREST PREDICTION
# ==========================================

def predict_human(features):

    print("\nSTEP 4: RANDOM FOREST PREDICTION")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model_bundle = joblib.load(
        MODEL_PATH
    )

    model = model_bundle["model"]

    model_features = model_bundle["features"]

    if model_bundle["category"] != "usrr":
        raise ValueError("Loaded model is not a USRR model.")

    if model_features != FEATURES:
        raise ValueError(
            "Feature order does not match trained model."
        )

    X = np.array(
        [[features[name] for name in model_features]],
        dtype=np.float64
    )

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    print("\nPREDICTION PROBABILITIES")

    for label, probability in zip(
        model.classes_,
        probabilities
    ):

        print(
            f"{label}: {probability * 100:.2f}%"
        )

    print("\n" + "=" * 55)

    print("FINAL RADAR HUMAN DETECTION RESULT")

    print("=" * 55)

    if prediction == "human":

        print("PREDICTION: HUMAN DETECTED")

    else:

        print("PREDICTION: NO HUMAN DETECTED")

    print("=" * 55)

    return prediction, probabilities


# ==========================================
# 6. MAIN FUNCTION
# ==========================================

def main():

    print("\n==========================================")
    print("RADAR BASED HUMAN DETECTION")
    print("INDEPENDENT TESTING")
    print("==========================================")

    if not TEST_BIN.exists():

        raise FileNotFoundError(
            f"Testing file not found: {TEST_BIN}"
        )

    print("\nTesting file:", TEST_BIN)

    print("File size:", TEST_BIN.stat().st_size)

    # ADC decoding

    radar_cube = decode_adc(TEST_BIN)

    # Range-Time analysis

    range_time = calculate_range_time(
        radar_cube
    )

    # Feature extraction

    features = extract_features(
        range_time
    )

    # Random Forest prediction

    prediction, probabilities = predict_human(
        features
    )

    print("\nTESTING COMPLETED")

    print("Actual class: UNKNOWN")

    print("Predicted class:", prediction)


# ==========================================
# 7. EXECUTION
# ==========================================

if __name__ == "__main__":
    main()