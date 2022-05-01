import os

# Configuration file for parameters
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
data_df = os.path.join(
    ROOT_DIR, "data", "metadata", "humbugdb_zenodo_0_0_2.csv"
)

data_dir_train = os.path.join(ROOT_DIR, "data", "audio", "train")
data_dir_dev_a = os.path.join(ROOT_DIR, "data", "audio", "dev", "a")
data_dir_dev_b = os.path.join(ROOT_DIR, "data", "audio", "dev", "b")
data_dir_test = os.path.join(ROOT_DIR, "data", "audio", "test")
plot_dir = os.path.join(ROOT_DIR, "data", "plots")
model_dir = os.path.join(
    ROOT_DIR, "models"
)  # Model sub-directory created in config_keras

# Librosa settings
# Feature output directory
# sub-directory for mosquito event_detection
dir_out_MED = os.path.join(ROOT_DIR, "data", "features", "MED")
rate = 8000
win_size = 30
step_size = 10
n_feat = 128
NFFT = 2048
n_hop = NFFT / 4
frame_duration = n_hop / rate  # Frame duration in ms
# Normalisation
norm_per_sample = True

default_model_weights_path = os.path.join(
    ROOT_DIR,
    "models",
    "Win_30_Stride_10_2022_04_27_18_45_11-e01val_accuracy0.9798.hdf5",
)


# Running run_baseline.py script:
# By default, load model object and predict over dev a, dev b.
# Extra_eval and retrain_model require re-creating features.

retrain_model = False  # Train model from scratch (default settings) or load default baseline
predict_dev = True  # To control whether to create new predictions over dev A, dev B
extra_eval = True  # Perform evaluation over addictional feature metrics
predict_test = False  # Perform prediction over data in /data/audio/test/


# Calculating window size based on desired min duration (sample chunks)
# default at 8000Hz: 2048 NFFT -> NFFT/4 for window size = hop length in librosa.
# Recommend lowering NFFT to 1024 so that the default hop length is 256 (or 32 ms).
# Then a win size of 60 produces 60x32 = 1.92 (s) chunks for training

min_duration = win_size * frame_duration  # Change to match 1.92 (later)

# Create directories if they do not exist:
for directory in [plot_dir, dir_out_MED, model_dir]:
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print("Created directory:", directory)
