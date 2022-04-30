import collections
import math
import os
import pickle

import librosa
import numpy as np
import skimage.util

import config

# Sound type unique values: {'background', 'mosquito', 'audio'}; class labels: {1, 0, 0}

# Extract features from wave files with id corresponding to dataframe data_df.


def get_feat(data_df, data_dir, rate, min_duration, n_feat):
    """Returns features extracted with Librosa. A list of features, with the number of items equal to the number of input recordings"""
    X = []
    y = []
    bugs = []
    idx = 0
    skipped_files = []
    for row_idx_series in data_df.iterrows():
        idx += 1
        if idx % 100 == 0:
            print("Completed", idx, "of", len(data_df))
        row = row_idx_series[1]
        label_duration = row["length"]
        if label_duration > min_duration:
            _, file_format = os.path.splitext(row["name"])
            filename = os.path.join(data_dir, str(row["id"]) + file_format)
            length = librosa.get_duration(filename=filename)
            #             assert math.isclose(length,label_duration, rel_tol=0.01), "File: %s label duration (%.4f) does not match audio length (%.4f)" % (row['path'], label_duration, length)

            if math.isclose(length, label_duration, rel_tol=0.01):
                signal, rate = librosa.load(filename, sr=rate)
                feat = librosa.feature.melspectrogram(y=signal, sr=rate, n_mels=n_feat)
                feat = librosa.power_to_db(feat, ref=np.max)
                if config.norm_per_sample:
                    feat = (feat - np.mean(feat)) / np.std(feat)
                X.append(feat)
                if row["sound_type"] == "mosquito":
                    y.append(1)
                elif row[
                    "sound_type"
                ]:  # Condition to check we are not adding empty (or unexpected) labels as 0
                    y.append(0)
            else:
                print(
                    "File: %s label duration (%.4f) does not match audio length (%.4f)"
                    % (row["name"], label_duration, length)
                )
                bugs.append([row["name"], label_duration, length])

        else:
            skipped_files.append([row["id"], row["name"], label_duration])
    return X, y, skipped_files, bugs


def get_signal(data_df, data_dir, rate, min_duration):
    """Returns raw audio with Librosa, and corresponding label longer than min_duration"""
    X = []
    y = []
    idx = 0
    bugs = []
    skipped_files = []
    label_dict = {}
    for row_idx_series in data_df.iterrows():
        row = row_idx_series[1]
        label_duration = row["length"]
        if label_duration > min_duration:
            _, file_format = os.path.splitext(row["name"])
            filename = os.path.join(data_dir, str(row["id"]) + file_format)

            length = librosa.get_duration(filename=filename)
            #             assert math.isclose(length,label_duration, rel_tol=0.01), "File: %s label duration (%.4f) does not match audio length (%.4f)" % (row['path'], label_duration, length)

            if math.isclose(length, label_duration, rel_tol=0.01):
                signal, rate = librosa.load(filename, sr=rate)
                label_dict[idx] = [row["id"], row["name"], row["length"]]
                idx += 1
                X.append(signal)
                if row["sound_type"] == "mosquito":
                    y.append(1)
                elif row[
                    "sound_type"
                ]:  # Condition to check we are not adding empty (or unexpected) labels as 0
                    y.append(0)
            else:
                print(
                    "File: %s label duration (%.4f) does not match audio length (%.4f)"
                    % (row["name"], label_duration, length)
                )
                bugs.append([row["name"], label_duration, length])

        else:
            skipped_files.append([row["id"], row["name"], label_duration])
    return X, y, label_dict, skipped_files, bugs


def reshape_feat(feats, labels, win_size, step_size):
    """Reshaping features from get_feat to be compatible for classifiers expecting a 2D slice as input. Parameter `win_size` is
    given in number of feature windows (in librosa this is the hop length divided by the sample rate.)
    Can code to be a function of time and hop length instead in future."""

    feats_windowed_array = []
    labels_windowed_array = []
    for idx, feat in enumerate(feats):
        if np.shape(feat)[1] < win_size:
            print("Length of recording shorter than supplied window size.")
            pass
        else:
            feats_windowed = skimage.util.view_as_windows(
                feat.T, (win_size, np.shape(feat)[0]), step=step_size
            )
            labels_windowed = np.full(len(feats_windowed), labels[idx])
            feats_windowed_array.append(feats_windowed)
            labels_windowed_array.append(labels_windowed)
    return np.vstack(feats_windowed_array), np.hstack(labels_windowed_array)


def get_train_dev_from_df(df_train, df_dev_a, df_dev_b, debug=False):

    pickle_name_train = (
        "log_mel_feat_train_"
        + str(config.n_feat)
        + "_win_"
        + str(config.win_size)
        + "_step_"
        + str(config.step_size)
        + "_norm_"
        + str(config.norm_per_sample)
        + ".pickle"
    )
    # step = window for dev (no augmentation of dev):
    pickle_name_dev = (
        "log_mel_feat_dev_"
        + str(config.n_feat)
        + "_win_"
        + str(config.win_size)
        + "_step_"
        + str(config.win_size)
        + "_norm_"
        + str(config.norm_per_sample)
        + ".pickle"
    )

    if not os.path.isfile(os.path.join(config.dir_out_MED, pickle_name_train)):
        print("Extracting training features...")
        X_train, y_train, skipped_files_train, bugs_train = get_feat(
            data_df=df_train,
            data_dir=config.data_dir_train,
            rate=config.rate,
            min_duration=config.min_duration,
            n_feat=config.n_feat,
        )
        X_train, y_train = reshape_feat(
            X_train, y_train, config.win_size, config.step_size
        )

        log_mel_feat_train = {
            "X_train": X_train,
            "y_train": y_train,
            "bugs_train": bugs_train,
        }

        if debug:
            print("Bugs train", bugs_train)

        with open(os.path.join(config.dir_out_MED, pickle_name_train), "wb") as f:
            pickle.dump(log_mel_feat_train, f, protocol=4)
            print(
                "Saved features to:",
                os.path.join(config.dir_out_MED, pickle_name_train),
            )

    else:
        print(
            "Loading training features found at:",
            os.path.join(config.dir_out_MED, pickle_name_train),
        )
        with open(
            os.path.join(config.dir_out_MED, pickle_name_train), "rb"
        ) as input_file:
            log_mel_feat = pickle.load(input_file)
            X_train = log_mel_feat["X_train"]
            y_train = log_mel_feat["y_train"]

    if not os.path.isfile(os.path.join(config.dir_out_MED, pickle_name_dev)):
        print("Extracting dev features...")

        X_dev_a, y_dev_a, skipped_files_dev_a, bugs_dev_a = get_feat(
            data_df=df_dev_a,
            data_dir=config.data_dir_dev_a,
            rate=config.rate,
            min_duration=config.min_duration,
            n_feat=config.n_feat,
        )
        X_dev_b, y_dev_b, skipped_files_dev_b, bugs_dev_b = get_feat(
            data_df=df_dev_b,
            data_dir=config.data_dir_dev_b,
            rate=config.rate,
            min_duration=config.min_duration,
            n_feat=config.n_feat,
        )
        X_dev_a, y_dev_a = reshape_feat(
            X_dev_a, y_dev_a, config.win_size, config.win_size
        )  # dev should be strided with step = window.
        X_dev_b, y_dev_b = reshape_feat(
            X_dev_b, y_dev_b, config.win_size, config.win_size
        )

        log_mel_feat_dev = {
            "X_dev_a": X_dev_a,
            "X_dev_b": X_dev_b,
            "y_dev_a": y_dev_a,
            "y_dev_b": y_dev_b,
        }

        if debug:
            print("Bugs dev A", bugs_dev_a)
            print("Bugs dev B", bugs_dev_b)

        with open(os.path.join(config.dir_out_MED, pickle_name_dev), "wb") as f:
            pickle.dump(log_mel_feat_dev, f, protocol=4)
            print(
                "Saved features to:", os.path.join(config.dir_out_MED, pickle_name_dev)
            )
    else:
        print(
            "Loading dev features found at:",
            os.path.join(config.dir_out_MED, pickle_name_dev),
        )
        with open(
            os.path.join(config.dir_out_MED, pickle_name_dev), "rb"
        ) as input_file:
            log_mel_feat = pickle.load(input_file)

            X_dev_a = log_mel_feat["X_dev_a"]
            y_dev_a = log_mel_feat["y_dev_a"]
            X_dev_b = log_mel_feat["X_dev_b"]
            y_dev_b = log_mel_feat["y_dev_b"]

    return X_train, y_train, X_dev_a, y_dev_a, X_dev_b, y_dev_b


def get_dev_from_df(df_dev_a, df_dev_b, debug=False, pickle_name=None):

    if not pickle_name:
        pickle_name_dev = (
            "log_mel_feat_dev_"
            + str(config.n_feat)
            + "_win_"
            + str(config.win_size)
            + "_step_"
            + str(config.win_size)
            + "_norm_"
            + str(config.norm_per_sample)
            + ".pickle"
        )
    else:
        pickle_name_dev = pickle_name

    if not os.path.isfile(os.path.join(config.dir_out_MED, pickle_name_dev)):
        print("Extracting dev features...")

        X_dev_a, y_dev_a, skipped_files_dev_a, bugs_dev_a = get_feat(
            data_df=df_dev_a,
            data_dir=config.data_dir_dev_a,
            rate=config.rate,
            min_duration=config.min_duration,
            n_feat=config.n_feat,
        )
        X_dev_b, y_dev_b, skipped_files_dev_b, bugs_dev_b = get_feat(
            data_df=df_dev_b,
            data_dir=config.data_dir_dev_b,
            rate=config.rate,
            min_duration=config.min_duration,
            n_feat=config.n_feat,
        )
        X_dev_a, y_dev_a = reshape_feat(
            X_dev_a, y_dev_a, config.win_size, config.win_size
        )  # dev should be strided with step = window.
        X_dev_b, y_dev_b = reshape_feat(
            X_dev_b, y_dev_b, config.win_size, config.win_size
        )

        log_mel_feat_dev = {
            "X_dev_a": X_dev_a,
            "X_dev_b": X_dev_b,
            "y_dev_a": y_dev_a,
            "y_dev_b": y_dev_b,
        }

        if debug:
            print("Bugs dev A", bugs_dev_a)
            print("Bugs dev B", bugs_dev_b)

        with open(os.path.join(config.dir_out_MED, pickle_name_dev), "wb") as f:
            pickle.dump(log_mel_feat_dev, f)
            print(
                "Saved features to:", os.path.join(config.dir_out_MED, pickle_name_dev)
            )
    else:
        print(
            "Loading dev features found at:",
            os.path.join(config.dir_out_MED, pickle_name_dev),
        )
        with open(
            os.path.join(config.dir_out_MED, pickle_name_dev), "rb"
        ) as input_file:
            log_mel_feat = pickle.load(input_file)

            X_dev_a = log_mel_feat["X_dev_a"]
            y_dev_a = log_mel_feat["y_dev_a"]
            X_dev_b = log_mel_feat["X_dev_b"]
            y_dev_b = log_mel_feat["y_dev_b"]

    return X_dev_a, y_dev_a, X_dev_b, y_dev_b
