import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
# Evaluation with PSDS
from eval import do_eval
from evaluate import evaluate_model_timestamp, get_results
# Features
from feat_util import get_dev_from_df, get_train_dev_from_df
from keras_util import evaluate_model, train_model
from mozbnn_model import build_model


df = pd.read_csv(config.data_df) # Read metadata csv from config.data_df.

# To be kept: please do not edit the dev set: these paths select dev set A, dev set B 
# These refer to the former test sets A and B of HumBugDB

idx_dev_a = np.logical_and(df["country"] == "Tanzania", df["location_type"] == "field")
idx_dev_b = np.logical_and(df["country"] == "UK", df["location_type"] == "culture")
idx_train = np.logical_not(np.logical_or(idx_dev_a, idx_dev_b))
df_dev_a = df[idx_dev_a]
df_dev_b = df[idx_dev_b]


df_train = df[idx_train]

# Modify by addition or sub-sampling of df_train here
# df_train ...

# Assertion to check that train does NOT appear in dev:
assert (
    len(np.where(pd.concat([df_train, df_dev_a, df_dev_b]).duplicated())[0]) == 0
), "Train dataframe contains overlap with dev A, dev B"


# Creation of validation data ground truth and labels to be used in 
# temporal scoring with psds_eval.py

meta_df = df_dev_a[["id", "length"]].copy()
meta_df.rename(columns={"id": "filename", "length": "duration"}, inplace=True)
meta_df.to_csv(
    os.path.join(config.ROOT_DIR, "data", "labels", "dev", "a", "meta.csv"),
    sep="\t",
    index=False,
)


df_A = df_dev_a[df_dev_a.sound_type == "mosquito"]
gt = df_A[["id", "length"]].copy()
gt.rename(columns={"id": "filename", "length": "offset"}, inplace=True)
gt["onset"] = 0
gt["event_label"] = "mosquito"
gt.to_csv(
    os.path.join(config.ROOT_DIR, "data", "labels", "dev", "a", "gt.csv"),
    sep="\t",
    index=False,
)

### dev B
meta_df = df_dev_b[["id", "length"]].copy()
meta_df.rename(columns={"id": "filename", "length": "duration"}, inplace=True)
meta_df.to_csv(
    os.path.join(config.ROOT_DIR, "data", "labels", "dev", "b", "meta.csv"),
    sep="\t",
    index=False,
)


df_B = df_dev_b[df_dev_b.sound_type == "mosquito"]
gt = df_B[["id", "length"]].copy()
gt.rename(columns={"id": "filename", "length": "offset"}, inplace=True)
gt["onset"] = 0
gt["event_label"] = "mosquito"
gt.to_csv(
    os.path.join(config.ROOT_DIR, "data", "labels", "dev", "b", "gt.csv"),
    sep="\t",
    index=False,
)


# Step 3: Model training


if config.retrain_model:
    X_train, y_train, X_dev_a, y_dev_a, X_dev_b, y_dev_b = get_train_dev_from_df(
        df_train, df_dev_a, df_dev_b, debug=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    model, model_name_checkpoint = train_model(X_train, y_train, X_val, y_val)

    model_name_string = model_name_checkpoint.split("-e")[
        0
    ]  # Get partial string with date
    model_path = os.path.join(config.model_dir, "keras")

    for i in os.listdir(model_path):
        if (
            model_name_string in i
        ):  # Match to latest checkpoint. Better to select manually in code below after training
            path, model_name = os.path.split(i)  # get model_name from full path
    print("Using model with path, name:", model_path, model_name)
else:
    # Example in Keras
    model = build_model()
    model.load_weights(config.default_model_weights_path)
    print("Loaded model:", config.default_model_weights_path)
    model_name = os.path.split(config.default_model_weights_path)[
        -1
    ]  # Recover name of model for saving text files


# ### Optional: also visualise additional information
# This section is for visualising the `dev` data in the same fashion as `train`, as well as calculate scores over segment-based sections, where each data sample represents one window of the data. Note that here, edges of the recordings which do not fit into full windows are discarded, which causes some slight discrepancy between the timestamp evaluation methods and the segment-based feature evaluations. You may use these to help train and debug models, but be aware that the final score will be calculated using the PSDS method above.

if config.extra_eval:
    if not config.retrain_model:
        X_dev_a, y_dev_a, X_dev_b, y_dev_b = get_dev_from_df(df_dev_a, df_dev_b)

    # Generate BNN samples. Run with `n_samples` = 1 for deterministic NN, `n` >= 30 for BNN. Calculate the predictive entropy (PE), mutual information (MI), and log probabilities. Also plot the ROC curve and confusion matrix. Outputs are saved to `config.plot_dir` with `filename`. The code automatically aggregates features over the appropriate output shape depending on the feature type defined at the start of the notebook.

    # ### dev A evaluation

    y_preds_all = evaluate_model(
        model, X_dev_a, y_dev_a, 30
    )  # Predict directly over feature windows (1.92 s)
    PE, MI, log_prob = get_results(
        y_preds_all, y_dev_a, filename=model_name + "_dev_a"
    )

    # ### dev B evaluation

    y_preds_all = evaluate_model(
        model, X_dev_b, y_dev_b, 30
    )  # Predict directly over feature windows (1.92 s)
    PE, MI, log_prob = get_results(
        y_preds_all, y_dev_b, filename=model_name + "_dev_b"
    )


# The final score will be evaluated with the [PSDS](https://github.com/audioanalytic/psds_eval) scoring function,
# which is a method for extending event-based F scores to calculate area-under-curve for a range of classifier thresholds.
# These metrics are calculated below:

if config.predict_dev:
    # Evaluate over dev A
    evaluate_model_timestamp(
        ".wav",
        config.data_dir_dev_a,
        os.path.join(config.ROOT_DIR, "data", "predictions", "dev", "a"),
        20,
        model_weights_path = config.default_model_weights_path
    )

    # Evaluate over dev B
    evaluate_model_timestamp(
        ".wav",
        config.data_dir_dev_b,
        os.path.join(config.ROOT_DIR, "data", "predictions", "dev", "b"),
        20,
        model_weights_path = config.default_model_weights_path
    )

# Calculate challenge metrics
    do_eval("./data/predictions/dev/a", "./data/labels/dev/a",
     filename = os.path.join(config.plot_dir, model_name + "_psd_dev_a.png"))
    do_eval("./data/predictions/dev/b", "./data/labels/dev/b",
     filename = os.path.join(config.plot_dir, model_name + "_psd_dev_b.png"))


if config.predict_test:
    evaluate_model_timestamp(
        ".wav",
        config.data_dir_test,
        os.path.join(config.ROOT_DIR, "data", "predictions", "test"),
        2,
        model_weights_path=config.default_model_weights_path,
    )

    do_eval("./data/predictions/test", "./data/labels/test")
