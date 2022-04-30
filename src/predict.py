import argparse
import os

import librosa
import numpy as np
import pandas as pd

import config
import util
from mozbnn_model import build_model


def get_output_file_name(
    root_out, filename, audio_format, step_size, threshold
):
    if filename.endswith(audio_format):
        output_filename = filename[
            :-4
        ]  # remove file extension for renaming to other formats.
    else:
        output_filename = filename  # no file extension present
    return (
        os.path.join(root_out, output_filename)
        + "_BNN_step_"
        + str(step_size)
        + "_"
        + f"{threshold:.1f}"
        + ".txt"
    )


def write_output(
    data_path,
    predictions_path,
    audio_format,
    model_weights_path,
    det_threshold=np.arange(0.1, 1.1, 0.1),
    n_samples=10,
    feat_type="log-mel",
    n_feat=config.n_feat,
    win_size=config.win_size,
    step_size=config.win_size,
    n_hop=config.n_hop,
    sr=config.rate,
    norm_per_sample=config.norm_per_sample,
    debug=False,
):
    model = build_model()
    model.load_weights(model_weights_path)
    print("Loaded model:", model_weights_path)
    mozz_audio_list = []

    print("Processing:", data_path, "for audio format:", audio_format)

    i_signal = 0
    for root, dirs, files in os.walk(data_path):
        for filename in files:
            if audio_format not in filename:
                continue
            print(root, filename)
            i_signal += 1
            try:
                x, x_l = util.get_wav_for_path_pipeline(
                    [os.path.join(root, filename)], sr=sr
                )
                if debug:
                    print(filename + " signal length", x_l)
                if x_l < (n_hop * win_size) / sr:
                    print("Signal length too short, skipping:", x_l, filename)
                else:
                    #
                    X_CNN = util.get_feat(
                        x,
                        sr=sr,
                        feat_type=feat_type,
                        n_feat=n_feat,
                        norm_per_sample=norm_per_sample,
                        flatten=False,
                    )

                    X_CNN = util.reshape_feat(
                        X_CNN, win_size=win_size, step_size=step_size
                    )
                    out = []
                    for i in range(n_samples):
                        out.append(model.predict(X_CNN))

                    G_X, U_X, _ = util.active_BALD(np.log(out), X_CNN, 2)

                    y_to_timestamp = np.repeat(np.mean(out, axis=0), step_size, axis=0)
                    G_X_to_timestamp = np.repeat(G_X, step_size, axis=0)
                    U_X_to_timestamp = np.repeat(U_X, step_size, axis=0)

                    root_out = root.replace(data_path, predictions_path)
                    print("dir_out", root_out, "filename", filename)

                    if not os.path.exists(root_out):
                        os.makedirs(root_out)

                    # Iterate over threshold for threshold-independent metrics

                    for th in det_threshold:
                        preds_list = util.detect_timestamps_BNN(
                            y_to_timestamp,
                            G_X_to_timestamp,
                            U_X_to_timestamp,
                            det_threshold=th,
                        )

                        if debug:
                            print(preds_list)
                            for times in preds_list:
                                mozz_audio_list.append(
                                    librosa.load(
                                        os.path.join(root, filename),
                                        offset=float(times[0]),
                                        duration=float(times[1]) - float(times[0]),
                                        sr=sr,
                                    )[0]
                                )

                        text_output_filename = get_output_file_name(
                            root_out,
                            filename,
                            audio_format,
                            step_size,
                            th,
                        )

                        np.savetxt(
                            text_output_filename,
                            preds_list,
                            fmt="%s",
                            delimiter="\t",
                        )
                    print("Processed:", filename)
            except Exception as e:
                print(
                    "[ERROR] Unable to load {}, {} ".format(
                        os.path.join(root, filename), e
                    )
                )

    print("Total files of " + str(audio_format) + " format processed:", i_signal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This function writes the predictions of the model."""
    )
    parser.add_argument("--extension", default=".wav", type=str)
    parser.add_argument(
        "--norm",
        default=True,
        help="Normalise feature windows with respect to themselves.",
    )

    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_folder_path = os.path.join(project_path, "models")
    parser.add_argument(
        "--model_weights_path",
        default=os.path.join(
            models_folder_path,
            "Win_30_Stride_10_2022_04_27_18_45_11-e01val_accuracy0.9798.hdf5",
        ),
        type=str,
        help="Path to model weights.",
    )
    parser.add_argument(
        "--win_size", default=config.win_size, type=int, help="Window size."
    )
    parser.add_argument(
        "--step_size", default=config.win_size, type=int, help="Step size."
    )
    parser.add_argument(
        "--custom_threshold",
        help="Specify threshold between 0.0 and 1.0 for mosquito classification. Overrides default of ten threshold outputs.",
    )
    parser.add_argument(
        "--BNN_samples", default=10, type=int, help="Number of MC dropout samples."
    )

    args = parser.parse_args()

    extension = args.extension
    win_size = args.win_size
    step_size = args.step_size
    n_samples = args.BNN_samples
    norm_per_sample = args.norm
    model_weights_path = args.model_weights_path
    if args.custom_threshold is None:
        th = np.arange(0.1, 1.1, 0.1)
        print("no custom th detected")
    else:
        th = [args.custom_threshold]
        print("custom th detected")

    data_path = os.path.join(project_path, "data/audio/test")
    predictions_path = os.path.join(project_path, "data/predictions/test")

    write_output(
        data_path,
        predictions_path,
        extension,
        model_weights_path=model_weights_path,
        norm_per_sample=norm_per_sample,
        win_size=win_size,
        step_size=step_size,
        n_samples=n_samples,
        det_threshold=th,
    )

    for i, th in enumerate(th):
        df_list = []
        for filename in os.listdir(predictions_path):
            if filename.endswith(f"{th:.1f}" + ".txt"):
                df_pred = pd.read_csv(
                    os.path.join(predictions_path, filename),
                    sep="\t",
                    names=["onset", "offset", "event_label"],
                )
                filename = filename.split("_BNN_")[0]
                df_pred["event_label"] = "mosquito"
                df_pred["filename"] = filename
                df_list.append(df_pred)

        if len(df_list) > 0:
            pd.concat(df_list).to_csv(
                predictions_path + "/baseline_" + f"{th:.1f}" + ".csv",
                sep="\t",
                index=False,
            )
