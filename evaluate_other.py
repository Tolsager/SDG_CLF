import argparse
import os
from typing import Union

import torch

from sdg_clf import utils, evaluation, dataset_utils


def get_prediction_path(method: str, dataset_name: str, split: str) -> str:
    prediction_path = f"predictions/{dataset_name}/{split}/{method}.pkl"

    return prediction_path


def load_predictions(prediction_path: str) -> Union[torch.Tensor, None]:
    if os.path.exists(prediction_path):
        return utils.load_pickle(prediction_path)
    else:
        return None


def main(method: str, dataset_name: str, split: str, save_predictions: bool = True, overwrite: bool = False):
    # load predictions that already exists if overwrite is False
    prediction_paths = get_prediction_path(method, dataset_name, split)
    if not overwrite:
        predictions = load_predictions(prediction_paths)
    else:
        predictions = None

    # load dataframe
    df = dataset_utils.get_processed_df(dataset_name, split)

    # create predictions if any are missing
    if predictions is None:
        # get the text column of the df as a list of strings
        samples = df["text"].tolist()
        if method == "osdg":
            predictions = evaluation.predict_multiple_samples_osdg(samples)
        elif method == "aurora":
            predictions = evaluation.predict_multiple_samples_aurora(samples)

        if save_predictions:
            utils.save_pickle(prediction_paths, predictions)

    labels = dataset_utils.get_labels_tensor(df)

    metrics = utils.get_metrics(0.5)
    utils.update_metrics(metrics, {"label": labels, "prediction": predictions})
    metrics_values = utils.compute_metrics(metrics)
    print(metrics_values)


if __name__ == "__main__":
    # set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="sdg_clf")
    parser.add_argument("--dataset_name", type=str, default="scopus")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.dataset_name, split=args.split, save_predictions=args.save_predictions, overwrite=args.overwrite)
