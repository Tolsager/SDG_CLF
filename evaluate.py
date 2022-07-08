import argparse
import os

import torch

import sdg_clf
from sdg_clf import utils, evaluation, dataset_utils


def main(dataset_name: str, split: str, model_weights: list[str] = None, model_types: list[str] = None,
         save_predictions: bool = False, overwrite: bool = False, threshold: float = 0.5, method: str = "sdg_clf"):
    if method == "sdg_clf":
        prediction_paths = [f"predictions/{dataset_name}/{split}/{model_weights[i]}.pkl" for i in
                            range(len(model_weights))]
    else:
        prediction_paths = [f"predictions/{dataset_name}/{split}/{method}.pkl"]

    # load predictions that already exists
    predictions = []
    for i in range(len(prediction_paths)):
        if os.path.exists(prediction_paths[i]) and not overwrite:
            predictions.append(utils.load_pickle(prediction_paths[i]))
        else:
            predictions.append(None)
    dataset = dataset_utils.load_preprocessed_dataset(dataset_name, split)

    # create predictions if any are missing
    if None in predictions or overwrite:
        for i in range(len(predictions)):
            if predictions[i] is None:
                if method == "sdg_clf":
                    predictions[i] = evaluation.predict_dataset(dataset, method=method, model_weight=model_weights[i],
                                                                model_type=model_types[i])
                else:
                    predictions[i] = evaluation.predict_dataset(dataset, method=method)
                if save_predictions:
                    utils.save_pickle(prediction_paths[i], predictions[i])
    if len(predictions) > 1:
        combined_predictions = []
        for i in range(len(predictions[0])):
            combined_predictions.append(
                evaluation.combine_predictions([predictions[j][i] for j in range(len(predictions))]))
        predictions = combined_predictions

    if method == "sdg_clf":
        if dataset_name == "osdg":
            predictions = [torch.sum(k, dim=0) for k in predictions]
            new_predictions = torch.zeros((len(predictions), 17))
            for i in range(len(predictions)):
                new_predictions[i][predictions[i].argmax()] = 1
            predictions = new_predictions
        else:
            predictions = [k > threshold for k in predictions]
            predictions = [torch.any(k, dim=0).int() for k in predictions]
            predictions = torch.stack(predictions, dim=0)
    labels = dataset["label"]
    labels = torch.tensor(labels)

    metrics = utils.get_metrics(threshold=threshold, multilabel=True)

    utils.update_metrics(metrics, {"label": labels, "prediction": predictions})
    metrics_values = utils.compute_metrics(metrics)
    print(metrics_values)


if __name__ == "__main__":
    # set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="scopus")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_weights", type=str, nargs="+")
    parser.add_argument("--model_types", type=str, nargs="+")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--method", type=str, default="sdg_clf")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args.dataset, split=args.split, model_weights=args.model_weights, model_types=args.model_types,
         save_predictions=args.save_predictions, overwrite=args.overwrite, method=args.method, threshold=args.threshold)
