import argparse
import os
import torch

from sdg_clf import utils, evaluation, dataset_utils


def main(dataset_name: str, split: str, model_weights: list[str] = None, model_types: list[str] = None,
         save_predictions: bool = True, overwrite: bool = False, threshold: float = 0.5):
    os.chdir(os.path.dirname(__file__))

    predictions = evaluation.get_raw_predictions_sdg_clf(dataset_name, split, model_weights, save_predictions,
                                                         overwrite)
    if len(model_weights) > 1:
        predictions = predictions[0]
    else:
        predictions = evaluation.combine_multiple_predictions(predictions)
    predictions = evaluation.threshold_multiple_predictions(predictions, threshold)
    predictions = evaluation.predict_multiple_strategy_any(predictions)
    df = dataset_utils.get_processed_df(dataset_name, split)
    labels = dataset_utils.get_labels_tensor(df)

    metrics = utils.get_metrics(threshold=threshold)

    utils.update_metrics(metrics, {"label": labels, "prediction": predictions})
    metrics_values = utils.compute_metrics(metrics)
    utils.print_metrics(metrics_values)


if __name__ == "__main__":
    # set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="scopus")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--model_weights", type=str, nargs="+")
    parser.add_argument("--model_types", type=str, nargs="+")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args.dataset_name, split=args.split, model_weights=args.model_weights, model_types=args.model_types,
         save_predictions=args.save_predictions, overwrite=args.overwrite, threshold=args.threshold)
