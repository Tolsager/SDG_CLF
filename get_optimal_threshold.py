import argparse
import os

import torch

from sdg_clf import utils, evaluation, dataset_utils, modelling, base


def get_prediction_paths(dataset_name: str, split: str, model_weights: list[str] = None) -> list[str]:
    prediction_paths = [f"predictions/{dataset_name}/{split}/{model_weights[i]}.pkl" for i in
                        range(len(model_weights))]
    return prediction_paths


def load_predictions(prediction_paths: list[str]) -> list[torch.Tensor]:
    predictions = []
    for i in range(len(prediction_paths)):
        if os.path.exists(prediction_paths[i]):
            predictions.append(utils.load_pickle(prediction_paths[i]))
        else:
            predictions.append(None)
    return predictions


def main(dataset_name: str, split: str, model_weights: list[str] = None, model_types: list[str] = None,
         save_predictions: bool = False, overwrite: bool = False, threshold: float = 0.5):
    if dataset_name == "osdg":
        raise ValueError("The OSDG dataset only has a single label for each sample so no threshold is used")
    # load predictions that already exist if overwrite is False
    n_models = len(model_types)
    prediction_paths = get_prediction_paths(dataset_name, split, model_weights)
    if not overwrite:
        predictions = load_predictions(prediction_paths)
    else:
        predictions = [None] * n_models

    # load dataframe
    df = dataset_utils.get_processed_df(dataset_name, split)
    samples = df["processed_text"].tolist()

    # create predictions if any are missing
    for i in range(len(predictions)):
        if predictions[i] is None:
            # load the model
            transformer_model = modelling.load_model(model_types[i], model_weights[i])
            tokenizer = utils.get_tokenizer(model_types[i])
            transformer = base.Transformer(transformer_model, tokenizer)
            predictions[i] = transformer.predict_multiple_samples_no_threshold(samples)

            if save_predictions:
                utils.save_pickle(prediction_paths[i], predictions[i])

    # combine the predictions if an ensemble of models is used
    if len(predictions) > 1:
        predictions = evaluation.combine_multiple_predictions(predictions)

    labels = dataset_utils.get_labels_tensor(df)

    best_threshold, best_f1 = evaluation.get_optimal_threshold(predictions, labels)
    print(f"Best threshold: {best_threshold}")
    print(f"Best F1: {best_f1}")


if __name__ == "__main__":
    # set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="scopus")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_weights", type=str, nargs="+")
    parser.add_argument("--model_types", type=str, nargs="+")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args.dataset_name, split=args.split, model_weights=args.model_weights, model_types=args.model_types,
         save_predictions=args.save_predictions, overwrite=args.overwrite, threshold=args.threshold)
