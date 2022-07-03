import argparse
import os

import datasets
import torch
import tqdm

import sdg_clf
from sdg_clf import utils, base, evaluation


def main(dataset_name: str, split: str, model_weights: list[str] = None, model_types: list[str] = None,
         save_predictions: bool = False, overwrite: bool = False, threshold: float = 0.5):
    # load dataset from data/processed
    dataset = datasets.load_from_disk(f"data/processed/{dataset_name}/base")[split]
    prediction_paths = [f"predictions/{dataset_name}/{split}/{model_weights[i]}" + ".pkl" for i in
                        range(len(model_weights))]
    predictions = []
    for i in range(len(model_weights)):
        if os.path.exists(prediction_paths[i]) and not overwrite:
            predictions.append(utils.load_pickle(prediction_paths[i]))
        else:
            tokenizer = utils.get_tokenizer(model_types[i])
            model = sdg_clf.model.load_model(model_weights[i] + ".pt", model_types[i])
            transformer = base.Transformer(model, tokenizer)
            current_predictions = []
            for text in tqdm.tqdm(dataset["text"]):
                pred = transformer.predict(text)
                current_predictions.append(pred)
            predictions.append(current_predictions)

            if save_predictions:
                prediction_dir = f"predictions/{dataset_name}/{split}"
                os.makedirs(prediction_dir, exist_ok=True)

                path = f"{prediction_dir}/{model_weights[i]}.pkl"
                utils.save_pickle(path, current_predictions)

    # get optimal threshold
    labels = dataset["label"]
    labels = torch.tensor(labels)
    combined_predictions = []
    for i in range(len(predictions[0])):
        combined_predictions.append(
            evaluation.combine_predictions([predictions[j][i] for j in range(len(predictions))]))

    predictions = combined_predictions
    predictions = [k > threshold for k in predictions]
    predictions = [torch.any(k, dim=0).int() for k in predictions]
    predictions = torch.stack(predictions, dim=0)

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
    args = parser.parse_args()
    main(args.dataset, args.split, args.model_weights, args.model_types, args.save_predictions, args.overwrite)
