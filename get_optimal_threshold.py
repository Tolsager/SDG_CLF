import argparse

import datasets
import pandas as pd
import torch
import tqdm

import sdg_clf
from sdg_clf import base
from sdg_clf import evaluation
from sdg_clf import utils


def main(dataset_name: str, split: str, model_weights: list[str] = None, model_types: list[str] = None,
         predictions: str = None, save_predictions: bool = False):
    # load dataset from data/processed
    dataset = datasets.load_from_disk(f"data/processed/{dataset_name}/base")[split]
    if predictions is None:
        # initialize model and tokenizer to the Transformer class and get predictions
        transformer_list = []
        for model_type, model_weight in zip(model_types, model_weights):
            tokenizer = utils.get_tokenizer(model_type)
            model = sdg_clf.model.load_model(model_weight, model_type)
            transformer_list.append(base.Transformer(model, tokenizer))
        # get predictions
        predictions = []
        print("Getting predictions...")
        for text in tqdm.tqdm(dataset["text"]):
            prediction = evaluation.predict_no_threshold(text, transformers=transformer_list)
            predictions.append(prediction)
        next_number = utils.get_next_number("predictions")
        prediction_name = f"{dataset_name}_{split}_{next_number}"
        if save_predictions:
            utils.save_pickle(f"predictions/{prediction_name}.pkl", predictions, )
            print(f"Predictions saved to predictions/{prediction_name}.pkl")
    else:
        prediction_name = predictions
        prediction_path = "predictions/" + prediction_name
        if not prediction_name.endswith(".pkl"):
            prediction_path += ".pkl"
        predictions = utils.load_pickle(prediction_path)

    # get optimal threshold
    labels = dataset["label"]
    labels = torch.tensor(labels)
    optimal_threshold, best_f1_score = evaluation.get_optimal_threshold(predictions, labels)
    print(f"Optimal threshold: {round(optimal_threshold, 4)} with f1 score: {round(best_f1_score, 4)}")
    df = pd.read_csv("optimal_thresholds.csv")
    df = pd.concat(
        (df, pd.DataFrame({"predictions_name": [prediction_name], "threshold": [round(optimal_threshold, 4)]})))
    df.to_csv("optimal_thresholds.csv", index=False)

if __name__ == "__main__":
    # set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="scopus")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_weights", type=str, nargs="+")
    parser.add_argument("--model_types", type=str, nargs="+")
    parser.add_argument("--predictions", type=str, default=None)
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()
    main(args.dataset, args.split, args.model_weights, args.model_types, args.predictions, args.save_predictions)
