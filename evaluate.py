import argparse
from typing import Union

import datasets
import torch.cuda

from sdg_clf.evaluation import get_predictions, get_optimal_threshold, avg_predictions_ensemble, combine_predictions
from sdg_clf.utils import get_metrics, set_metrics_to_device, reset_metrics, update_metrics, compute_metrics, \
    load_pickle


def evaluate(method: str = "sdg_clf", tweet: bool = False, split: str = "test",
             model_types: Union[str, list[str]] = None,
             model_weights: Union[str, list[str]] = None, n_samples: int = None, overwrite: bool = False,
             print_latex: bool = False):
    if tweet:
        name_ds = "twitter"
    else:
        name_ds = "scopus"
    if method == "sdg_clf":
        predictions = []
        if tweet:
            split_val = "validation"
        else:
            split_val = "train"
        if not isinstance(model_types, list):
            model_types = [model_types]
            model_weights = [model_weights]
        for model_type, model_weight in zip(model_types, model_weights):
            prediction = get_predictions(method=method, tweet=tweet, split=split_val, model_type=model_type,
                                         model_weight=model_weight)
            predictions.append(prediction)

        labels_val = datasets.load_from_disk(f"data/processed/{name_ds}/base")[split_val]["label"]
        labels_val = torch.tensor(labels_val)
        threshold = get_optimal_threshold(predictions, labels_val, tweet=tweet)

        predictions = []
        for model_type, model_weight in zip(model_types, model_weights):
            prediction = get_predictions(method=method, tweet=tweet, split=split, model_type=model_type,
                                         model_weight=model_weight, overwrite=overwrite)
            predictions.append(prediction)
        if len(model_types) > 1:
            predictions = avg_predictions_ensemble(predictions)
        else:
            predictions = predictions[0]
        if not tweet:
            predictions = combine_predictions(predictions, threshold)[:n_samples]
        else:
            predictions = torch.stack(predictions, dim=0)
    else:
        predictions = get_predictions(method=method, tweet=tweet, split=split, n_samples=n_samples, overwrite=overwrite)
        threshold = 0.5
    ds = datasets.load_from_disk(f"data/processed/{name_ds}/base")[split]
    if method == "osdg":
        fails = load_pickle(f"predictions/fails_osdg_{name_ds}_{split}.pkl")
        ds = ds.filter(lambda example, idx: idx not in fails, with_indices=True)
    labels = torch.tensor(ds["label"][:n_samples])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = get_metrics(threshold, multilabel=True)
    set_metrics_to_device(metrics)
    reset_metrics(metrics)
    labels = labels.to(device)
    predictions = predictions.to(device)
    update_metrics(metrics, {"label": labels, "prediction": predictions})
    metrics_values = compute_metrics(metrics)
    with open("results.txt", "a") as f:
        f.write(f"\n\n\nDataset: {'Twitter' if tweet else 'Scopus'}\nSplit: {split}\nMethod : {method}\n")
        if model_types is not None:
            f.write(f"Models: {model_types}\nThreshold: {threshold}\n")
        if method != "sdg_clf":
            f.write(f"Number of samples: {n_samples}\n")
        f.write("\nOverall metrics\n")
        for k, v in metrics_values.items():
            f.write(f"{k}: {round(v.cpu().item(), 4)} ")
        f.write("\n")

    results = []
    metrics = get_metrics(threshold, multilabel=True, num_classes=False)
    del metrics["accuracy"]
    set_metrics_to_device(metrics)
    for i in range(17):
        labels_temp = labels[:, i]
        preds = predictions[:, i]
        reset_metrics(metrics)
        update_metrics(metrics, {"label": labels_temp, "prediction": preds})
        metrics_values = compute_metrics(metrics)
        results.append(metrics_values)

    with open("results.txt", "a") as f:
        f.write("\nMetrics per SDG\n")
        for i in range(17):
            f.write(f"sdg{i + 1}: ")
            for k, v in results[i].items():
                f.write(f"{k}: {round(v.cpu().item(), 4)} ")
            f.write("\n")

    if print_latex:
        for i in range(17):
            s = f"SDG{i + 1}"
            for val in results[i].values():
                s += f" & {round(val.item(), 4)}"
            s += r" \\"
            print(s)
    return results



parser = argparse.ArgumentParser()
parser.add_argument("-m", "--method", help="Method to evaluate from {sdg_clf, aurora, osdg}", default="sdg_clf")
parser.add_argument("-s", "--split",
                    help="Split to evaluate on. Twitter data has {train, validation, test} while Scoups data has {train, test}",
                    action='store_true', default=False)
parser.add_argument("-t", "--tweet", help="If True, evaluate on Twitter data, else Scopus", action="store_true",
                    default=False)
parser.add_argument("-mt", "--model_types", help="names of huggingface models e.g. -mt roberta-base roberta-large",
                    nargs="+", default=[])
parser.add_argument("-mw", "--model_weights",
                    help="names of model weight files e.g. -mw best_roberta-base.pt best_roberta-large.pt", nargs="+",
                    default=[])
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=3e-5)
parser.add_argument("-wd", "--weight_decay", help="optimizer weight decay", type=float, default=1e-2)
parser.add_argument("-n", "--n_samples", help="number of samples to evaluate on. Defaults to all samples", type=int,
                    default=None)
parser.add_argument("-p", "--print_latex", help="if the results per SDG should be printed as a latex table.",
                    action="store_true", default=False)
parser.add_argument("-o", "--overwrite", help="If true, overwrite potential existing predictions", action="store_false",
                    default=False)
args = parser.parse_args()
evaluate(method=args.method, tweet=args.tweet, split=args.split, model_types=args.model_types,
         model_weights=args.model_weights, n_samples=args.n_samples, print_latex=args.print_latex,
         overwrite=args.overwrite)
