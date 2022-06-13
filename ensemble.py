import transformers
import os
import numpy as np
import datasets

from sdg_clf.dataset_utils import get_dataset
import torch
from typing import Union
from sdg_clf.utils import prepare_long_text_input
from sdg_clf.model import load_model
from sdg_clf.make_predictions import get_tweet_preds, get_scopus_preds
from sdg_clf.trainer import get_metrics
from sdg_clf.utils import set_metrics_to_device, reset_metrics, update_metrics, compute_metrics


def test_ensemble(model_weights: list[str], model_types: list[str], tweet: bool = True):
    """Load finetuned models and predict with their mean.
    The optimal threshold is found on one set and evaluated on the test set

    Args:
        model_weights: names of the model weights e.g. best_albert.pt
        model_types: huggingface models e.g. roberta-large
        tweet:

    Returns:

    """
    if tweet:
        split_val = "validation"
        predictor = get_tweet_preds
        labels_val = datasets.load_from_disk("data/processed/tweets/base")[split_val]["label"]
    else:
        split_val = "train"
        predictor = get_scopus_preds
        labels_val = datasets.load_from_disk("data/processed/scopus/base")[split_val]["label"]
    split_test = "test"
    labels_val = torch.tensor(labels_val)

    # predictions for each model
    predictions = []
    for weights, model_type in zip(model_weights, model_types):
        prediction = predictor(model_type, split_val, weights=weights)
        predictions.append(prediction)

    # get optimal threshold
    if tweet:
        predictions = torch.stack(predictions, dim=-1)
        predictions = torch.mean(predictions, dim=-1)

        best_f1 = (0, 0)
        for threshold in np.linspace(0, 1, 101):
            metrics = get_metrics(threshold=threshold, multilabel=True)
            set_metrics_to_device(metrics)
            reset_metrics(metrics)
            update_metrics(metrics, {"label": labels_val.to("cuda"), "prediction": predictions.to("cuda")})
            metrics_values = compute_metrics(metrics)
            if metrics_values["f1"] > best_f1[1]:
                best_f1 = (threshold, metrics_values["f1"])
    else:
        new_predictions = []
        for i in range(len(predictions[0])):
            # each model can have different number of sequences even though they have the same max length
            # if model0 has 2 predictions and model1 has 1 prediction, we only use the first predictions from both
            min_size = min([p[i].shape[0] for p in predictions])
            total = predictions[0][i][:min_size]
            for l in range(1, len(predictions)):
                total += predictions[l][i][:min_size]
            pred = total / len(predictions)
            new_predictions.append(pred)

        best_f1 = (0, 0)
        metrics = get_metrics(threshold=0.5, multilabel=True)
        set_metrics_to_device(metrics)
        for threshold in np.linspace(0, 1, 101):
            reset_metrics(metrics)
            predictions = [k > threshold for k in new_predictions]
            predictions = [torch.any(k, dim=0).int() for k in predictions]
            predictions = torch.stack(predictions, dim=0)
            update_metrics(metrics, {"label": labels_val.to("cuda"), "prediction": predictions.to("cuda")})
            metrics_values = compute_metrics(metrics)
            if metrics_values["f1"] > best_f1[1]:
                best_f1 = (threshold, metrics_values["f1"])

    # optimal threshold
    threshold = best_f1[0]

    # predictions from test set
    predictions = []
    for weights, model_type in zip(model_weights, model_types):
        prediction = predictor(model_type, split_test, weights=weights)
        predictions.append(prediction)

    if tweet:
        predictions = torch.stack(predictions, dim=-1)
        predictions = torch.mean(predictions, dim=-1)
        labels_test = datasets.load_from_disk("data/processed/tweets/base")[split_test]["label"]
        labels_test = torch.tensor(labels_test)
        metrics = get_metrics(threshold=threshold, multilabel=True)
        set_metrics_to_device(metrics)
        reset_metrics(metrics)
        update_metrics(metrics, {"label": labels_test.to("cuda"), "prediction": predictions.to("cuda")})
        metrics_values = compute_metrics(metrics)
    else:
        labels_test = datasets.load_from_disk("data/processed/scopus/base")[split_test]["label"]
        labels_test = torch.tensor(labels_test)
        new_predictions = []
        for i in range(len(predictions[0])):
            min_size = min([p[i].shape[0] for p in predictions])
            total = predictions[0][i][:min_size]
            for l in range(1, len(predictions)):
                total += predictions[l][i][:min_size]
            pred = total / len(predictions)
            new_predictions.append(pred)

        metrics = get_metrics(threshold=0.5, multilabel=True)
        set_metrics_to_device(metrics)
        reset_metrics(metrics)
        predictions = [k > threshold for k in new_predictions]
        predictions = [torch.any(k, dim=0).int() for k in predictions]
        predictions = torch.stack(predictions, dim=0)
        update_metrics(metrics, {"label": labels_test.to("cuda"), "prediction": predictions.to("cuda")})
        metrics_values = compute_metrics(metrics)
    print(f"Best threshold: {threshold}")
    print("Metrics for best threshold:")
    print(metrics_values)
    with open("results_ensemble.txt", "a") as f:
        f.write(f"""
Dataset: {"Tweets" if tweet else "Scopus"}
Model types: {model_types}
best threshold: {threshold}
metrics: {metrics_values}\n""")

    # metrics pr. SDG
    results = []
    metrics = get_metrics(threshold, multilabel=True, num_classes=False)
    set_metrics_to_device(metrics)
    for i in range(17):
        labels = labels_test[:, i]
        preds = predictions[:, i]
        reset_metrics(metrics)
        update_metrics(metrics, {"label": labels.to("cuda"), "prediction": preds.to("cuda")})
        metrics_values = compute_metrics(metrics)
        results.append(metrics_values)

    with open("results_ensemble.txt", "a") as f:
        for i in range(17):
            f.write(f"sdg{i+1}: {results[i]}\n")





if __name__ == "__main__":
    test_ensemble(["best_deberta.pt", "best_roberta-large.pt"], model_types=["microsoft/deberta-v3-large", "roberta-large"])
    test_ensemble(["best_albert.pt", "best_deberta.pt", "best_roberta-large.pt"], model_types=["albert-large-v2", "microsoft/deberta-v3-large", "roberta-large"])
    test_ensemble(["best_deberta.pt", "best_roberta-large.pt"], model_types=["microsoft/deberta-v3-large", "roberta-large"], tweet=False)
    test_ensemble(["best_albert.pt", "best_deberta.pt", "best_roberta-large.pt"], model_types=["albert-large-v2", "microsoft/deberta-v3-large", "roberta-large"], tweet=False)
