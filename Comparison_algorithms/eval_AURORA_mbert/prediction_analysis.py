import pandas as pd
import datasets
import torch
import torchmetrics
import numpy as np
import csv

def get_metrics(threshold: bool = 0.95, num_classes: int =17, multilabel: bool = True):
    metrics = {
    "accuracy": {
        "goal": "maximize",
        "metric": torchmetrics.Accuracy(threshold=threshold, subset_accuracy=True, multiclass=not multilabel),
    },
    "precision": {
        "goal": "maximize",
        "metric": torchmetrics.Precision(threshold=threshold, num_classes=num_classes, multiclass=not multilabel),
    },
    "recall": {
        "goal": "maximize",
        "metric": torchmetrics.Recall(threshold=threshold, num_classes=num_classes, multiclass=not multilabel),
    },
    "f1": {
        "goal": "maximize",
        "metric": torchmetrics.F1Score(threshold=threshold, num_classes=num_classes, multiclass=not multilabel),
    },
            }
    return metrics

def compute_metrics(metrics, preds, targets):
    """
    Computes the metrics dictionary using torchmetrics

    Returns:
        Dictionary of metrics computed by torchmetrics
    """
    metric_values = {
        k: v["metric"](preds, targets) for k, v in metrics.items()
    }
    return metric_values

def compute_metrics_per_sdg(metrics, preds, targets):
    results = []
    for i in range(17):

        for val in eval_trainer.compute_metrics().values():
            s += f" & {round(val.item(), 4)}"
        s += r" \\"
        print(s)

def preds_above_threshold(data: pd.DataFrame, threshold: float):
    """Filters predictions above specified threshold.
    Returns torch of predictions
    """
    above_threshold=data.where(data > threshold, other=0)
    above_threshold = above_threshold.where(above_threshold == 0, other = 1)
    above_threshold = torch.tensor(above_threshold.values.tolist())
    return above_threshold

def classifications_metrics(preds: pd.DataFrame, labels: torch.tensor, threshold: float = 0.95, per_class: bool = False):
    preds = preds_above_threshold(preds, threshold)
    if per_class:
        metrics = get_metrics(threshold=threshold, num_classes = 1)
        performance = {}
        for i in range(17):
            performance[f'sdg{i+1}'] = compute_metrics(metrics, preds[:,i], labels[:,i])
    else:
        metrics = get_metrics(threshold=threshold, num_classes=17)
        performance = compute_metrics(metrics, preds, labels)
    return performance

if __name__ == "__main__":
    # For scopus:
    labels_true = datasets.load_from_disk("../../data/processed/scopus/base")['test']['label']
    preds_mbert = pd.read_csv('predictions/predictions_test.csv')

    # For tweets:
    #labels_true = datasets.load_from_disk("../data/processed/tweets/base")['test']['label'][:3000]
    #preds_mbert = pd.read_csv('predictions/predictions_tweets_test.csv')

    labels_true = torch.tensor(labels_true)
    preds_mbert = preds_mbert[[str(i) for i in range(1,18)]]
    results=classifications_metrics(preds_mbert, labels_true, 0.95)
    print(results)
    results_per_class=classifications_metrics(preds_mbert, labels_true, 0.95, per_class = True)
    for i in range(17):
        s = f"SDG{i + 1}"
        for met, val in results_per_class[f'sdg{i+1}'].items():
            if met == 'accuracy':
                continue
            else:
                s += f" & {round(val.item(), 4)}"
        s += r" \\"
        print(s)