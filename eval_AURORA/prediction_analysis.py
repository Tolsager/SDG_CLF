import pandas as pd
import datasets
import torch
import torchmetrics
import numpy as np
import csv

def get_metrics(threshold):
    multilabel = True
    metrics = {
    "accuracy": {
        "goal": "maximize",
        "metric": torchmetrics.Accuracy(threshold=threshold, subset_accuracy=True, multiclass=not multilabel),
    },
    "auroc": {
        "goal": "maximize",
        "metric": torchmetrics.AUROC(num_classes=17),
    },
    "precision": {
        "goal": "maximize",
        "metric": torchmetrics.Precision(threshold=threshold, num_classes=17, multiclass=not multilabel),
    },
    "recall": {
        "goal": "maximize",
        "metric": torchmetrics.Recall(threshold=threshold, num_classes=17, multiclass=not multilabel),
    },
    "f1": {
        "goal": "maximize",
        "metric": torchmetrics.F1Score(threshold=threshold, num_classes=17, multiclass=not multilabel),
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

def preds_above_threshold(data: pd.DataFrame, threshold: float):
    """Filters predictions above specified threshold.
    Returns torch of predictions
    """
    above_threshold=data.where(data > threshold, other =0)
    above_threshold = above_threshold.where(above_threshold == 0, other = 1)
    above_threshold = torch.tensor(above_threshold.values.tolist())
    return above_threshold

def classifications_metrics(preds: pd.DataFrame, labels: torch.tensor, threshold: float = 0.95):
    print(threshold)
    preds = preds_above_threshold(preds, threshold)
    metrics = get_metrics(threshold = threshold)
    performance = compute_metrics(metrics, preds, labels)
    return performance

if __name__ == "__main__":
    labels_true = datasets.load_from_disk("../data/processed/scopus/base")['test']['label']
    labels_true = torch.tensor(labels_true)
    preds_mbert = pd.read_csv('predictions/predictions_test.csv')
    preds_mbert = preds_mbert[[str(i) for i in range(1,18)]]
    results = []
    for thres in np.linspace(0.5,1,51):
        results.append(classifications_metrics(preds_mbert, labels_true, thres))

    columns = results[0].keys()

    with open("predictions/metrics_test.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for key in results:
            writer.writerow(key)