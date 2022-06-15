import transformers
import os
import numpy as np
import datasets
import matplotlib.pyplot as plt

from sdg_clf.dataset_utils import get_dataset
import torch
from typing import Union
from sdg_clf.utils import prepare_long_text_input
from sdg_clf.model import load_model
from make_predictions import get_tweet_preds, get_scopus_preds
from sdg_clf.trainer import get_metrics
from sdg_clf.utils import set_metrics_to_device, reset_metrics, update_metrics, compute_metrics


def test_ensemble(model_weights: list[str], model_types: list[str], tweet: bool = True, log=False, return_f1: bool = False):
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
            # pred = predictions[0][i].mean(dim=0)
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
            # pred = predictions[0][i].mean(dim=0)
            new_predictions.append(pred)

        metrics = get_metrics(threshold=0.5, multilabel=True)
        set_metrics_to_device(metrics)
        reset_metrics(metrics)
        predictions = [k > threshold for k in new_predictions]
        predictions = [torch.any(k, dim=0).int() for k in predictions]
        predictions = torch.stack(predictions, dim=0)
        update_metrics(metrics, {"label": labels_test.to("cuda"), "prediction": predictions.to("cuda")})
        metrics_values = compute_metrics(metrics)
    # print(f"Best threshold: {threshold}")
    # print("Metrics for best threshold:")
    # print(metrics_values)
    if log:
        with open("results_ensemble.txt", "a") as f:
            f.write(f"""
Dataset: {"Tweets" if tweet else "Scopus"}
Model types: {model_types}
best threshold: {threshold}
metrics: {metrics_values}\n""")

    # metrics pr. SDG
    results = []
    metrics = get_metrics(threshold, multilabel=True, num_classes=False)
    del metrics["accuracy"]
    set_metrics_to_device(metrics)
    for i in range(17):
        labels = labels_test[:, i]
        preds = predictions[:, i]
        reset_metrics(metrics)
        update_metrics(metrics, {"label": labels.to("cuda"), "prediction": preds.to("cuda")})
        metrics_values = compute_metrics(metrics)
        results.append(metrics_values)

    if log:
        with open("results_ensemble.txt", "a") as f:
            for i in range(17):
                f.write(f"sdg{i+1}: {results[i]}\n")

    for i in range(17):
        s = f"SDG{i+1}"
        for val in results[i].values():
            s += f" & {round(val.item(),4)}"
        s += r" \\"
        print(s)

    if return_f1:
        f1 = [i["f1"].cpu().item() for i in results]
        return f1

def plot_f1_per_sdg(f1_scores: list[list[float]]):
    fig, ax = plt.subplots()
    ax.bar(range(17), np.random.random(17), width=0.2, color="b")
    ax.bar(np.array(range(17)) - 0.2, np.random.random(17), width=0.2, color="r")
    plt.show()
def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())

    ax.set_xticks(range(17),[str(i) for i in range(1, 18)])
    ax.set_title("F1-score per SDG on the Twitter test set")
    ax.set_xlabel("SDG")
    ax.set_ylabel("F1")
    # ax.set_xticklabels([str(i) for i in range(1, 18)])


if __name__ == "__main__":
    # test_ensemble(["best_deberta.pt", "best_roberta-large.pt"], model_types=["microsoft/deberta-v3-large", "roberta-large"], log=True)
    # test_ensemble(["best_albert.pt", "best_deberta.pt", "best_roberta-large.pt"], model_types=["albert-large-v2", "microsoft/deberta-v3-large", "roberta-large"], log=True)
    # test_ensemble(["best_deberta.pt", "best_roberta-large.pt"], model_types=["microsoft/deberta-v3-large", "roberta-large"], tweet=False, log=True)
    # test_ensemble(["best_albert.pt", "best_deberta.pt", "best_roberta-large.pt"], model_types=["albert-large-v2", "microsoft/deberta-v3-large", "roberta-large"], tweet=False, log=True)
    # test_ensemble(["best_albert.pt"], model_types=["albert-large-v2"], log=True, tweet=False)
    # test_ensemble(["best_deberta.pt"], model_types=["microsoft/deberta-v3-large"], log=True, tweet=True)
    # test_ensemble(["best_deberta.pt"], model_types=["microsoft/deberta-v3-large"], log=True, tweet=False)
    f1_roberta = test_ensemble(["best_roberta.pt"], model_types=["roberta-large"], log=False, tweet=False, return_f1=True)
    f1_deberta = test_ensemble(["best_deberta.pt"], model_types=["microsoft/deberta-v3-large"], log=False, tweet=True, return_f1=True)
    f1_albert = test_ensemble(["best_albert.pt"], model_types=["albert-large-v2"], log=False, tweet=True, return_f1=True)
    f1_ensemble = test_ensemble(["best_albert.pt", "best_deberta.pt", "best_roberta-large.pt"], model_types=["albert-large-v2", "microsoft/deberta-v3-large", "roberta-large"], log=False, return_f1=True, tweet=True)

    # print()
    fig, ax = plt.subplots()
    bar_plot(ax, data={"RoBERTa-large": f1_roberta, "DeBERTa": f1_deberta, "ALBERT": f1_albert, "Ensemble": f1_ensemble})
    plt.show()

