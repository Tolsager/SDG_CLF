import requests
from osdg_ip import ip1, ip2
import datasets
from tqdm import tqdm
import torch
import ast
import pickle
import torchmetrics


def get_scopus_predictions(api: str, split: str = "train", test: bool = False):
    ds_dict = datasets.load_from_disk("data/processed/scopus/base")
    ds = ds_dict[split]
    predictions = []
    if api == "first":
        ip = ip1
    elif api == "second":
        ip = ip2
    if test:
        ds = ds.select([0, 1])
    for sample in tqdm(ds):
        text = sample["Abstract"]
        data = {"query": text}
        prediction = requests.post(ip, data=data)
        prediction = prediction.text
        prediction = ast.literal_eval(prediction)
        prediction_list = [0] * 17
        for pred in prediction:
            sdg_pred = int(pred[0][4:]) - 1
            prediction_list[sdg_pred] = 1
        predictions.append(prediction_list)
    with open("osdg_predictions", "wb+") as f:
        pickle.dump(predictions, f)


def score_predictions(metrics: dict, path_predictions: str, test: bool = False):
    ds_dict = datasets.load_from_disk("data/processed/scopus/base")
    ds = ds_dict["train"]
    if test:
        ds = ds.select([0, 1])
    predictions = pickle.load(path_predictions)
    predictions = torch.tensor(predictions)
    labels = ds["label"]
    labels = torch.tensor(labels)

    results = {}
    for name, metric in metrics.items():
        metric(predictions, labels)
        results[name] = metric.compute()
    return results


if __name__ == "__main__":
    get_scopus_predictions(api="first")
    multilabel = True
    metrics = {
        "accuracy":
            torchmetrics.Accuracy(subset_accuracy=True, multiclass=not multilabel),
        "auroc":
            torchmetrics.AUROC(num_classes=17),
        "precision":
            torchmetrics.Precision(num_classes=17, multiclass=not multilabel),
        "recall":
            torchmetrics.Recall(num_classes=17, multiclass=not multilabel),
        "f1":
            torchmetrics.F1Score(num_classes=17, multiclass=not multilabel),
    }
    with open("osdg_predictions", "rb") as f:
        results = score_predictions(metrics, f, test=True)
    print(results)
