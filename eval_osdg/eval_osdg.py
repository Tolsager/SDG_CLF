import requests
from osdg_ip import ip1, ip2
import datasets
from tqdm import tqdm
import torch
import ast
import pickle
import torchmetrics
import os


def get_scopus_predictions(api: str, split: str = "train", test: bool = False):
    path_save = "osdg_predictions"
    if os.path.exists(path_save):
        choice = input("File 'osdg_predictions' already exists. \n Type '1' to change save path\n Type '2' to overwrite\n")
        if choice == "1":
            path_save = input("Type a new file name: ")
    ds_dict = datasets.load_from_disk("../data/processed/scopus/base")
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
        # prediction = ast.literal_eval(prediction)
        # prediction_list = [0] * 17
        # for pred in prediction:
        #     sdg_pred = int(pred[0][4:]) - 1
        #     prediction_list[sdg_pred] = 1



        # predictions.append(prediction_list)
        predictions.append(prediction)
    with open(path_save, "wb+") as f:
        pickle.dump(predictions, f)


def score_predictions(metrics: dict, path_predictions: str, test: bool = False):
    for metric in metrics.values():
        metric.reset()
    ds_dict = datasets.load_from_disk("../data/processed/scopus/base")
    ds = ds_dict["train"]
    if test:
        ds = ds.select([0, 1])
    with open(path_predictions, "rb") as f:
        predictions = pickle.load(f)
    predictions = torch.tensor(predictions)
    labels = ds["label"]
    labels = torch.tensor(labels)

    results = {}
    for name, metric in metrics.items():
        metric(predictions, labels)
        results[name] = metric.compute()
    return results


def debug_osdg():
    with open("osdg_predictions", "rb") as f:
        predictions = pickle.load(f)

    ds_dict = datasets.load_from_disk("../data/processed/scopus/base")
    ds = ds_dict["train"]
    texts = ds["Abstract"]
    labels = ds["label"]
    for i in range(len(predictions)):
        text = texts[i]
        label = labels[i]
        prediction = predictions[i]


if __name__ == "__main__":
    # debug_osdg()
    get_scopus_predictions(api="first", )
    assert False
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
    results = score_predictions(metrics, "osdg_predictions",)
    print(results)
