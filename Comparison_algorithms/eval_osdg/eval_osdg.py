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
    ds_dict = datasets.load_from_disk("../../data/processed/scopus/base")
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

def convert_predictions(predictions):
    predictions_new = []
    for prediction in predictions:
        prediction = ast.literal_eval(prediction)
        prediction_list = [0] * 17
        for pred in prediction:
            sdg_pred = int(pred[0][4:]) - 1
            prediction_list[sdg_pred] = 1
        predictions_new.append(prediction_list)
    return predictions_new

def get_tweet_predictions(api: str, split: str = "validation", test: bool = False, min_length: int = 40, path_save: str = "osdg_predictions_tweets"):
    if os.path.exists(path_save):
        choice = input("File 'osdg_predictions_tweets' already exists. \n Type '1' to change save path\n Type '2' to overwrite\n")
        if choice == "1":
            path_save = input("Type a new file name: ")
    ds_dict = datasets.load_from_disk("../../data/processed/tweets/base")
    ds = ds_dict[split]
    ds = ds.filter(lambda example: len(example['text'].split()) > min_length)
    ds = ds.select([i for i in range(3000)])
    predictions = []
    if api == "first":
        ip = ip1
    elif api == "second":
        ip = ip2
    if test:
        ds = ds.select([0, 1, 2])
    failed_predictions = []
    idx = 0
    for sample in tqdm(ds):
        text = sample["text"]
        data = {"query": text}
        prediction = requests.post(ip, data=data)
        prediction = prediction.text
        prediction = ast.literal_eval(prediction)
        prediction_list = [0] * 17
        try:
            for pred in prediction:
                sdg_pred = int(pred[0][4:]) - 1
                prediction_list[sdg_pred] = 1
            predictions.append(prediction_list)
        except:
            failed_predictions.append(idx)
        idx += 1
        #predictions.append(prediction)
    with open(path_save, "wb+") as f:
        pickle.dump(predictions, f)
    return failed_predictions


def score_predictions(metrics: dict, path_predictions: str, split: str = 'train', test: bool = False, tweet: bool = False, fails: list = [], min_length: int = 40):
    for metric in metrics.values():
        metric.reset()
    if tweet:
        ds_dict = datasets.load_from_disk("../../data/processed/tweets/base")
        ds = ds_dict[split]
        ds = ds.filter(lambda example: len(example['text'].split())>min_length)
        ds = ds.select([i for i in range(3000)])
        ds = ds.filter(lambda example, indice: indice not in fails, with_indices=True)
    else:
        ds_dict = datasets.load_from_disk("../../data/processed/scopus/base")
        ds = ds_dict[split]
    if test:
        ds = ds.select([0, 1, 2])
    with open(path_predictions, "rb") as f:
        predictions = pickle.load(f)
    if not tweet:
        predictions = convert_predictions(predictions)
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

    ds_dict = datasets.load_from_disk("../../data/processed/scopus/base")
    ds = ds_dict["train"]
    texts = ds["Abstract"]
    labels = ds["label"]
    for i in range(len(predictions)):
        text = texts[i]
        label = labels[i]
        prediction = predictions[i]


if __name__ == "__main__":
    # debug_osdg()
    #get_scopus_predictions(api="first", )
    #failed_predictions = get_tweet_predictions(api="first", split="test", path_save="osdg_predictions_tweets_test")
    #assert False
    multilabel = True
    metrics = {
        "accuracy":
            torchmetrics.Accuracy(subset_accuracy=True, multiclass=not multilabel),
        "precision":
            torchmetrics.Precision(num_classes=17, multiclass=not multilabel),
        "recall":
            torchmetrics.Recall(num_classes=17, multiclass=not multilabel),
        "f1":
            torchmetrics.F1Score(num_classes=17, multiclass=not multilabel),
    }
    results = score_predictions(metrics=metrics, path_predictions="osdg_predictions_scopus_test.pkl", split="test")
    print(results)
    # #print(failed_predictions) #85 failed predictions in the range 0:3000 for test
    # failed_predictions = [3, 61, 103, 175, 177, 232, 233, 255, 288, 312, 345, 355, 407, 472, 522, 564, 565, 659, 680, 764, 779, 792, 817, 829, 839, 880, 927, 941, 967, 998, 1014, 1027, 1061, 1074, 1103, 1193, 1201, 1219, 1246, 1253, 1254, 1266, 1308, 1397, 1402, 1463, 1468, 1536, 1556, 1625, 1628, 1681, 1700, 1741, 1744, 1761, 1868, 1915, 1967, 2078, 2094, 2113, 2218, 2323, 2387, 2445, 2517, 2586, 2607, 2615, 2637, 2667, 2692, 2712, 2718, 2742, 2751, 2776, 2798, 2828, 2858, 2885, 2908, 2926, 2958]
    # results = score_predictions(metrics, "osdg_predictions", split ="test", tweets = False)
    # # results = score_predictions(metrics, "osdg_predictions_tweets_test", split = "test", tweets = True, fails = failed_predications)
    # print(results)
    # get_scopus_predictions(api="first")
    # get_scopus_predictions(api="first", split="test")
    # output:
    # {'accuracy': tensor(0.0683), 'auroc': tensor(0.7089), 'precision': tensor(0.2409), 'recall': tensor(0.5780), 'f1': tensor(0.3400)}
