import ast
import numpy as np
import os

import datasets
import requests
import torch
from tqdm import tqdm

from .aurora_mbert import create_aurora_predictions
from .dataset_utils import create_base_dataset, get_dataloader, get_tokenizer
from .model import load_model
from .osdg_ip import ip1
from .utils import load_pickle, save_pickle, prepare_long_text_input, reset_metrics, update_metrics, compute_metrics, \
    get_metrics, set_metrics_to_device


def get_predictions(method: str = "sdg_clf", tweet: bool = False, split: str = "test", model_type: str = None,
                    model_weight: str = None, n_samples: int = None, overwrite: bool = False):
    """
    loads or creates predictions on scopus or twitter data

    Args:
        method: {"sdg_clf", "osdg", "aurora"}
        tweet: if Scopus or Twitter data
        split: dataset split
        model_type: huggingface model type such as "roberta-large"
        model_weight: name of a weight file in finetuned_models e.g. "best_roberta.pt"
        n_samples: number of samples to evaluate on. Ignored when method is "sdg_clf"
        overwrite: if predictions should be created and overwrite previous predictions if they exist

    Returns:

    """
    if tweet:
        name_ds = "twitter"
    else:
        name_ds = "scopus"
    if method == "sdg_clf":
        path_predictions = f"predictions/{model_type}/{name_ds}_{split}.pkl"
    else:
        path_predictions = f"predictions/{method}_{name_ds}_{split}.pkl"

    # if predictions already exist, load them
    if os.path.exists(path_predictions) and not overwrite:
        predictions = load_pickle(path_predictions)
    else:
        # check if base dataset exists else create it
        path_base = f"data/processed/{name_ds}/base"
        if not os.path.exists(path_base):
            create_base_dataset(tweet=tweet)

        if method == "sdg_clf":
            if tweet:
                batch_size = 20
            else:
                tokenizer = get_tokenizer(model_type)
                batch_size = 1
            dataloader = get_dataloader(split, model_type, tweet=tweet, batch_size=batch_size)
            model = load_model(model_weight, model_type)
            model.eval()
            predictions = []
            device = "cuda" if torch.cuda.is_available() else "cpu"
            for sample in tqdm(dataloader):
                with torch.no_grad():
                    if tweet:
                        pred = model(sample["input_ids"].to(device),
                                     sample["attention_mask"].to(device)).logits.sigmoid()
                        predictions.append(pred)
                    else:
                        iids = sample["input_ids"]
                        model_in = prepare_long_text_input(iids, tokenizer=tokenizer)
                        model_out = model(**model_in).logits.sigmoid()
                        predictions.append(model_out)
            if tweet:
                predictions = torch.concat(predictions, dim=0)
        elif method == "osdg":
            predictions, failed_predictions = create_osdg_predictions(tweet=tweet, split=split, n_samples=n_samples)
            save_pickle(f"predictions/fails_osdg_{name_ds}_{split}.pkl", failed_predictions)

        elif method == "aurora":
            predictions = create_aurora_predictions(tweet=tweet, split=split, n_samples=n_samples)
        save_pickle(path_predictions, predictions)
    return predictions


def create_osdg_predictions(tweet: bool = False, split: str = "test", n_samples: int = 1400):
    if tweet:
        name_ds = "twitter"
        name_text = "text"
    else:
        name_ds = "scopus"
        name_text = "Abstract"
    ds_dict = datasets.load_from_disk(f"data/processed/{name_ds}/base")
    ds = ds_dict[split]
    ip = ip1
    if tweet:
        ds = ds.filter(lambda example: len(example[name_text].split()) > 40)
        ds = ds.select([i for i in range(3000)])

    idx = 0
    pbar = tqdm(total=n_samples)
    predictions = []
    failed_predictions = []
    while len(predictions) < n_samples:
        sample = ds[idx]
        text = sample[name_text]
        data = {"query": text}
        prediction = requests.post(ip, data=data)
        prediction = prediction.text
        prediction = ast.literal_eval(prediction)
        prediction_list = [0] * 17
        try:
            for pred in prediction:
                sdg_pred = int(pred[0][4:]) - 1
                prediction_list[sdg_pred] = 1
            prediction_tensor = torch.tensor(prediction_list)
            predictions.append(prediction_tensor)
            pbar.update(1)
        except:
            failed_predictions.append(idx)
        idx += 1
    predictions = torch.stack(predictions, dim=0)
    return predictions, failed_predictions


def get_optimal_threshold(predictions: torch.Tensor, labels: torch.Tensor, tweet: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tweet:
        predictions = torch.stack(predictions, dim=-1)
        predictions = torch.mean(predictions, dim=-1)

        best_f1 = (0, 0)
        for threshold in np.linspace(0, 1, 101):
            metrics = get_metrics(threshold=threshold, multilabel=True)
            set_metrics_to_device(metrics)
            reset_metrics(metrics)
            update_metrics(metrics, {"label": labels.to(device), "prediction": predictions.to(device)})
            metrics_values = compute_metrics(metrics)
            if metrics_values["f1"] > best_f1[1]:
                best_f1 = (threshold, metrics_values["f1"])
    else:
        predictions = avg_predictions_ensemble(predictions)

        best_f1 = (0, 0)
        metrics = get_metrics(threshold=0.5, multilabel=True)
        set_metrics_to_device(metrics)
        for threshold in np.linspace(0, 1, 101):
            reset_metrics(metrics)
            predictions_temp = combine_predictions(predictions, threshold)
            update_metrics(metrics, {"label": labels.to(device), "prediction": predictions_temp.to(device)})
            metrics_values = compute_metrics(metrics)
            if metrics_values["f1"] > best_f1[1]:
                best_f1 = (threshold, metrics_values["f1"])

    # optimal threshold
    threshold = best_f1[0]
    return threshold

def avg_predictions_ensemble(predictions: list[torch.Tensor]):
    new_predictions = []
    for i in range(len(predictions[0])):
        min_size = min([p[i].shape[0] for p in predictions])
        total = predictions[0][i][:min_size]
        for l in range(1, len(predictions)):
            total += predictions[l][i][:min_size]
        pred = total / len(predictions)
        new_predictions.append(pred)
    return new_predictions


def combine_predictions(predictions: list[torch.Tensor], threshold: float):
    predictions = [k > threshold for k in predictions]
    predictions = [torch.any(k, dim=0).int() for k in predictions]
    predictions = torch.stack(predictions, dim=0)
    return predictions
