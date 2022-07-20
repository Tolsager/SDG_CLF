import ast
import os
from typing import Union

import datasets
import requests
import torch
import torchmetrics
import tqdm

import sdg_clf
from sdg_clf import base
from sdg_clf import utils
from sdg_clf.dataset_utils import create_base_dataset, get_dataloader, get_tokenizer
from sdg_clf.model import load_model
from sdg_clf import osdg_ip
from sdg_clf.utils import load_pickle, save_pickle, prepare_long_text_input
from sdg_clf import aurora_mbert


# predicts the SDG classification for a given text by choosing a method of sdg_clf, aurora, or osdg
# def predict(text: str, method: str = "sdg_clf", transformers: Union[base.Transformer, list[base.Transformer]] = None,
#             threshold: float = 0.5):
#     if method == "sdg_clf":
#         prediction = predict_sdg_clf(text, transformers=transformers, threshold=threshold)
#     elif method == "osdg":
#         prediction = predict_osdg(text)
#     elif method == "aurora":
#         prediction = predict_sample_aurora(text)
#     return prediction


def predict_sample_osdg(text: str, ip: str = osdg_ip.ip1) -> Union[torch.Tensor, None]:
    """
    Predict the SDG classification for a given text using the OSDG server.
    Args:
        text: the text to predict on
        ip: the ip address of the OSDG server

    Returns:
        the prediction from the server as a tensor of shape (17)

    """
    osdg_prediction = request_osdg_prediction(text, ip=ip)
    if "ERROR" in osdg_prediction:
        print("OSDG unable to predict")
        print("The following error message was received:")
        print(f"\t {osdg_prediction}")
        print()
        return None
    else:
        prediction_tensor = process_osdg_prediction(osdg_prediction)

    return prediction_tensor


def predict_multiple_samples_osdg(samples: list[str], ip: str = osdg_ip.ip1) -> list[torch.Tensor]:
    """
    Predict on multiple samples

    Args:
        samples: texts to be predicted on
        ip: ip address of the server

    Returns:
        predictions

    """
    predictions = []
    for sample in samples:
        prediction = predict_sample_osdg(sample, ip=ip)
        predictions.append(prediction)
    return predictions


def request_osdg_prediction(text: str, ip: str = osdg_ip.ip1) -> str:
    """
    Requests the prediction from the OSDG server.
    Args:
        text: text to predict on
        ip: ip address of the server

    Returns:
        unprocessed prediction from the server as a string

    """
    data = {"query": text}
    osdg_prediction = requests.post(ip, data=data)
    osdg_prediction = osdg_prediction.text
    return osdg_prediction


def process_osdg_prediction(osdg_prediction: str) -> torch.Tensor:
    """
    Processes the prediction from the OSDG server.
    Args:
        osdg_prediction: unprocessed prediction from the server as a string

    Returns:
        prediction from the server as a tensor of shape (17)

    """
    osdg_prediction = ast.literal_eval(osdg_prediction)
    prediction_list = [0] * 17
    for pred in osdg_prediction:
        sdg_pred = int(pred[0][4:]) - 1
        prediction_list[sdg_pred] = 1
    prediction_tensor = torch.tensor(prediction_list)
    return prediction_tensor


def predict_multiple_samples_aurora(samples: list[str]) -> torch.Tensor:
    """
    Predict on mutiple samples

    Args:
        samples: texts to be predicted on

    Returns:
        predictions of shape (n, 17) with n being the number of samples

    """
    predictions = aurora_mbert.create_aurora_predictions(samples)

    return predictions



def get_average_predictions(text: str, transformers: Union[base.Transformer, list[base.Transformer]]) -> torch.Tensor:
    if isinstance(transformers, base.Transformer):
        transformers = [transformers]
    predictions = []
    for transformer in transformers:
        prediction = transformer.predict(text)
        predictions.append(prediction)
    average_predictions = combine_predictions(predictions)
    return average_predictions


def threshold_predictions(predictions: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    threshold_predictions = predictions > threshold
    return threshold_predictions


def predict_any(threshold_predictions: torch.Tensor) -> torch.Tensor:
    prediction = torch.any(threshold_predictions, dim=0, keepdim=True)
    return prediction


# def predict_sdg_clf(text: str, transformers: Union[base.Transformer, list[base.Transformer]] = None,
#                     threshold: float = 0.5) -> torch.Tensor:
#     if isinstance(transformers, base.Transformer):
#         transformers = [transformers]
#     predictions = []
#     for transformer in transformers:
#         prediction = transformer.predict(text)
#         predictions.append(prediction)
#     if len(transformers) != 1:
#         predictions = combine_predictions(predictions)
#     else:
#         predictions = predictions[0]
#     thresholded_predictions = threshold_predictions(predictions, threshold, threshold=threshold)
#     any_prediction = predict_any(thresholded_predictions)
#
#     return any_prediction


# def predict_no_threshold(text: str,
#                          transformers: Union[base.Transformer, list[base.Transformer]]) -> torch.Tensor:
#     if isinstance(transformers, base.Transformer):
#         transformers = [transformers]
#     predictions = []
#     for transformer in transformers:
#         prediction = transformer.predict(text)
#         predictions.append(prediction)
#     if len(transformers) != 1:
#         predictions = combine_predictions(predictions)
#     else:
#         predictions = predictions[0]
#
#     return predictions


# def get_optimal_threshold(predictions: list[torch.Tensor], labels: torch.Tensor) -> tuple[float]:
#     f1 = torchmetrics.F1Score(num_classes=17, multiclass=False)
#     # try 100 thresholds from 0.0 to 1.0
#     thresholds = torch.linspace(0.0, 1.0, 100)
#     best_f1 = 0.0
#     for threshold in thresholds:
#         f1.reset()
#         thresholded_predictions = [threshold_predictions(pred, threshold) for pred in predictions]
#         any_predictions = [predict_any(pred) for pred in thresholded_predictions]
#         any_predictions = torch.concat(any_predictions, dim=0)
#         f1.update(any_predictions, labels)
#         f1_score = f1.compute()
#         if f1_score > best_f1:
#             best_f1 = f1_score
#             best_threshold = threshold
#     best_threshold = best_threshold.item()
#     best_f1 = best_f1.item()
#
#     return best_threshold, best_f1


# def predict_dataset(dataset: datasets.Dataset, method: str, model_weight: str = None, model_type: str = None,
#                     ):
#     predictions = []
#     if method == "sdg_clf":
#         tokenizer = utils.get_tokenizer(model_type)
#         model = sdg_clf.model.load_model(model_weight + ".pt", model_type)
#         transformer = base.Transformer(model, tokenizer)
#         for text in tqdm.tqdm(dataset["text"]):
#             pred = transformer.predict(text)
#             predictions.append(pred)
#     elif method == "osdg":
#         for text in tqdm.tqdm(dataset["text"]):
#             pred = predict_osdg(text)
#             predictions.append(pred)
#     return predictions


def combine_predictions(predictions: list[torch.tensor]) -> torch.tensor:
    longest_prediction = max(predictions, key=lambda x: x.shape[0])
    prediction_counter = torch.zeros(longest_prediction.shape)
    total_prediction = torch.zeros(longest_prediction.shape)
    for prediction in predictions:
        prediction_counter[:prediction.shape[0]] += 1
        total_prediction[:prediction.shape[0]] += prediction
    average_predictions = total_prediction / prediction_counter
    return average_predictions


# def get_predictions(method: str = "sdg_clf", tweet: bool = False, split: str = "test", model_type: str = None,
#                     model_weight: str = None, n_samples: int = None, overwrite: bool = False):
#     """
#
#     Args:
#         method: {"sdg_clf", "osdg", "aurora"}
#
#     Returns:
#     """
#     if tweet:
#         name_ds = "twitter"
#     else:
#         name_ds = "scopus"
#     if method == "sdg_clf":
#         path_predictions = f"predictions/{model_type}/{name_ds}_{split}.pkl"
#     else:
#         path_predictions = f"predictions/{method}_{name_ds}_{split}.pkl"
#     # if predictions already exist, load them
#     if os.path.exists(path_predictions) and not overwrite:
#         predictions = load_pickle(path_predictions)
#     else:
#         # check if base dataset exists else create it
#         path_base = f"data/processed/{name_ds}/base"
#         if not os.path.exists(path_base):
#             create_base_dataset(tweet=tweet)
#
#         if method == "sdg_clf":
#             if tweet:
#                 tokenizer = get_tokenizer(model_type)
#             dataloader = get_dataloader(split, model_type, tweet=tweet, n_samples=n_samples)
#             model = load_model(model_weight, model_type)
#             model.eval()
#             predictions = []
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             for sample in tqdm(dataloader):
#                 with torch.no_grad():
#                     if tweet:
#                         pred = model(sample["input_ids"].to(device),
#                                      sample["attention_mask"].to(device)).logits.sigmoid()
#                         predictions.append(pred)
#                     else:
#                         iids = sample["input_ids"]
#                         model_in = prepare_long_text_input(iids, tokenizer=tokenizer)
#                         model_out = model(**model_in).logits.sigmoid()
#                         predictions.append(model_out)
#             if tweet:
#                 predictions = torch.concat(predictions, dim=0)
#         elif method == "osdg":
#             predictions, failed_predictions = create_osdg_predictions(tweet=tweet, split=split, n_samples=n_samples)
#             save_pickle(f"predictions/fails_osdg_{name_ds}_{split}.pkl", predictions)
#
#         elif method == "aurora":
#             predictions = create_aurora_predictions(tweet=tweet, split=split, n_samples=n_samples)
#         save_pickle(path_predictions, predictions)
#     return predictions


# def create_osdg_predictions(tweet: bool = False, split: str = "test", n_samples: int = 1400):
#     if tweet:
#         name_ds = "twitter"
#         name_text = "text"
#     else:
#         name_ds = "scopus"
#         name_text = "Abstract"
#     ds_dict = datasets.load_from_disk(f"data/processed/{name_ds}/base")
#     ds = ds_dict[split]
#     ip = ip1
#     if tweet:
#         ds = ds.filter(lambda example: len(example[name_text].split()) > 40)
#         ds = ds.select([i for i in range(3000)])
#
#     idx = 0
#     pbar = tqdm(total=n_samples)
#     predictions = []
#     failed_predictions = []
#     while len(predictions) <= n_samples:
#         sample = ds[idx]
#         text = sample[name_text]
#         data = {"query": text}
#         prediction = requests.post(ip, data=data)
#         prediction = prediction.text
#         prediction = ast.literal_eval(prediction)
#         prediction_list = [0] * 17
#         try:
#             for pred in prediction:
#                 sdg_pred = int(pred[0][4:]) - 1
#                 prediction_list[sdg_pred] = 1
#             prediction_tensor = torch.tensor(prediction_list)
#             predictions.append(prediction_tensor)
#             pbar.update(1)
#         except:
#             failed_predictions.append(idx)
#         idx += 1
#     predictions = torch.stack(predictions, dim=0)
#     return predictions, failed_predictions
