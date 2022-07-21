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
    predictions = predictions > threshold
    return predictions


def threshold_multiple_predictions(predictions: list[torch.Tensor], threshold: float = 0.5) -> list[torch.Tensor]:
    predictions = [threshold_predictions(pred, threshold) for pred in predictions]
    return predictions


def predict_strategy_any(threshold_predictions: torch.Tensor) -> torch.Tensor:
    prediction = torch.any(threshold_predictions, dim=0, keepdim=True)
    return prediction


def predict_multiple_strategy_any(predictions: list[torch.Tensor]) -> list[torch.Tensor]:
    predictions = [predict_strategy_any(pred) for pred in predictions]
    return predictions


def get_optimal_threshold(predictions: list[torch.Tensor], labels: torch.Tensor) -> tuple[float]:
    f1 = torchmetrics.F1Score(num_classes=17, multiclass=False)
    # try 100 thresholds from 0.0 to 1.0
    thresholds = torch.linspace(0.0, 1.0, 100)
    best_f1 = 0.0
    for threshold in thresholds:
        f1.reset()
        thresholded_predictions = threshold_multiple_predictions(predictions, threshold)
        any_predictions = predict_multiple_strategy_any(thresholded_predictions)
        any_predictions = torch.concat(any_predictions, dim=0)
        f1.update(any_predictions, labels)
        f1_score = f1.compute()
        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold
    best_threshold = best_threshold.item()
    best_f1 = best_f1.item()

    return best_threshold, best_f1


def combine_predictions(predictions: list[torch.tensor]) -> torch.tensor:
    longest_prediction = max(predictions, key=lambda x: x.shape[0])
    prediction_counter = torch.zeros(longest_prediction.shape)
    total_prediction = torch.zeros(longest_prediction.shape)
    for prediction in predictions:
        prediction_counter[:prediction.shape[0]] += 1
        total_prediction[:prediction.shape[0]] += prediction
    average_predictions = total_prediction / prediction_counter
    return average_predictions


def combine_multiple_predictions(predictions: list[list[torch.tensor]]) -> list[torch.tensor]:
    # iterate over all predictions
    combined_predictions = []
    for i in range(len(predictions[0])):
        current_predictions = [predictions[j][i] for j in range(len(predictions))]
        combined_predictions.append(combine_predictions(current_predictions))
    return combined_predictions
