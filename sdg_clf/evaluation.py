import ast
import numpy as np
import numpy.typing as npt
import time
import json
import os.path
from typing import Union

import requests
import torch
import torchmetrics

from sdg_clf import base, utils, dataset_utils, aurora_mbert, modelling

try:
    from sdg_clf.osdg_ip import stable_ip, new_ip
except ImportError:
    # make sure the evaluation module can run if the osdg_ip module is not available
    stable_ip = ""
    new_ip = ""


def predict_sample_osdg_stable_api(text: str) -> Union[torch.Tensor, None]:
    """
    Predict the SDG classification for a given text using the OSDG server.
    Args:
        text: the text to predict on

    Returns:
        the prediction from the server as a tensor of shape (17)

    """
    osdg_prediction = request_osdg_prediction(text)
    if "ERROR" in osdg_prediction:
        print("OSDG unable to predict")
        print("The following error message was received:")
        print(f"\t {osdg_prediction}")
        print()
        return None
    else:
        prediction_tensor = process_osdg_prediction(osdg_prediction)

    return prediction_tensor


def predict_multiple_samples_osdg_stable_api(samples: list[str]) -> list[torch.Tensor]:
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
        prediction = predict_sample_osdg_stable_api(sample)
        predictions.append(prediction)
    return predictions


def request_osdg_prediction_stable_api(text: str) -> str:
    """
    Requests the prediction from the OSDG server.
    Args:
        text: text to predict on

    Returns:
        unprocessed prediction from the server as a string

    """
    data = {"query": text}
    osdg_prediction = requests.post(stable_ip, data=data)
    osdg_prediction = osdg_prediction.text
    return osdg_prediction


def process_osdg_prediction_stable_api(osdg_prediction: str) -> torch.Tensor:
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


def request_osdg_predictions_new_api(texts: list[str]) -> list[str]:
    """
    Requests the predictions from the OSDG server and returns the task ids of all the predictions
    Args:
        texts: text to predict on

    Returns:
        task_ids

    """
    url = new_ip + "text-upload"
    task_ids = []
    for text in texts:
        data = {"text": text, "token": "dtAAymjvxzKXyTNqNY2z"}
        result = requests.post(url, data=data)
        task_id = json.loads(result.text)
        task_ids.append(task_id)
    return task_ids


def get_raw_osdg_predictions_new_api(task_ids: list[str]) -> list[list[list[str, int]]]:
    """
    retrieves the predictions from the OSDG server
    Args:
        task_ids: the ids of the texts to retrieve the predictions for

    Returns:
        the raw predictions

    """
    url = new_ip + "retrieve-results"
    predictions = []
    for task_id in task_ids:
        while True:
            data = {"task_id": task_id, "token": "dtAAymjvxzKXyTNqNY2z"}
            result = requests.post(url, data=data)
            res = json.loads(result.text)
            if res["status"] == "Error: Could not extract text":
                time.sleep(2)
            else:
                predictions.append(res["document_sdg_labels"])
                break
    return predictions


def process_raw_osdg_predictions_new_api(raw_predictions: list[list[list[str, int]]]) -> npt.NDArray:
    """
    Processes the raw predictions from the OSDG server into a binary numpy array
    of shape (n, 17) with n being the number of samples
    Args:
        raw_predictions: the predictions from the server

    Returns:
        binary np array

    """
    predictions = []
    for pred in raw_predictions:
        new_pred = np.zeros(17)
        for sdg in pred:
            sdg_id = int(sdg[0][4:]) - 1
            new_pred[sdg_id] = 1
        predictions.append(new_pred)
    # stack the predictions
    predictions = np.stack(predictions, axis=0)
    return predictions


def get_osdg_predictions_new_api(texts: list[str]) -> npt.NDArray:
    task_ids = request_osdg_predictions_new_api(texts)
    raw_predictions = get_raw_osdg_predictions_new_api(task_ids)
    predictions = process_raw_osdg_predictions_new_api(raw_predictions)
    return predictions


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


def threshold_predictions(predictions: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    predictions = predictions > threshold
    return predictions


def threshold_multiple_predictions(predictions: list[torch.Tensor], threshold: float = 0.5) -> list[torch.Tensor]:
    predictions = [threshold_predictions(pred, threshold) for pred in predictions]
    return predictions


def predict_strategy_any(threshold_predictions: torch.Tensor) -> torch.Tensor:
    prediction = torch.any(threshold_predictions, dim=0)
    return prediction


def predict_multiple_strategy_any(predictions: list[torch.Tensor]) -> torch.Tensor:
    predictions = [predict_strategy_any(pred) for pred in predictions]
    predictions = torch.stack(predictions, dim=0)
    return predictions


def get_raw_predictions_sdg_clf(dataset_name: str, split: str, model_weights: list[str],
                                save_predictions: bool = True, overwrite: bool = False) -> list[list[torch.Tensor]]:
    model_types = modelling.get_model_types(model_weights)
    # the number of models in the ensemble
    n_models = len(model_types)
    prediction_paths = utils.get_prediction_paths(dataset_name, split, model_weights)
    if not overwrite:
        predictions = utils.load_predictions(prediction_paths)
    else:
        predictions = [None] * n_models

    # load dataframe
    df = dataset_utils.get_processed_df(dataset_name, split)
    samples = df["processed_text"].tolist()

    # create predictions if any are missing
    for i in range(len(predictions)):
        if predictions[i] is None:
            # load the model
            transformer = base.get_transformer(model_types[i], model_weights[i])
            predictions[i] = transformer.predict_multiple_samples_no_threshold(samples)

            if save_predictions:
                os.makedirs(os.path.dirname(prediction_paths[i]), exist_ok=True)
                utils.save_pickle(prediction_paths[i], predictions[i])
    return predictions


def get_predictions_other(method: str, dataset_name: str, split: str, save_predictions: bool = True,
                          overwrite: bool = False) -> list[torch.Tensor]:
    prediction_paths = utils.get_prediction_paths(method=method, dataset_name=dataset_name, split=split)
    if not overwrite:
        predictions = utils.load_predictions(prediction_paths)
    else:
        predictions = None

    # load dataframe
    df = dataset_utils.get_processed_df(dataset_name, split)

    # create predictions if any are missing
    if predictions is None:
        # get the text column of the df as a list of strings
        samples = df["text"].tolist()
        if method == "osdg_stable":
            predictions = predict_multiple_samples_osdg_stable_api(samples)
        elif method == "osdg_new":
            predictions = get_osdg_predictions_new_api(samples)
        elif method == "aurora":
            predictions = predict_multiple_samples_aurora(samples)

        if save_predictions:
            utils.save_pickle(prediction_paths, predictions)
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
