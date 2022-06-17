import ast
import os
from .aurora_mbert import create_aurora_predictions

import datasets
import requests
import torch
from tqdm import tqdm

from .dataset_utils import create_base_dataset, get_dataloader, get_tokenizer
from .model import load_model
from .osdg_ip import ip1
from .utils import load_pickle, save_pickle, prepare_long_text_input


def get_predictions(method: str = "sdg_clf", tweet: bool = False, split: str = "test", model_type: str = None,
                    model_weight: str = None, n_samples: int = None, overwrite: bool = False):
    """

    Args:
        method: {"sdg_clf", "osdg", "aurora"}

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
                tokenizer = get_tokenizer(model_type)
            dataloader = get_dataloader(split, model_type, tweet=tweet, n_samples=n_samples)
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
            save_pickle(f"predictions/fails_osdg_{name_ds}_{split}.pkl", predictions)

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
    while len(predictions) <= n_samples:
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
