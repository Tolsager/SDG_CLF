import os
import pickle

import torch
from tqdm import tqdm
from sdg_clf.dataset_utils import get_dataloader, get_tokenizer
from sdg_clf.utils import prepare_long_text_input
from sdg_clf.model import load_model


def get_tweet_preds(model_type, split, weights=None,
                    batch_size: int = 20):
    path_predictions = "predictions"
    if not os.path.exists(f"{path_predictions}/{model_type}/tweet_{split}.pkl"):
        dataloader = get_dataloader(split, model_type, tweet=True,
                                    batch_size=batch_size)
        os.makedirs(f"{path_predictions}/{model_type}", exist_ok=True)
        model = load_model(weights, model_type)
        model.eval()
        preds = []
        for sample in tqdm(dataloader):
            with torch.no_grad():
                pred = model(sample["input_ids"].to("cuda:0"), sample["attention_mask"].to("cuda:0")).logits.sigmoid()
                preds.append(pred)
        preds = torch.concat(preds, dim=0)
        with open(f"{path_predictions}/{model_type}/tweet_{split}.pkl", "wb") as f:
            pickle.dump(preds, f)
    with open(f"{path_predictions}/{model_type}/tweet_{split}.pkl", "rb") as f:
        tweet_preds = pickle.load(f)
    return tweet_preds


def get_scopus_preds(model_type, split, weights=None):
    path_predictions = "predictions"
    if not os.path.exists(f"predictions/{model_type}/scopus_{split}.pkl"):
        dataloader = get_dataloader(split, model_type, tweet=False, batch_size=1)
        tokenizer = get_tokenizer(model_type)
        os.makedirs(f"{path_predictions}/{model_type}", exist_ok=True)
        max_length, step_size = 260, 260
        model = load_model(weights, model_type)
        model.eval()
        preds = []
        for sample in tqdm(dataloader):
            with torch.no_grad():
                iids = sample["input_ids"]
                model_in = prepare_long_text_input(iids, tokenizer=tokenizer, max_length=max_length, step_size=step_size)
                model_out = model(**model_in).logits.sigmoid()
                # mean is less intuitive and provides worse results
                # pred = torch.mean(model_out, dim=0)
                preds.append(model_out)
        # if we concat, we don't know which predictions belong to which text
        # preds = torch.concat(preds, dim=0)
        with open(f"{path_predictions}/{model_type}/scopus_{split}.pkl", "wb") as f:
            pickle.dump(preds, f)
    with open(f"{path_predictions}/{model_type}/scopus_{split}.pkl", "rb") as f:
        scopus_preds = pickle.load(f)

    return scopus_preds
