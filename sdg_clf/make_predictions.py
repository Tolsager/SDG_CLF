import os
import pickle

import torch
from tqdm import tqdm
from .dataset_utils import get_dataloader, get_tokenizer
from .utils import prepare_long_text_input


def get_tweet_preds(model, model_type, split, path_predictions, path_data: str, path_tokenizers: str,
                    batch_size: int = 20):
    dataloader = get_dataloader(split, model_type, tweet=True, path_data=path_data, path_tokenizers=path_tokenizers,
                                batch_size=batch_size)
    if not os.path.exists(f"{path_predictions}/{model_type}/tweet_{split}.pkl"):
        os.makedirs(f"{path_predictions}/{model_type}", exist_ok=True)
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


def get_scopus_preds(model, model_type, split, path_predictions, path_data: str, path_tokenizers: str):
    dataloader = get_dataloader(split, model_type, tweet=False, path_data=path_data, path_tokenizers=path_tokenizers, batch_size=1)
    tokenizer = get_tokenizer(model_type, path_tokenizers=path_tokenizers)
    if not os.path.exists(f"{path_predictions}/{model_type}/scopus_{split}.pkl"):
        os.makedirs(f"{path_predictions}/{model_type}", exist_ok=True)
        max_length, step_size = 260, 260
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
