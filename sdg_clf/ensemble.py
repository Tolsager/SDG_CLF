import transformers
import os

from .dataset_utils import get_dataset
import torch
from typing import Union
from .utils import prepare_long_text_input


def test_ensemble(paths_models: list[str], model_types: list[str], dataset: str = "scopus", strategy: str = "mean",
                  path_tokenizers: str = "tokenizers", path_pretrained_models: str = "../pretrained_models",
                  path_all_model_predictions: str = "../all_model_predictions"):
    predictions = []
    if dataset == "scopus":
        tweet = False
    elif dataset == "tweets":
        tweet = True
    if not os.path.exists(path_all_model_predictions):
        for i, path_model, model_type in enumerate(zip(paths_models, model_types)):
            ds_dict = get_dataset(model_type, tweet=tweet, path_data="../data", path_tokenizers="../tokenizers")
            if dataset == "scopus":
                ds_val = ds_dict["train"]
                ds_test = ds_dict["test"]
            elif dataset == "tweets":
                ds_val = ds_dict["validation"]
                ds_test = ds_dict["test"]

            if i == 0:
                ds_val.set_format("pt", columns=["input_ids", "label"])
                labels_val = ds_val["label"]
            else:
                ds_val.set_format("pt", columns=["input_ids"])

            dl_val = torch.utils.data.dataloader(ds_val)

            path_tokenizer = os.path.join(path_tokenizers, model_type)
            if not os.path.exists(path_tokenizer):
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_type)
                os.makedirs(path_tokenizer)
                tokenizer.save_pretrained(path_tokenizer)
            else:
                tokenizer = transformers.AutoTokenizer.from_pretrained(path_tokenizer)

            path_base_model = os.path.join(path_pretrained_models, model_type)
            if not os.path.exists(path_base_model):
                model = transformers.AutoModelForSequenceClassification.from_pretrained(
                    model_type, num_labels=17
                )
                model.save_pretrained(path_base_model)
            else:
                model = transformers.AutoModelForSequenceClassification.from_pretrained(
                    path_base_model, num_labels=17
                )

            model.cuda()
            model_outputs_all = []
            for sample in dl_val:
                with torch.no_grad():
                    input_ids = sample["input_ids"]
                    if dataset == "scopus":
                        model_inputs = prepare_long_text_input(input_ids, tokenizer)
                    elif dataset == "tweet":
                        model_inputs = {"input_ids", input_ids, "attention_mask", sample["attention_mask"]}
                    model_outputs = model(**model_inputs).logits.sigmoid()
                    model_outputs_all.append(model_outputs)
            predictions.append(model_outputs_all)
