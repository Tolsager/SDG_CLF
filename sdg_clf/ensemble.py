import transformers
import os

from .dataset_utils import get_dataset
import torch
from typing import Union


def prepare_long_text_input(input_ids: Union[list[int], torch.Tensor], tokenizer: transformers.PreTrainedTokenizer,
                            max_length: int = 260,
                            step_size: int = 260):
    """
    Prepare longer text for classification task by breaking into chunks

    Args:
        input_ids: Tokenized full text
        max_length (int, optional): Max length of each tokenized text chunk

    Returns:
        Dictionary of chunked data ready for classification
    """
    device = "cuda"
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()[0]
    input_ids = [input_ids[x:x + max_length - 2] for x in range(0, len(input_ids), step_size)]
    attention_masks = []
    for i in range(len(input_ids)):
        input_ids[i] = [tokenizer.cls_token_id] + input_ids[i] + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids[i])
        while len(input_ids[i]) < max_length:
            input_ids[i] += [tokenizer.pad_token_id]
            attention_mask += [0]
        attention_masks.append(attention_mask)

    input_ids = torch.tensor(input_ids, device=device)
    attention_mask = torch.tensor(attention_masks, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


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
