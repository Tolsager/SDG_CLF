import os
import numpy.typing as npt
import pickle
import random
from typing import Union, Any

import numpy as np
import torch
import torchmetrics
import transformers


def seed_everything(seed_value: int):
    """Sets seed for random, numpy, torch, and os to run controlled experiments

    Args:
        seed_value (int): Integer specifying seed
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def get_tokenizer(tokenizer_type: str):
    path_tokenizers = "tokenizers"
    path_tokenizer = os.path.join(path_tokenizers, tokenizer_type)
    if not os.path.exists(path_tokenizer):
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_type)
        tokenizer.save_pretrained(path_tokenizer)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(path_tokenizer)
    return tokenizer


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


def print_metrics(metrics: dict[str, torch.Tensor]) -> None:
    print("Metrics")
    print("--------")
    for k, v in metrics.items():
        print(f"{k}: {v.item()}")


def load_pickle(path: str):
    with open(path, "rb") as f:
        contents = pickle.load(f)
    return contents


def save_pickle(path: str, obj: object):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb+") as f:
        pickle.dump(obj, f)


def get_next_number(dir_path: str) -> int:
    """
    Get the next number in the directory.
    Args:
        dir_path: path to the directory with files enumerated as "anything_number.pkl"

    Returns:
        the next number in the sequence

    """
    files = os.listdir(dir_path)
    if len(files) == 1:
        return 0
    else:
        file_names = [os.path.splitext(f)[0] for f in files if not f.startswith(".")]
        file_numbers = [int(name.split("_")[-1]) for name in file_names if "_" in name]
        return max(file_numbers) + 1


def move_to(obj: Union[torch.Tensor, dict, list], device: str):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def get_prediction_paths(dataset_name: str, split: str, model_weights: list[str] = None, method: str = None,
                         idx_start: int = None, idx_end: int = None
                         ) -> Union[list[str], str]:
    if method == "osdg_stable" or method == "osdg_new" or method == "aurora":
        prediction_paths = f"predictions/{dataset_name}/{split}/{method}.pkl"
        if idx_start is not None and idx_end is not None:
            prediction_paths = f"predictions/{dataset_name}/{split}/{method}_{idx_start}-{idx_end}.pkl"
    else:
        # remove potential file extension
        model_weights = [os.path.splitext(w)[0] for w in model_weights]
        prediction_paths = [f"predictions/{dataset_name}/{split}/{model_weights[i]}.pkl" for i in
                            range(len(model_weights))]
    return prediction_paths


def load_predictions(prediction_paths: Union[list[str], str]) -> Union[list[torch.Tensor], torch.Tensor, None]:
    if isinstance(prediction_paths, str):
        if os.path.exists(prediction_paths):
            return load_pickle(prediction_paths)
        else:
            return None

    # otherwise it's a list of paths and the method is sdg_clf
    predictions = []
    for i in range(len(prediction_paths)):
        if os.path.exists(prediction_paths[i]):
            predictions.append(load_pickle(prediction_paths[i]))
        else:
            predictions.append(None)
    return predictions


def print_prediction(prediction: npt.NDArray) -> None:
    print("SDGs found in text")
    print("--------------------")
    sdg_dict = {1: "No Poverty", 2: "Zero Hunger", 3: "Good Health and Well-Being", 4: "Quality Education",
                5: "Gender Equality", 6: "Clean Water and Sanitation", 7: "Affordable and Clean Energy",
                8: "Decent Work and Economic Growth", 9: "Industry, Innovation and Infrastructure",
                10: "Reduced Inequalities", 11: "Sustainable Cities and Communities",
                12: "Responsible Consumption and Production", 13: "Climate Action", 14: "Life Below Water",
                15: "Life On Land", 16: "Peace, Justice and Strong Institutions", 17: "Partnerships for the Goals"}
    if not np.any(prediction):
        print("No SDGs found")
    else:
        for i in range(17):
            if prediction[i] == 1:
                print(f"    SDG {i + 1}: {sdg_dict[i + 1]}")
