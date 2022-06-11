import random
import transformers
import numpy as np
import torch
import os
from typing import Union


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


def reset_weights(m):
    # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def get_tokenizer(tokenizer_type: str, path_tokenizers: str):
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
