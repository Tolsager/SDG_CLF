import os
import re

import torch
import transformers
from transformers import AutoModelForSequenceClassification


def load_model(model_type: str = None, weights_name: str = None) -> AutoModelForSequenceClassification:
    """
    Load a pretrained or fine-tuned model.
    Args:
        model_type: Huggingface model type. Is inferred automatically if weights_name is provided
        weights_name: name of the model weights in the "finetuned_models" i.e. "roberta-large-model1.pt".
            weights are assumed to be named "[model_type]_model[k].pt" where k is the number of the model.

    Returns:
        model: pretrained or fine-tuned model

    """
    # determine model_type from file_name
    if weights_name is not None and model_type is None:
        model_type = os.path.basename(weights_name).split("_")[0]

    # load model architecture
    path_pretrained_model = os.path.join("pretrained_models", model_type)
    if os.path.exists(path_pretrained_model):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(path_pretrained_model)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=17)
        model.save_pretrained(f"pretrained_models/{model_type}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # load weights if given
    if weights_name is not None:
        path_weights = os.path.join("finetuned_models", weights_name)
        model.load_state_dict(torch.load(path_weights, map_location=device))
    return model


def get_next_model_number(model_type: str) -> int:
    """
    Get the next number in the sequence of model weights.
    Args:
        model_type: Huggingface model type.

    Returns:
        the next number in the sequence

    """
    files = os.listdir(f"finetuned_models")
    model_type_files = [f for f in files if f.startswith(f"{model_type}_model")]
    if len(model_type_files) == 0:
        return 0
    else:
        # could find the number by using the length of model_type_files but it could fail if a model is sent from
        # one user to another
        model_numbers = [int(re.search(r"_model(\d+)\.pt", f).group(1)) for f in model_type_files]
        return max(model_numbers) + 1


def get_model_types(model_weights: list[str]) -> list[str]:
    """
    Get the model types from the model weights.
    Args:
        model_weights: list of model weights.

    Returns:
        list of model types.

    """
    return [os.path.basename(w).split("_")[0] for w in model_weights]
