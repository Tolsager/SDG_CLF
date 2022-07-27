import os

import torch
import transformers
from transformers import AutoModelForSequenceClassification


def load_model(model_type: str = None, weights: str = None):
    path_pretrained_model = os.path.join("pretrained_models", model_type)
    if os.path.exists(path_pretrained_model):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(path_pretrained_model)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=17)
        model.save_pretrained(f"pretrained_models/{model_type}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if weights is not None:
        path_weights = os.path.join("finetuned_models", weights)
        model.load_state_dict(torch.load(path_weights))
    return model