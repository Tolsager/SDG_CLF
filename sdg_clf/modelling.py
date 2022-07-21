import os

import torch
import transformers
from transformers import AutoModelForSequenceClassification


def load_model(model_type: str = None, weights: str = None):
    if weights is not None:
        path_weights = os.path.join("finetuned_models", weights)
        model = AutoModelForSequenceClassification.from_pretrained(path_weights)
    else:
        path_pretrained_model = os.path.join("pretrained_models", model_type)
        if os.path.exists(path_pretrained_model):
            model = transformers.AutoModelForSequenceClassification.from_pretrained(path_pretrained_model,
                                                                                    num_labels=17)
        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=17)
            model.save_pretrained("pretrained_models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model
