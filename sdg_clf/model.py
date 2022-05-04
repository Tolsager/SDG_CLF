# Hugging Face transformers packages
# Why do we get the RobertaForSequenceClassification model, if we don't use it?
from transformers import (
    AutoModelForSequenceClassification,
    RobertaForSequenceClassification,
)

# Why are we installing these?!
import torch.nn as nn
import transformers


def get_model(
    num_labels=17,
    pretrained_path: str = None,
):
    """Function calling the from_pretrained method on Huggingface hosted pretrained models

    Args:
        num_labels (int, optional): The number of labels to be classified. Defaults to 17.
        pretrained_path (str, optional): String that mentions the model id to be retrieved. Defaults to None.

    Returns:
        _type_: _description_
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_path, num_labels=num_labels
    )
    return model


"""
Make new model inheriting from AutoModelForSequenceClassification
model.forward_scopus(input_ids):
    split input ids into chunks
    prepend chunk with [CLS]
    append chunk with [EOS]
    pad chunk

    for every chunk:
        preds.append(model(chunk))
    
    combine prediction and output final predictions
"""
