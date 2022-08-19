import argparse
import os

import numpy as np
import psutil
import torch
from scipy import special

from sdg_clf import base, dataset_utils



def main(text: str, model, tokenizer) -> list[int]:
    text = dataset_utils.process_text(text)
    transformer = base.Transformer(torch.tensor([1]), tokenizer)
    model_inputs = transformer.prepare_model_inputs(text)
    model_inputs = {k: v.numpy() for k, v in model_inputs.items()}
    logits = model.run(None, model_inputs)[0]
    # apply sigmoid
    outputs = special.expit(logits)
    # apply threshold
    outputs = outputs > 0.27
    outputs = np.any(outputs, axis=0)
    # get indices
    indices = outputs.nonzero()[0]
    sdgs = indices + 1
    sdgs = sdgs.tolist()
    return sdgs

