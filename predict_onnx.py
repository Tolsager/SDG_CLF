import argparse
import os

import numpy as np
import psutil
import torch
from scipy import special

from sdg_clf import modelling, utils, base, dataset_utils

os.environ["OMP_NUM_THREADS"] = f"{psutil.cpu_count()}"
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"


def main(text: str) -> list[int]:
    text = dataset_utils.process_text(text)
    onnx_model = modelling.create_model_for_provider("finetuned_models/roberta-large_1608124504.onnx")
    tokenizer = utils.get_tokenizer("roberta-large")
    transformer = base.Transformer(torch.tensor([1]), tokenizer)
    model_inputs = transformer.prepare_model_inputs(text)
    model_inputs = {k: v.numpy() for k, v in model_inputs.items()}
    logits = onnx_model.run(None, model_inputs)[0]
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


if __name__ == "__main__":
    # sdgs = main("By April 2022, the coronavirus causing COVID-19 had infected more than 500 million people and killed more than 6.2 million worldwide. However, the most recent estimates suggest that the global number of excess deaths directly and indirectly attributable to COVID-19 could be as high as three times this figure. The pandemic has severely disrupted essential health services, shortened life expectancy and exacerbated inequities in access to basic health services between countries and people, threatening to undo years of progress in some health areas. Furthermore, immunization coverage dropped for the first time in 10 years and deaths from tuberculosis and malaria increased. ")
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
