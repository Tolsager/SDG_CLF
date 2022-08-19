import os
import onnxruntime
import pytorch_lightning as pl
import re

import torch
import transformers
from transformers import AutoModelForSequenceClassification


def load_model(model_type: str = None, weights_name: str = None, device: str = None) -> AutoModelForSequenceClassification:
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
        model_type = weights_name.split("_")[0]

    # load model architecture
    path_pretrained_model = os.path.join("pretrained_models", model_type)
    if os.path.exists(path_pretrained_model):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(path_pretrained_model)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=17)
        model.save_pretrained(f"pretrained_models/{model_type}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # load weights if given
    if weights_name is not None:
        path_weights = os.path.join("finetuned_models", weights_name)
        state_dict = torch.load(path_weights, map_location=device)["state_dict"]
        # remove "model." prefix from state_dict keys
        state_dict = {k[6:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
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
        model_numbers = [int(re.search(r"_model(\d+)", f).group(1)) for f in model_type_files]
        return max(model_numbers) + 1


def get_model_types(model_weights: list[str]) -> list[str]:
    """
    Get the model types from the model weights.
    Args:
        model_weights: list of model weights.

    Returns:
        list of model types.

    """
    return [w.split("_")[0] for w in model_weights]


def optimize_model_for_inference(model: pl.LightningModule, save_filename: str) -> None:
    """
    Optimize the model's inference speed by converting it to ONNX.
    Args:
        model: model to optimize
        save_filename: name of the file to save the model to.
    """
    # optimize model
    input_sample = (torch.randn((1, 260)), torch.randn((1, 260)))
    model.to_onnx(save_filename, input_sample, export_params=True)


def get_model_weights() -> list[str]:
    """
    Get the model weights from the "finetuned_models" folder.
    Returns:
        list of model weights.
    """
    model_weights = []
    for dirpath, dirnames, filenames in os.walk("finetuned_models"):
        # all ".ckpt" files
        ckpt_files = [os.path.join(dirpath, f) for f in filenames if f.endswith(".ckpt")]
        # remove "finetuned_models/" prefix from ckpt_files
        ckpt_files = [os.path.relpath(f, "finetuned_models") for f in ckpt_files]
        model_weights.extend(ckpt_files)

    model_weights.sort()
    return model_weights


def create_model_for_provider(model_path: str, provider: str = "CPUExecutionProvider"):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session