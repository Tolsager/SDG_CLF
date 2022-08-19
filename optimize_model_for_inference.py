from sdg_clf import modelling
import torch
import os
from onnxruntime import quantization
import argparse


def main(model_weights: str):
    model = modelling.load_model(weights_name=model_weights, device="cpu")
    # convert model to onnx
    sample_input_ids = torch.randint(0, 30_000, (1, 260))
    sample_attention_mask = torch.ones((1, 260))
    input_dict = {"input_ids": sample_input_ids, "attention_mask": sample_attention_mask}
    sample_input = (input_dict,)
    save_name = "finetuned_models/" + os.path.splitext(model_weights)[0] + ".onnx"
    torch.onnx.export(model, sample_input, save_name, export_params=True, opset_version=11, do_constant_folding=True,)
    quantization.quantize_dynamic(save_name, save_name, weight_type=quantization.QuantType.QInt8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights", type=str, help="Name of the model weights to optimize.")
    args = parser.parse_args()
    main(args.model_weights)
