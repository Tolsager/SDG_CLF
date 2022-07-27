import argparse
import os
import torch

from sdg_clf import evaluation, base, utils, modelling


def main(text: str, model_weights: list[str] = None, threshold: float = 0.5):
    os.chdir(os.path.dirname(__file__))
    model_types = modelling.get_model_types(model_weights)
    transformer_list = base.get_multiple_transformers(model_types, model_weights)
    predictions = [transformer.predict_sample_no_threshold(text) for transformer in transformer_list]
    average_prediction = evaluation.combine_predictions(predictions)
    threshold_prediction = evaluation.threshold_predictions(average_prediction, threshold)
    threshold_prediction = torch.squeeze(threshold_prediction)
    utils.print_prediction(threshold_prediction)


if __name__ == "__main__":
    # set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model_weights", type=str, nargs="+")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args.text, model_weights=args.model_weights, threshold=args.threshold)
