import argparse
import pandas as pd
import os
import torch

from sdg_clf import evaluation, base, utils, modelling, dataset_utils


def main(text: str = None, file: str = None, column: str = "text", save_path: str = "predictions.csv",
         model_weights: list[str] = None,
         threshold: float = 0.5):
    model_types = modelling.get_model_types(model_weights)
    transformer_list = base.get_multiple_transformers(model_types, model_weights)

    if text is not None:
        predictions = [transformer.predict_sample_no_threshold(text) for transformer in transformer_list]
        average_prediction = evaluation.combine_predictions(predictions)
        threshold_prediction = evaluation.threshold_predictions(average_prediction, threshold)
        threshold_prediction = torch.squeeze(threshold_prediction)
        utils.print_prediction(threshold_prediction)
    elif file is not None:
        # check if csv file
        if file.endswith(".csv"):
            df = pd.read_csv(file)
            samples = df[column].tolist()
            processed_samples = [dataset_utils.process_text(sample) for sample in samples]
            predictions = [transformer.predict_multiple_samples_no_threshold(processed_samples) for transformer in
                           transformer_list]
            average_predictions = evaluation.combine_multiple_predictions(predictions)
            threshold_predictions = evaluation.threshold_multiple_predictions(average_predictions, threshold)
            threshold_predictions = evaluation.predict_multiple_strategy_any(threshold_predictions)
            numpy_predictions = torch.concat(threshold_predictions, dim=0).numpy().astype(int)
            df[[f"sdg{i}" for i in range(1, 18)]] = numpy_predictions
            df.to_csv(save_path, index=False)
            print(f"Predictions have been saved to {save_path}")
        else:
            print("Please provide a csv file")
    else:
        raise ValueError("Either text or file must be provided")


if __name__ == "__main__":
    # set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="text to predict")
    parser.add_argument("--file", type=str,
                        help="file to predict on. Must be a csv file")
    parser.add_argument("--column", type=str, default="text", help="column in the .csv file to predict on")
    parser.add_argument("--save_path", type=str, default="predictions.csv", help="path to save predictions to")
    parser.add_argument("--model_weights", type=str, nargs="+")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args.text, args.file, args.column, args.save_path, model_weights=args.model_weights, threshold=args.threshold)
