import argparse

import pandas as pd

from sdg_clf import utils, dataset_utils, inference


def main(text: str = None, file: str = None, column: str = "text", save_path: str = "predictions.csv",
         model_weights: list[str] = None,
         threshold: float = 0.5):
    if text is not None:
        text = [dataset_utils.process_text(text)]
        prediction = inference.get_final_predictions_sdg_clf(text, model_weights, threshold)
        prediction = prediction.flatten()
        utils.print_prediction(prediction)
    elif file is not None:
        # check if csv file
        if file.endswith(".csv"):
            df = pd.read_csv(file)
            texts = df[column].tolist()
            texts = [dataset_utils.process_text(sample) for sample in texts]
            predictions = inference.get_final_predictions_sdg_clf(texts, model_weights, threshold)
            df[[f"sdg{i}" for i in range(1, 18)]] = predictions
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
    parser.add_argument("--model_weights", type=str, nargs="+", required=True, help="filenames of the model weights")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args.text, args.file, args.column, args.save_path, model_weights=args.model_weights, threshold=args.threshold)
