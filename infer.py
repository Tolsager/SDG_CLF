import argparse
import os

from sdg_clf import evaluation, base


def main(text: str, model_weights: list[str] = None, model_types: list[str] = None, threshold: float = 0.5):
    os.chdir(os.path.dirname(__file__))
    transformer_list = base.get_multiple_transformers(model_types, model_weights)
    predictions = [transformer.predict(text) for transformer in transformer_list]
    average_predictions = evaluation.combine_predictions(predictions)
    threshold_predictions = evaluation.threshold_predictions(average_predictions, threshold)
    print(threshold_predictions)


if __name__ == "__main__":
    # set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights", type=str, nargs="+")
    parser.add_argument("--model_types", type=str, nargs="+")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(model_weights=args.model_weights, model_types=args.model_types,
         threshold=args.threshold)
