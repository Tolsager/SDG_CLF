import argparse

from sdg_clf import evaluation, dataset_utils


def main(dataset_name: str, split: str, model_weights: list[str] = None, model_types: list[str] = None,
         save_predictions: bool = False, overwrite: bool = False, threshold: float = 0.5):
    if dataset_name == "osdg":
        raise ValueError("The OSDG dataset only has a single label for each sample so no threshold is used")
    # load predictions that already exist if overwrite is False

    predictions = evaluation.get_predictions_sdg_clf(dataset_name, split, model_weights, model_types, save_predictions,
                                                     overwrite)
    df = dataset_utils.get_processed_df(dataset_name, split)
    labels = dataset_utils.get_labels_tensor(df)

    best_threshold, best_f1 = evaluation.get_optimal_threshold(predictions, labels)
    print(f"Best threshold: {best_threshold}")
    print(f"Best F1: {best_f1}")


if __name__ == "__main__":
    # set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="scopus")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_weights", type=str, nargs="+")
    parser.add_argument("--model_types", type=str, nargs="+")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args.dataset_name, split=args.split, model_weights=args.model_weights, model_types=args.model_types,
         save_predictions=args.save_predictions, overwrite=args.overwrite, threshold=args.threshold)
