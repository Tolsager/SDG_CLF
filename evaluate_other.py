import argparse
import os

from sdg_clf import utils, evaluation, dataset_utils


def main(method: str, dataset_name: str, split: str, save_predictions: bool = True, overwrite: bool = False):
    os.chdir(os.path.dirname(__file__))
    predictions = evaluation.get_predictions_other(method, dataset_name, split, save_predictions, overwrite)

    # load dataframe
    df = dataset_utils.get_processed_df(dataset_name, split)
    labels = dataset_utils.get_labels_tensor(df)

    metrics = utils.get_metrics()
    utils.update_metrics(metrics, {"label": labels, "prediction": predictions})
    metrics_values = utils.compute_metrics(metrics)
    print(metrics_values)


if __name__ == "__main__":
    # set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="osdg")
    parser.add_argument("--dataset_name", type=str, default="scopus")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.method, args.dataset_name, split=args.split, save_predictions=args.save_predictions,
         overwrite=args.overwrite)
