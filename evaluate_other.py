import argparse

import torch

from sdg_clf import evaluation, dataset_utils, base


def main(method: str, dataset_name: str, split: str, save_predictions: bool = True, overwrite: bool = False,
         idx_start=None, idx_end=None):
    """
    Evaluate a method on a dataset.

    Args:
        method: {"osdg_stable", "osdg_new", "aurora", "base"}. the method to use
        dataset_name: the name of the dataset
        split: the split to use
        save_predictions: whether to save the predictions
        overwrite: whether to overwrite the predictions if they already exist
    Returns:
        None

    """
    # load dataframe
    df = dataset_utils.get_processed_df(dataset_name, split)
    target = dataset_utils.get_labels_tensor(df)

    if method == "base":
        preds = torch.zeros((len(df), 17))
        # SDG4 is the most common SDG in the Twitter dataset, so we set it to 1
        preds[:, 3] = 1

    else:
        preds = evaluation.get_predictions_other(method, dataset_name, split, save_predictions, overwrite, idx_start,
                                                 idx_end)

    # select subset of target if idx_start and idx_end are given
    if idx_start is not None and idx_end is not None:
        target = target[idx_start:idx_end]

    # compute metrics
    metrics = base.Metrics()
    metrics.update(preds, target)
    metrics.compute()
    metrics.print()


if __name__ == "__main__":
    # set up argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="osdg_stable")
    parser.add_argument("--dataset_name", type=str, default="scopus")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--idx_start", type=int, default=None)
    parser.add_argument("--idx_end", type=int, default=None)
    args = parser.parse_args()
    main(args.method, args.dataset_name, split=args.split, overwrite=args.overwrite, idx_start=args.idx_start,
         idx_end=args.idx_end)
