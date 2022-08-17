from sdg_clf import modelling, utils, evaluation, dataset_utils
import tqdm


def main() -> None:
    # get the model_weights if not given
    model_weights = modelling.get_model_weights()

    powerset = utils.get_powerset(model_weights)
    df = dataset_utils.get_processed_df("scopus", "val")
    target = dataset_utils.get_labels_tensor(df)
    best_stats = {"threshold": None, "model_weights": None, "f1": 0}
    for model_weights_subset in tqdm.tqdm(powerset, desc="Ensembles tested"):
        preds = evaluation.get_raw_predictions_sdg_clf("scopus", "val", model_weights_subset)
        preds = preds[0]

        threshold, f1 = evaluation.get_optimal_threshold(preds, target)
        if f1 > best_stats["f1"]:
            best_stats["threshold"] = threshold
            best_stats["model_weights"] = model_weights_subset
            best_stats["f1"] = f1

    print(best_stats)


if __name__ == "__main__":
    main()
