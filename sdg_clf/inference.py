import numpy.typing as npt
from sdg_clf import modelling, base, evaluation


def get_final_predictions_sdg_clf(texts: list[str], model_weights: list[str], threshold: float = 0.5) -> npt.NDArray:
    model_types = modelling.get_model_types(model_weights)
    # the number of models in the ensemble
    n_models = len(model_types)
    predictions = []

    # create predictions if any are missing
    for i in range(n_models):
        # load the model
        transformer = base.get_transformer(model_types[i], model_weights[i])
        predictions.append(transformer.predict_multiple_samples_no_threshold(texts))

    # combine predictions
    predictions = evaluation.combine_multiple_predictions(predictions)
    # threshold predictions
    predictions = evaluation.threshold_multiple_predictions(predictions, threshold)
    # predict with strategy any
    predictions = evaluation.predict_multiple_strategy_any(predictions)
    # convert to numpy array
    predictions = predictions.numpy().astype(int)

    return predictions
