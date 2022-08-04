from sdg_clf import inference
import os
import numpy as np

os.chdir("..")


def test_get_final_predictions_sdg_clf():
    texts0 = ["This is a test text"]
    model_weights0 = ["albert-large-v2_model0.pt"]
    predictions0 = inference.get_final_predictions_sdg_clf(texts0, model_weights0)
    assert predictions0.shape == (1, 17)
    assert np.all(predictions0 == np.zeros((1, 17)))

    model_weights1 = model_weights0 + ["roberta-large_model0.pt"]
    predictions1 = inference.get_final_predictions_sdg_clf(texts0, model_weights1)
    assert predictions1.shape == (1, 17)
    assert np.all(predictions1 == np.zeros((1, 17)))

    texts1 = ["This is a test text", "This is a test text"]
    predictions1 = inference.get_final_predictions_sdg_clf(texts1, model_weights0)
    assert predictions1.shape == (2, 17)
    assert np.all(predictions1 == np.zeros((2, 17)))

    predictions2 = inference.get_final_predictions_sdg_clf(texts1, model_weights1)
    assert predictions2.shape == (2, 17)
    assert np.all(predictions2 == np.zeros((2, 17)))

