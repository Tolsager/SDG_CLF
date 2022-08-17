from sdg_clf import modelling
import os

os.chdir("..")


def test_get_model_weights():
    model_weights = modelling.get_model_weights()
    assert len(model_weights) == 2
    assert not model_weights[0].startswith("finetuned_models")
