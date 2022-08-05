import torch
from sdg_clf import evaluation
import os

os.chdir("..")


def test_get_optimal_threshold():
    predictions = [torch.ones((2, 17)), torch.ones((1, 17))]
    labels = torch.ones((2, 17)).type(torch.int)
    optimal_threshold = evaluation.get_optimal_threshold(predictions, labels)
    assert 0 <= optimal_threshold <= 1


class TestOSDG:
    invalid_text = "="

    valid_text = """
        A safe water supply is the backbone of a healthy economy, yet is woefully under prioritized, globally. 

    It is estimated that waterborne diseases have an economic burden of approximately USD 600 million a year in India. This is especially true for drought- and flood-prone areas, which affected a third of the nation in the past couple of years..

    Less than 50 per cent of the population in India has access to safely managed drinking water. Chemical contamination of water, mainly through fluoride and arsenic, is present in 1.96 million dwellings. 

    Excess fluoride in India may be affecting tens of millions of people across 19 states, while equally worryingly, excess arsenic may affect up to 15 million people in West Bengal, according to the World Health Organization.

    Moreover, two-thirds of India’s 718 districts are affected by extreme water depletion, and the current lack of planning for water safety and security is a major concern. One of the challenges is the fast rate of groundwater depletion in India, which is known as the world’s highest user of this source due to the proliferation of drilling over the past few decades. Groundwater from over 30 million access points supplies 85 per cent of drinking water in rural areas and 48 per cent of water requirements in urban areas.
        """

    def test_predict_sample_osdg(self):
        prediction = evaluation.predict_sample_osdg_stable_api(self.invalid_text)
        assert prediction is None
        prediction = evaluation.predict_sample_osdg_stable_api(self.valid_text)
        assert prediction.shape[0] == 17

    def test_predict_multiple_samples_osdg(self):
        ds_valid_invalid = [self.valid_text, self.invalid_text]
        ds_valid_valid = [self.valid_text, self.valid_text]
        ds_invalid_invalid = [self.invalid_text, self.invalid_text]
        predictions_valid_invalid = evaluation.predict_multiple_samples_osdg(ds_valid_invalid)
        predictions_valid_valid = evaluation.predict_multiple_samples_osdg(ds_valid_valid)
        predictions_invalid_invalid = evaluation.predict_multiple_samples_osdg(ds_invalid_invalid)
        assert predictions_valid_invalid[0].shape[0] == 17
        assert predictions_valid_invalid[1] is None
        assert predictions_valid_valid[0].shape[0] == 17
        assert predictions_valid_valid[1].shape[0] == 17
        assert predictions_invalid_invalid[0] is None
        assert predictions_invalid_invalid[1] is None


def test_predict_multiple_samples_aurora():
    # change the working directory
    samples = ["This is a test", "This is another test"]
    predictions = evaluation.predict_multiple_samples_aurora(samples)
    assert predictions.shape == (2, 17)


def test_get_osdg_predictions_new_api():
    text = [
        "For those who work, having a job does not guarantee a decent living. In fact, 8 per cent of employed workers and their families worldwide lived in extreme poverty in 2018. One out of five children live in extreme poverty. Ensuring social protection for all children and other vulnerable groups is critical to reduce poverty."]
    prediction = evaluation.get_osdg_predictions_new_api(text)
    print(prediction)
    assert prediction.shape == (1, 17)
    texts = text + ["At the same time, a profound change of the global food and agriculture system is needed if we are to nourish the more than 690 million people who are hungry today – and the additional 2 billion people the world will have by 2050. Increasing agricultural productivity and sustainable food production are crucial to help alleviate the perils of hunger"]
    predictions = evaluation.get_osdg_predictions_new_api(texts)
    print(predictions)
    assert predictions.shape == (2, 17)
