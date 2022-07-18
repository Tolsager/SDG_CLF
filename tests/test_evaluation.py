import torch
from sdg_clf import evaluation


def test_get_optimal_threshold():
    predictions = [torch.ones((2, 17)), torch.ones((1, 17))]
    labels = torch.ones((2, 17)).type(torch.int)
    optimal_threshold = evaluation.get_optimal_threshold(predictions, labels)
    assert 0 <= optimal_threshold <= 1


def test_predict_sample_osdg():
    invalid_text = "="
    prediction = evaluation.predict_sample_osdg(invalid_text)
    assert prediction is None
    valid_text = """
    A safe water supply is the backbone of a healthy economy, yet is woefully under prioritized, globally. 

It is estimated that waterborne diseases have an economic burden of approximately USD 600 million a year in India. This is especially true for drought- and flood-prone areas, which affected a third of the nation in the past couple of years..

Less than 50 per cent of the population in India has access to safely managed drinking water. Chemical contamination of water, mainly through fluoride and arsenic, is present in 1.96 million dwellings. 

Excess fluoride in India may be affecting tens of millions of people across 19 states, while equally worryingly, excess arsenic may affect up to 15 million people in West Bengal, according to the World Health Organization.

Moreover, two-thirds of India’s 718 districts are affected by extreme water depletion, and the current lack of planning for water safety and security is a major concern. One of the challenges is the fast rate of groundwater depletion in India, which is known as the world’s highest user of this source due to the proliferation of drilling over the past few decades. Groundwater from over 30 million access points supplies 85 per cent of drinking water in rural areas and 48 per cent of water requirements in urban areas.
    """
    prediction = evaluation.predict_sample_osdg(valid_text)
    assert prediction.shape[0] == 17
