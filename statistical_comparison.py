import pickle

import torch
import datasets
from scipy.stats import beta, binom
from statsmodels.stats.contingency_tables import mcnemar


from sdg_clf.make_predictions import get_scopus_preds, get_tweet_preds


def mcnemar_test(model1, model2, tweet:bool=False):

    thresholds = {"roberta-base": 0.62 if tweet else 0.48,
                  "roberta-large": 0.81 if tweet else 0.19,
                  "microsoft/deberta-v3-large": 0.74 if tweet else 0.2,
                  "albert-large-v2": 0.56 if tweet else 0.2}

    both_correct = 0
    both_wrong = 0
    model1_correct = 0
    model2_correct = 0

    if tweet:
        labels = datasets.load_from_disk("data/processed/tweets/base")["test"]["label"]
        model1_pred = get_tweet_preds(model1, "test")
        model2_pred = get_tweet_preds(model2, "test")

    else:
        labels = datasets.load_from_disk("data/processed/scopus/base")["test"]["label"]
        model1_pred = get_scopus_preds(model1, "test")
        model2_pred = get_scopus_preds(model2, "test")

    labels = torch.tensor(labels)
    model1_pred = torch.stack([pred > thresholds[model1] for pred in model1_pred], dim=0).int().cpu()
    model2_pred = torch.stack([pred > thresholds[model2] for pred in model2_pred], dim=0).int().cpu()

    for pred1, pred2, label in zip(model1_pred, model2_pred, labels):
        if torch.all(pred1 == pred2):
            if torch.all(pred1 == label):
                both_correct += 1
            else:
                both_wrong += 1
        else:
            if torch.all(pred1 == label):
                model1_correct += 1
            elif torch.all(pred2 == label):
                model2_correct += 1
            else:
                both_wrong += 1

    contingency_table = [[both_correct, model1_correct],
                         [model2_correct, both_wrong]]

    return mcnemar(contingency_table, exact=True)

if __name__ == '__main__':
    print(mcnemar_test("roberta-base", "roberta-large", tweet=False))
