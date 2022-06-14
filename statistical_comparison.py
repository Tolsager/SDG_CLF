import torch
import datasets
from scipy.stats import binom, beta

from make_predictions import get_scopus_preds, get_tweet_preds


def mcnemar_test(model1, model2, tweet: bool = False, alpha: float = 0.05):
    thresholds = {"roberta-base": 0.62 if tweet else 0.48,
                  "roberta-large": 0.81 if tweet else 0.19,
                  "microsoft/deberta-v3-large": 0.74 if tweet else 0.2,
                  "albert-large-v2": 0.56 if tweet else 0.2}

    if tweet:
        labels = datasets.load_from_disk("data/processed/tweets/base")["test"]["label"]
        model1_pred = get_tweet_preds(model1, "test")
        model2_pred = get_tweet_preds(model2, "test")
    else:
        labels = datasets.load_from_disk("data/processed/scopus/base")["test"]["label"]
        model1_pred = get_scopus_preds(model1, "test")
        model2_pred = get_scopus_preds(model2, "test")

    labels = torch.tensor(labels)
    model1_pred = any_threshold(model1_pred, thresholds[model1])
    model2_pred = any_threshold(model2_pred, thresholds[model2])

    both_correct = 0
    both_wrong = 0
    model1_correct = 0
    model2_correct = 0
    n = len(labels)

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

    # Confidence interval of the difference in accuracy:
    # E_theta = (model1_correct - model2_correct) / len(labels)
    # Q = (n ** 2 * (n + 1) * (E_theta + 1) * (1 - E_theta)) / (
    #         n * (model1_correct + model2_correct) - (model1_correct - model2_correct) ** 2)
    # f = (E_theta + 1) / 2 * (Q - 1)
    # g = (1 - E_theta) / 2 * (Q - 1)
    # lower_bound = 2 * beta.ppf(alpha / 2, a=f, b=g) - 1
    # upper_bound = 2 * beta.ppf(1 - alpha / 2, a=f, b=g) - 1
    p_value = 2 * binom.cdf(k=min([model1_correct, model2_correct]), n=model1_correct + model2_correct, p=1 / 2)
    theta_hat = (model1_correct - model2_correct) / n

    return p_value, theta_hat, contingency_table


def any_threshold(preds, threshold):
    preds = [pred > threshold for pred in preds]
    preds = [torch.any(pred, dim=0) for pred in preds]
    preds = torch.stack(preds, dim=0).int().cpu()
    return preds


if __name__ == '__main__':
    import pickle

    print(mcnemar_test(model1="roberta-base", model2="roberta-large", tweet=True))
    print(mcnemar_test(model1="roberta-base", model2="roberta-large", tweet=False))
