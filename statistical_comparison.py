import torch
import datasets
from scipy.stats import binom, beta

from make_predictions import get_scopus_preds, get_tweet_preds
from ensemble import test_ensemble


def mcnemar_test(model1, model2, tweet: bool = False):
    thresholds = {"roberta-base": 0.62 if tweet else 0.52,
                  "roberta-large": 0.81 if tweet else 0.18,
                  "microsoft/deberta-v3-large": 0.74 if tweet else 0.36,
                  "albert-large-v2": 0.56 if tweet else 0.2}

    if tweet:
        labels = datasets.load_from_disk("data/processed/tweets/base")["test"]["label"]
        model1_pred = get_tweet_preds(model1, "test")
        model2_pred = get_tweet_preds(model2, "test")
    else:
        labels = datasets.load_from_disk("data/processed/scopus/base")["test"]["label"]
        if isinstance(model1, dict):
            model1_pred = test_ensemble(model1["weights"], model1["types"], tweet=False, log=False, return_f1=False, return_ensemble_preds=True)
        else:
            model1_pred = get_scopus_preds(model1, "test")
        if isinstance(model2, dict):
            model2_pred = test_ensemble(model2["weights"], model2["types"], tweet=False, log=False, return_f1=False, return_ensemble_preds=True)
        else:    
            model2_pred = get_scopus_preds(model2, "test")
            
    labels = torch.tensor(labels)
    model1_pred = any_threshold(model1_pred, thresholds[model1]) if isinstance(model1, str) else model1_pred
    model2_pred = any_threshold(model2_pred, thresholds[model2]) if isinstance(model2, str) else model2_pred

    both_correct = 0
    both_wrong = 0
    model1_correct = 0
    model2_correct = 0
    n = len(labels)
    
    for pred1, pred2, label in zip(model1_pred, model2_pred, labels):
        pred1 = pred1.cuda()
        pred2 = pred2.cuda()
        label = label.cuda()
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
    # McNemar tests between
    # Roberta base and Roberta large
    m = 5
    alpha = 0.05
    bonferonni_correct_alpha = alpha / m
    print((rb_rl:=mcnemar_test("roberta-base", "roberta-large", tweet=False)))
    print(f"Models are dfifferent: {rb_rl[0] < bonferonni_correct_alpha}")
    # Roberta large and ensemble
    print((rl_e:=mcnemar_test("roberta-large", {"weights": ["best_albert.pt", "best_deberta.pt", "best_roberta-large.pt"], "types": ["albert-large-v2", "microsoft/deberta-v3-large", "roberta-large"]}, tweet=False)))
    print(f"Models are different: {rl_e[0] < bonferonni_correct_alpha}")
    # Albert and Deberta
    print((a_d:=mcnemar_test("albert-large-v2", "microsoft/deberta-v3-large", tweet=False)))
    print(f"Models are different: {a_d[0] < bonferonni_correct_alpha}")
    # Albert and Ensemble (worst vs best)
    print((a_e:=mcnemar_test("albert-large-v2", {"weights": ["best_albert.pt", "best_deberta.pt", "best_roberta-large.pt"], "types": ["albert-large-v2", "microsoft/deberta-v3-large", "roberta-large"]}, tweet=False)))
    print(f"Models are different: {a_e[0] < bonferonni_correct_alpha}")
    # Roberta base and ensemble
    print((rb_e:=mcnemar_test("roberta-base", {"weights": ["best_albert.pt", "best_deberta.pt", "best_roberta-large.pt"], "types": ["albert-large-v2", "microsoft/deberta-v3-large", "roberta-large"]}, tweet=False)))
    print(f"Models are different: {rb_e[0] < bonferonni_correct_alpha}")
