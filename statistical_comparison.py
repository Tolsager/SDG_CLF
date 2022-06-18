import torch
import datasets
from scipy.stats import binom, beta
from evaluate import evaluate


def mcnemar_test(model1, model2, tweet: bool = False, regular: bool = True):
    if tweet:
        labels = datasets.load_from_disk("data/processed/tweets/base")["test"]["label"]
    else:
        labels = datasets.load_from_disk("data/processed/scopus/base")["test"]["label"]

    labels = torch.tensor(labels)

    model1_pred = evaluate(method="sdg_clf", tweet=tweet, split="test", model_types=model1["types"],
                           model_weights=model1["weights"], return_preds=True)
    model2_pred = evaluate(method="sdg_clf", tweet=tweet, split="test", model_types=model2["types"],
                           model_weights=model2["weights"], return_preds=True)

    both_correct = 0
    both_wrong = 0
    model1_correct = 0
    model2_correct = 0
    n = len(labels)

    if not regular:
        model1_pred = model1_pred.flatten()
        model2_pred = model2_pred.flatten()
        labels = labels.flatten()

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


if __name__ == '__main__':
    # McNemar tests between
    regular = True
    tweet = False
    m = 6
    alpha = 0.05
    bonferonni_correct_alpha = alpha / m
    # Roberta base and Roberta large
    print((rb_rl := mcnemar_test({"types": "roberta-base", "weights": "best_roberta-base.pt"},
                                 {"types": "roberta-large", "weights": "best_roberta-large.pt"}, tweet=tweet,
                                 regular=regular)))
    print(f"Models are different: {rb_rl[0] < bonferonni_correct_alpha}")
    # Roberta large and ensemble
    print((rl_e := mcnemar_test({"types": "roberta-large", "weights": "best_roberta-large.pt"},
                                {"weights": ["best_albert.pt", "best_deberta.pt", "best_roberta-large.pt"],
                                 "types": ["albert-large-v2", "microsoft/deberta-v3-large", "roberta-large"]},
                                tweet=tweet, regular=regular)))
    print(f"Models are different: {rl_e[0] < bonferonni_correct_alpha}")
    # # # Albert and Deberta
    print((a_d := mcnemar_test({"types": "albert-large-v2", "weights": "best_albert.pt"},
                               {"types": "microsoft/deberta-v3-large", "weights": "best_deberta.pt"}, tweet=tweet,
                               regular=regular)))
    print(f"Models are different: {a_d[0] < bonferonni_correct_alpha}")
    # # # Albert and Ensemble (worst vs best)
    print((a_e := mcnemar_test({"types": "albert-large-v2", "weights": "best_albert.pt"},
                               {"types": ["albert-large-v2", "microsoft/deberta-v3-large", "roberta-large"],
                                "weights": ["best_albert.pt", "best_deberta.pt", "best_roberta-large.pt"]},
                               tweet=tweet, regular=regular)))
    print(f"Models are different: {a_e[0] < bonferonni_correct_alpha}")
    # # # Roberta base and ensemble
    print((rb_e := mcnemar_test({"types": "roberta-base", "weights": "best_roberta-base.pt"},
                                {"types": ["albert-large-v2", "microsoft/deberta-v3-large", "roberta-large"],
                                 "weights": ["best_albert.pt", "best_deberta.pt", "best_roberta-large.pt"]},
                                tweet=tweet, regular=regular)))
    print(f"Models are different: {rb_e[0] < bonferonni_correct_alpha}")
