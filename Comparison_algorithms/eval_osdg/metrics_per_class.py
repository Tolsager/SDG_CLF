from Comparison_algorithms.eval_AURORA_mbert.prediction_analysis import get_metrics, compute_metrics
import torch
import pickle
import datasets
from eval_osdg import convert_predictions

def classifications_metrics(preds: torch.tensor, labels: torch.tensor, per_class: bool = False):
    if per_class:
        metrics = get_metrics(num_classes = 1)
        performance = {}
        for i in range(17):
            performance[f'sdg{i+1}'] = compute_metrics(metrics, preds[:,i], labels[:,i])
    else:
        metrics = get_metrics(num_classes=17)
        performance = compute_metrics(metrics, preds, labels)
    return performance

if __name__ == "__main__":
    #For scopus:
    with open('osdg_predictions_scopus_test.pkl', 'rb') as f:
        preds = pickle.load(f)
    preds = convert_predictions(preds)
    labels_true = datasets.load_from_disk("../../data/processed/scopus/base")['test']['label']
    # with open("osdg_predictions_")

    # For tweets:
    #failed_preds = [3, 61, 103, 175, 177, 232, 233, 255, 288, 312, 345, 355, 407, 472, 522, 564, 565, 659, 680, 764,
    #                      779, 792, 817, 829, 839, 880, 927, 941, 967, 998, 1014, 1027, 1061, 1074, 1103, 1193, 1201, 1219,
    #                      1246, 1253, 1254, 1266, 1308, 1397, 1402, 1463, 1468, 1536, 1556, 1625, 1628, 1681, 1700, 1741,
    #                      1744, 1761, 1868, 1915, 1967, 2078, 2094, 2113, 2218, 2323, 2387, 2445, 2517, 2586, 2607, 2615,
    #                      2637, 2667, 2692, 2712, 2718, 2742, 2751, 2776, 2798, 2828, 2858, 2885, 2908, 2926, 2958]
    #
    #with open('osdg_predictions_tweets_test', 'rb') as f:
    #    preds = pickle.load(f)
    #labels_true = datasets.load_from_disk("../data/processed/tweets/base")['test']['label'][:3000]
    #for fail in sorted(failed_preds, reverse=True):
    #    del labels_true[fail]


    preds = torch.tensor(preds)
    labels_true = torch.tensor(labels_true)

    results = classifications_metrics(preds, labels_true)
    print(results)
    results_per_class=classifications_metrics(preds, labels_true, per_class = True)

    for i in range(17):
        s = f"SDG{i + 1}"
        for met, val in results_per_class[f'sdg{i+1}'].items():
            if met == 'accuracy':
                continue
            else:
                s += f" & {round(val.item(), 4)}"
        s += r" \\"
        print(s)