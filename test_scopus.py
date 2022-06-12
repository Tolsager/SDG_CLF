import os

import torch
import transformers

from sdg_clf.dataset_utils import get_dataset
from sdg_clf.trainer import SDGTrainer


def eval_scopus(model_type: str, path_model: str, split: str, strategy: str = "any"):
    ds_dict = get_dataset(model_type, tweet=False)
    ds = ds_dict[split]
    tokenizer = transformers.AutoTokenizer.from_pretrained(os.path.join("tokenizers", model_type))
    sdg_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        os.path.join("pretrained_models", model_type),
        num_labels=17)

    sdg_model.cuda()
    sdg_model.load_state_dict(torch.load(path_model))

    trainer = SDGTrainer(tokenizer=tokenizer, model=sdg_model)
    ds.set_format("pt", columns=["input_ids", "label"])
    dl = torch.utils.data.DataLoader(ds)

    if strategy == "any":
        metrics, (threshold, _) = trainer.test_scopus_any(dl, step_size=260, max_length=260)
    elif strategy == "mean":
        metrics, (threshold, _) = trainer.test_scopus_any(dl, step_size=260, max_length=260)

    for k, v in metrics.items():
        metrics[k] = v.item()

    with open("results_scopus.txt", "a") as f:
        f.write(f"""
model_type: {model_type}, path_model: {path_model}, split: {split}, strategy: {strategy}
best threshold: {threshold}
metrics: {metrics}
        """)


if __name__ == "__main__":
    eval_scopus("roberta-base", "pretrained_models/best_model_0603141006.pt", split="train")
    eval_scopus("roberta-base", "pretrained_models/best_model_0603141006.pt", split="train", strategy="mean")
