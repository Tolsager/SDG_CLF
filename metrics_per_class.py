from sdg_clf import trainer, dataset_utils
import torch
import pickle
import transformers
import os
from tqdm import tqdm
import numpy as np
from sdg_clf.trainer import get_metrics


def get_tweet_preds(model, model_type, split, dataloader):
    if not os.path.exists(f"predictions/{model_type}/tweet_{split}.pkl"):
        os.makedirs(f"predictions/{model_type}", exist_ok=True)
        model.eval()
        preds = []
        for sample in tqdm(dataloader):
            with torch.no_grad():
                pred = model(sample["input_ids"].to("cuda:0"), sample["attention_mask"].to("cuda:0")).logits.sigmoid()
                preds.append(pred)
        with open(f"predictions/{model_type}/tweet_{split}.pkl", "wb") as f:
            pickle.dump(preds, f)
    with open(f"predictions/{model_type}/tweet_{split}.pkl", "rb") as f:
        tweet_preds = pickle.load(f)
    return tweet_preds

def get_scopus_preds(model, model_type, split, dataloader, trainer):
    if not os.path.exists(f"predictions/{model_type}/scopus_{split}.pkl"):
        os.makedirs(f"predictions/{model_type}", exist_ok=True)
        max_length, step_size = 260, 260
        model.eval()
        preds = []
        for sample in tqdm(dataloader):
            with torch.no_grad():
                iids = sample["input_ids"]
                model_in = trainer.prepare_long_text_input(iids, max_length=max_length, step_size=step_size)
                model_out = model(**model_in).logits.sigmoid()
                pred = trainer.long_text_step(model_out)
                preds.append(pred)
        with open(f"predictions/{model_type}/scopus_{split}.pkl", "wb") as f:
            pickle.dump(preds, f)
    with open(f"predictions/{model_type}/scopus_{split}.pkl", "rb") as f:
        scopus_preds = pickle.load(f)
    
    return scopus_preds

def get_threshold(preds, labels, trainer):
    best_f1 = (0, 0)
    for threshold in np.linspace(0, 1, 101):
        metrics = get_metrics(threshold=threshold, multilabel=True)
        trainer.metrics = metrics
        trainer.set_metrics_to_device()
        trainer.reset_metrics()
        trainer.update_metrics({"label": labels.to(trainer.device), "prediction": preds.to(trainer.device)})
        metrics = trainer.compute_metrics()
        if metrics["f1"] > best_f1[1]:
            best_f1 = (threshold, metrics["f1"])
            best_metrics = metrics
    return best_f1[0], best_metrics

def performance(preds, labels, threshold, trainer):
    metrics = get_metrics(threshold=threshold, multilabel=True)
    trainer.metrics = metrics
    trainer.set_metrics_to_device()
    trainer.reset_metrics()
    trainer.update_metrics({"label": labels.to(trainer.device), "prediction": preds.to(trainer.device)})
    metrics = trainer.compute_metrics()
    return metrics



if __name__ == '__main__':
    eval_set = "scopus"
    model_type = "roberta-large"

    print(f"{model_type} evaluated on {eval_set}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(f"tokenizers/{model_type}")
    sdg_model = transformers.AutoModelForSequenceClassification.from_pretrained(f"pretrained_models/{model_type}",
                                                                                num_labels=17)
    sdg_model.cuda()
    sdg_model.load_state_dict(torch.load(f"models/{model_type}/brisk-cosmos-39_0608113410.pt"))

    metrics = trainer.get_metrics(0.5)

    eval_trainer = trainer.SDGTrainer(metrics=metrics, model=sdg_model, tokenizer=tokenizer)
    eval_trainer.set_metrics_to_device()

    if eval_set == "scopus":
        ds_dict = dataset_utils.load_ds_dict(model_type, tweet=False, path_data="data")
        train = ds_dict["train"]
        test = ds_dict["test"]

        train.set_format("pt", columns=["input_ids"])
        test.set_format("pt", columns=["input_ids"])

        train_labels = torch.tensor(train["label"])
        test_labels = torch.tensor(test["label"])

        dl_train = torch.utils.data.DataLoader(train)
        dl_test = torch.utils.data.DataLoader(test)

        train_preds = get_scopus_preds(sdg_model, model_type, "train", dl_train, eval_trainer)
        train_preds = torch.stack(train_preds, dim=0)

        test_preds = get_scopus_preds(sdg_model, model_type, "test", dl_test, eval_trainer)
        test_preds = torch.stack(test_preds, dim=0)

        threshold = get_threshold(train_preds, train_labels, eval_trainer)[0]
        overall_test_metrics = performance(test_preds, test_labels, threshold, eval_trainer)

    else:
        ds_dict = dataset_utils.load_ds_dict(model_type, tweet=True, path_data="data")
        validation = ds_dict["validation"]
        test = ds_dict["test"]

        validation.set_format("pt", columns=["input_ids", "attention_mask"])
        test.set_format("pt", columns=["input_ids", "attention_mask"])

        validation_labels = torch.tensor(validation["label"])
        test_labels = torch.tensor(test["label"])

        dl_validation = torch.utils.data.DataLoader(validation)
        dl_test = torch.utils.data.DataLoader(test)

        validation_preds = get_tweet_preds(sdg_model, model_type, "validation", dl_validation)
        validation_preds = torch.stack(validation_preds, dim=0).reshape(-1, 17)

        test_preds = get_tweet_preds(sdg_model, model_type, "test", dl_test)
        test_preds = torch.stack(test_preds, dim=0).reshape(-1, 17)

        threshold = get_threshold(validation_preds, validation_labels, eval_trainer)[0]
        overall_test_metrics = performance(test_preds, test_labels, threshold, eval_trainer)
    
    eval_trainer.metrics = get_metrics(threshold = threshold, multilabel=True, num_classes=1)
    print(threshold)
    print(overall_test_metrics)

    for i in range(17):
        s = f"SDG{i+1}"
        eval_trainer.set_metrics_to_device()
        eval_trainer.reset_metrics()
        eval_trainer.update_metrics(step_outputs={"label": test_labels[:, i].to(eval_trainer.device), \
                                                  "prediction": test_preds[:, i].to(eval_trainer.device)})
        # print(f"SDG {i + 1}")
        for val in eval_trainer.compute_metrics().values():
            s += f" & {round(val.item(),4)}"
        s += r" \\"
        print(s)