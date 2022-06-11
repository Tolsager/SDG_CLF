from sdg_clf import trainer, dataset_utils
import torch
import transformers
import numpy as np
from sdg_clf.trainer import get_metrics

from sdg_clf.make_predictions import get_tweet_preds, get_scopus_preds


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
    model_type = "roberta-base"

    print(f"{model_type} evaluated on {eval_set}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(f"tokenizers/{model_type}")
    sdg_model = transformers.AutoModelForSequenceClassification.from_pretrained(f"pretrained_models/{model_type}",
                                                                                num_labels=17)
    sdg_model.cuda()
    sdg_model.load_state_dict(torch.load(f"models/{model_type}/playful-sunset-10_0603190924.pt"))

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