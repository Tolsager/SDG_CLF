from sdg_clf import trainer, dataset_utils
import torch
import pickle
import transformers
import os
from tqdm import tqdm


def get_tweet_preds(model, dataloader):
    if not os.path.exists("tweet_predictions.pkl"):
        model.eval()
        preds = []
        for sample in tqdm(dataloader):
            with torch.no_grad():
                pred = model(sample["input_ids"].to("cuda:0")).logits.sigmoid()
                preds.append(pred)
        with open("tweet_predictions.pkl", "wb") as f:
            pickle.dump(preds, f)
    with open("tweet_predictions.pkl", "rb") as f:
        tweet_preds = pickle.load(f)
    return tweet_preds



if __name__ == '__main__':
    eval_set = "tweet"

    tokenizer = transformers.AutoTokenizer.from_pretrained("tokenizers/roberta-base")
    sdg_model = transformers.AutoModelForSequenceClassification.from_pretrained("pretrained_models/roberta-base",
                                                                                num_labels=17)
    sdg_model.cuda()
    sdg_model.load_state_dict(torch.load("playful-sunset-10_0603190924.pt"))

    metrics = trainer.get_metrics(0.26, True, num_classes=1)

    if eval_set == "scopus":
        ds_dict = dataset_utils.load_ds_dict("roberta-base", tweet=False, path_data="data")
        test = ds_dict["test"]
        labels = torch.tensor(test["label"])

        with open("scopus_predictions.pkl", "rb") as f:
            preds = pickle.load(f)
        preds = torch.stack(preds, dim=0)

    else:
        ds_dict = dataset_utils.load_ds_dict("roberta-base", tweet=True, path_data="data")
        val = ds_dict["validation"]
        val.set_format("pt", columns=["input_ids"])
        labels = torch.tensor(val["label"])
        dl = torch.utils.data.DataLoader(val)

        preds = get_tweet_preds(sdg_model, dl)
        preds = torch.stack(preds, dim=0).reshape(-1, 17)

    eval_trainer = trainer.SDGTrainer(metrics=metrics, model=sdg_model, tokenizer=tokenizer)
    eval_trainer.set_metrics_to_device()

    for i in range(17):
        eval_trainer.reset_metrics()
        eval_trainer.update_metrics(step_outputs={"label": labels[:, i].to(eval_trainer.device), \
                                                  "prediction": preds[:, i].to(eval_trainer.device)})
        print(f"SDG {i + 1}")
        print(eval_trainer.compute_metrics())