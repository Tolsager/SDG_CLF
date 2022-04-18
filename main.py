import pandas as pd
import torch
import os

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import transformers
from sklearn.model_selection import KFold
import wandb
import numpy as np

# our scripts
from trainer import SDGTrainer
import tweet_dataset
from model import get_model
import torchmetrics
import utils


def main(
    batch_size: int = 16,
    csv_path: str = "data/raw/allSDGtweets.csv",
    epochs: int = 2,
    multi_class: bool = False,
    call_tqdm: bool = True,
    nrows: int = None,
    folds: int = False,
    metrics: dict = None,
    seed: int = 0,
    model_type: str = "roberta-base",
):
    os.environ["WANDB_API_KEY"] = "bf4a3866ef6d0f0c18db1a02e1a49b8c6a71c4d8"
    os.chdir(os.path.dirname(__file__))
    utils.seed_everything(seed)
    ds = tweet_dataset.load_dataset(file=csv_path, nrows=nrows, multi_class=multi_class)
    ds.set_format("pt", columns=["input_ids", "attention_mask", "label"])
    model_path = "pretrained_models/" + model_type.replace("-", "_")
    if not os.path.exists(model_path):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_type, num_labels=17
        )

        os.makedirs(model_path)
        model.save_pretrained(model_path)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=17
        )
    # wandb.init(project="sdg_clf", entity="tolleren")
    wandb.init(project="sdg_clf", entity="rsm-git")
    wandb.config = {"epochs": epochs, "batch_size": batch_size, "learning_rate": 3e-5}

    if multi_class:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if not folds:
        ds["train"] = ds["train"].train_test_split(test_size=0.1)
        dl_train = DataLoader(ds["train"]["train"], batch_size=batch_size)
        dl_cv = DataLoader(ds["train"]["test"], batch_size=batch_size)
        # dl_test = DataLoader(ds_test, batch_size=batch_size)
        trainer = SDGTrainer(
            multi_class=multi_class,
            model=model,
            epochs=epochs,
            criterion=criterion,
            call_tqdm=call_tqdm,
            gpu_index=0,
            metrics=metrics,
        )
        best_val_acc = trainer.train(dl_train, dl_cv)
    else:
        val_accs = []
        kf = KFold(n_splits=folds)
        for i, (train_index, cv_index) in enumerate(
            kf.split([*range(ds["train"].num_rows)])
        ):
            print(f"Validating on fold {i}\n")
            ds_train = ds["train"].select(train_index)
            ds_cv = ds["train"].select(cv_index)
            dl_train = DataLoader(ds_train, batch_size=batch_size)
            dl_cv = DataLoader(ds_cv, batch_size=batch_size)

            model.apply(utils.reset_weights)
            trainer = SDGTrainer(
                model=model,
                epochs=epochs,
                criterion=criterion,
                call_tqdm=call_tqdm,
                gpu_index=0,
                metrics=metrics,
            )
            best_val_acc = trainer.train(dl_train, dl_cv)["accuracy"]
            val_accs.append(best_val_acc)
            wandb.log({"fold": i, "best_val_acc": best_val_acc})
        wandb.log({"avg_val_acc": np.mean(val_accs)})


if __name__ == "__main__":
    metrics = {
        "accuracy": {
            "goal": "maximize",
            "metric": torchmetrics.Accuracy(subset_accuracy=True),
        }
    }
    # main(
    #     batch_size=16,
    #     epochs=3,
    #     multi_class=True,
    #     call_tqdm=True,
    #     nrows=100,
    #     folds=3,
    #     metrics=metrics,
    # )

    main(
        batch_size=16,
        epochs=3,
        multi_class=True,
        call_tqdm=False,
        folds=10,
        metrics=metrics,
        model_type="roberta-base",
    )
