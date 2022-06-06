import pandas as pd
import datasets
import torch
import os
import argparse

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import transformers
from sklearn.model_selection import KFold
import wandb
import numpy as np

# our scripts
from sdg_clf.trainer import SDGTrainer
import sdg_clf.tweet_dataset
from sdg_clf.model import get_model
import torchmetrics
from sdg_clf import utils
from api_key import key
from sdg_clf.dataset_utils import get_dataset




def main(
    multi_label: bool = False,
    call_tqdm: bool = True,
    sample_data: bool = False,
    metrics: dict = None,
    seed: int = 0,
    model_type: str = "roberta-base",
    log: bool = True,
    save_model: bool = False,
    model: torch.nn.Module = False,
    save_metric: str = "accuracy",
    hypers: dict = {"learning_rate": 3e-5,
                    "batch_size": 16,
                    "epochs": 2,
                    "weight_decay": 1e-2},
):
    """main() completes multi_label learning loop for one ROBERTA model using one model.
    Performance metrics and hyperparameters are stored using weights and biases log and config respectively.

    Args:
        batch_size (int, optional): How many samples per batch to load in the Dataloader. Defaults to 16.
        epochs (int, optional): Epochs for trainer. Defaults to 2.
        multi_label (bool, optional): Set to true if problem is a multi-label problem; if false problem is a multi-class label. Defaults to False.
        call_tqdm (bool, optional): Whether the training process is verbosely displayed in the console. Defaults to True.
        sample_data (bool, optional): Set to True for debugging purposes on a smaller data subset. Defaults to False.
        metrics (dict, optional): Specification of metrics using torchmetrics. Defaults to None.
        seed (int, optional): Random seed specification for controlled experiments. Defaults to 0.
        model_type (str, optional): Specification of pre-trained huggingface model used. Defaults to "roberta-base".
        log (bool, optional): Enables/disables logging via. Weights and Biases. Defaults to True.
        save_model (bool, optional): Set to true if models should be saved during training. Defaults to False.
        save_metric (str, optional): Determines the metric to compare between models for updating. Defaults to "accuracy".
    """
    if not log:
        os.environ["WANDB_MODE"] = "offline"
    # Setup W and B project log
    os.environ["WANDB_API_KEY"] = key
    run = wandb.init(project="sdg_clf", entity="pydqn", config=hypers)

    os.chdir(os.path.dirname(__file__))
    utils.seed_everything(seed)
    # Setup correct directory and seed
    save_path = "data/processed"
    save_path += f"/{model_type}"

    # get the dataset dict with splits
    ds_dict = get_dataset(tokenizer_type=model_type, sample_data=sample_data)

    # convert the model input for every split to tensors
    for ds in ds_dict.values():
        ds.set_format("pt", columns=["input_ids", "attention_mask", "label"])
    if not model:

        # load model
        model_path = "pretrained_models/" + model_type

        if not os.path.exists(model_path):
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_type, num_labels=17
            )

            # I think hugggingface uses makedirs so the following line should be redundant but needs to be tested
            # os.makedirs(model_path)
            model.save_pretrained(model_path)
        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=17
            )

    # Set loss criterion for trainer
    if multi_label:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    dl_train = DataLoader(ds_dict["train"], batch_size=hypers["batch_size"])
    dl_cv = DataLoader(ds_dict["validation"], batch_size=hypers["batch_size"])
    trainer = SDGTrainer(
        model=model,
        criterion=criterion,
        call_tqdm=call_tqdm,
        gpu_index=0,
        metrics=metrics,
        log=log,
        save_filename=run.name,
        save_model=save_model,
        save_metric=save_metric,
        hypers=hypers,
    )
    trainer.train(dl_train, dl_cv)


if __name__ == "__main__":
    multilabel = True
    metrics = {
        "accuracy": {
            "goal": "maximize",
            "metric": torchmetrics.Accuracy(subset_accuracy=True, multiclass=not multilabel),
        },
        # "auroc": {
            # "goal": "maximize",
            # "metric": torchmetrics.AUROC(num_classes=17),
        # },
        "precision": {
            "goal": "maximize",
            "metric": torchmetrics.Precision(num_classes=17, multiclass=not multilabel),
        },
        "recall": {
            "goal": "maximize",
            "metric": torchmetrics.Recall(num_classes=17, multiclass=not multilabel),
        },
        "f1": {
            "goal": "maximize",
            "metric": torchmetrics.F1Score(num_classes=17, multiclass=not multilabel),
        },
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--multilabel", help="boolean controlling whether the problem is multiclass or multilabel", action="store_true", default=False)
    parser.add_argument("-b", "--batchsize", help="number of batches sent to model", type=int, default=8)
    parser.add_argument("-l", "--log", help="log results to weights and biases", action='store_true', default=False)
    parser.add_argument("-s", "--save", help="save models during training", action="store_true", default=False)
    parser.add_argument("-e", "--epochs", help="number of epochs to train model", type=int, default=2)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=3e-5)
    parser.add_argument("-wd", "--weight_decay", help="optimizer weight decay", type=float, default=1e-2)
    parser.add_argument("-n", "--n_layers", help="number of dense layers before classification head", type=int, default=0)
    args = parser.parse_args()
    # main(
    #     batch_size=16,
    #     epochs=3,
    #     multi_label=True,
    #     call_tqdm=True,
    #     nrows=100,
    #     folds=3,
    #     metrics=metrics,
    # )
    main(
        multi_label=args.multilabel,
        call_tqdm=True,
        metrics=metrics,
        model_type="roberta-base",
        log=args.log,
        sample_data=False,
        save_model=args.save,
        model=get_model(pretrained_path="pretrained_models/roberta-base"),
        hypers={"learning_rate": args.learning_rate,
                "batch_size": args.batchsize,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay},
    )
