import pandas as pd
import datasets
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
from sdg_clf.trainer import SDGTrainer
import sdg_clf.tweet_dataset
from sdg_clf.model import get_model
import torchmetrics
from sdg_clf import utils
from api_key import key


def main(
    batch_size: int = 16,
    csv_path: str = "data/raw/allSDGtweets.csv",
    epochs: int = 2,
    multi_label: bool = False,
    call_tqdm: bool = True,
    nrows: int = None,
    folds: int = False,
    metrics: dict = None,
    seed: int = 0,
    model_type: str = "roberta-base",
    log: bool = True,
):
    """main() completes multi_label learning loop for one ROBERTA model using either one model order cross-validation to find
    the best hyper parameters. Performance metrics and hyperparameters are stored using weights and biases log and config respectively.

    Args:
        batch_size (int, optional): How many samples per batch to load in the Dataloader. Defaults to 16.
        csv_path (str, optional): Path to csv file. Defaults to "data/raw/allSDGtweets.csv".
        epochs (int, optional): Epochs for trainer. Defaults to 2.
        multi_label (bool, optional): Set to true if problem is a multi-label problem; if false problem is a multi-class label. Defaults to False.
        call_tqdm (bool, optional): Whether the training process is verbosely displayed in the console. Defaults to True.
        nrows (int, optional): How many rows of the dataset will be loaded; The entire dataset will be loaded if nrows is None. Defaults to None.
        folds (int, optional): Number of folds in the k-fold cross validation; if False cross-validation is not used. Defaults to False.
        metrics (dict, optional): Specification of metrics using torchmetrics. Defaults to None.
        seed (int, optional): Random seed specification for controlled experiments. Defaults to 0.
        model_type (str, optional): Specification of pre-trained hugging face model used. Defaults to "roberta-base".
        log (bool, optional): Enables/disables logging via. Weights and Biases. Defaults to True.
    """

    # Setup W and B project log
    if log:
        os.environ["WANDB_API_KEY"] = key
        wandb.init(project="sdg_clf", entity="pydqn")
        wandb.config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 3e-5,
        }

    os.chdir(os.path.dirname(__file__))
    utils.seed_everything(seed)
    # Setup correct directory and seed
    save_path = "data/processed"
    save_path += f"/{model_type}"

    # get the dataset dict with splits
    ds_dict = tweet_dataset.get_dataset(tokenizer_type=model_type)

    # convert the model input for every split to tensors
    for ds in ds_dict.values():
        ds.set_format("pt", columns=["input_ids", "attention_mask", "label"])

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

    if not folds:
        dl_train = DataLoader(ds_dict["train"], batch_size=batch_size)
        dl_cv = DataLoader(ds_dict["validation"], batch_size=batch_size)
        trainer = SDGTrainer(
            model=model,
            epochs=epochs,
            criterion=criterion,
            call_tqdm=call_tqdm,
            gpu_index=0,
            metrics=metrics,
        )
        best_val_acc = trainer.train(dl_train, dl_cv)
    else:
        # code for folds which will not be used
        val_accs = []

        # Define k-fold cross validation using folds int
        kf = KFold(n_splits=folds)
        for i, (train_index, cv_index) in enumerate(
            kf.split([*range(ds["train"].num_rows)])
        ):
            print(f"Validating on fold {i}\n")

            # Load train and validation data based on current fold
            ds_train = ds["train"].select(train_index)
            ds_cv = ds["train"].select(cv_index)
            dl_train = DataLoader(ds_train, batch_size=batch_size)
            dl_cv = DataLoader(ds_cv, batch_size=batch_size)

            # Reset weights in model
            model.apply(utils.reset_weights)

            # Call instance of SDGTrainer and evaluate on cross validation set
            trainer = SDGTrainer(
                model=model,
                epochs=epochs,
                criterion=criterion,
                call_tqdm=call_tqdm,
                gpu_index=0,
                metrics=metrics,
            )
            best_val_acc = trainer.train(dl_train, dl_cv)["accuracy"]
            if log:
                wandb.log({"fold": i, "best_val_acc": best_val_acc})

            # Collect accuracies from current fold and store the average accuracy from all folds after completion
            val_accs.append(best_val_acc)
        if log:
            wandb.log({"avg_val_acc": np.mean(val_accs)})


if __name__ == "__main__":
    metrics = {
        "accuracy": {
            "goal": "maximize",
            "metric": torchmetrics.Accuracy(subset_accuracy=True, multiclass=False),
        }
    }
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
        batch_size=8,
        epochs=3,
        multi_label=True,
        call_tqdm=False,
        metrics=metrics,
        model_type="roberta-base",
        log=False,
    )
