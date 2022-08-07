import argparse
import dataclasses
import os

import torch
import pytorch_lightning as pl

import wandb
from api_key import key
from sdg_clf import dataset_utils
from sdg_clf import modelling
from sdg_clf import training
from sdg_clf import utils


@dataclasses.dataclass
class TrainingParameters:
    learning_rate: float = 3e-5
    batch_size: int = 32
    epochs: int = 2
    weight_decay: float = 1e-2


def main(
        call_tqdm: bool = True,
        seed: int = 0,
        model_type: str = "roberta-base",
        log: bool = True,
        save_model: bool = True,
        training_parameters: TrainingParameters = TrainingParameters(),
        tags: list = None,
        notes: str = "Evaluating baseline model",
        frac: float = 1.0,
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
    tags = [model_type] + tags if tags is not None else [model_type]
    if not log:
        os.environ["WANDB_MODE"] = "offline"
    # Setup W and B project log
    os.environ["WANDB_API_KEY"] = key
    run = wandb.init(project="sdg_clf", config=dataclasses.asdict(training_parameters), tags=tags, notes=notes)
    if not log:
        save_file_name = "model"
    else:
        save_file_name = run.name

    # Setup correct directory and seed
    os.chdir(os.path.dirname(__file__))
    utils.seed_everything(seed)
    transformer = modelling.load_model(model_type=model_type)

    dl_train = dataset_utils.get_dataloader("twitter", model_type, "train")
    dl_val = dataset_utils.get_dataloader("twitter", model_type, "val")

    # get metrics

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        fast_dev_run=True,
        precision=16,
        limit_train_batches=frac,
        min_epochs=training_parameters.epochs // 2,
        max_epochs=training_parameters.epochs,
        # logger=...,
        # resume_from_checkpoint=...,
        # weights_save_path=...,
        # callbacks=...,
        # enable_checkpointing=...,
    )
    LitModel = training.LitSDG(model=transformer)


    trainer.fit(LitModel, train_dataloaders=dl_train, val_dataloaders=dl_val)

    metrics = utils.get_metrics()
    # trainer = training.SDGTrainer(
    #     model=model,
    #     criterion=criterion,
    #     call_tqdm=call_tqdm,
    #     gpu_index=0,
    #     metrics=metrics,
    #     log=log,
    #     save_file_name=save_file_name,
    #     save_model=save_model,
    #     save_metric="f1",
    #     hypers=dataclasses.asdict(training_parameters),
    # )
    # trainer.train(dl_train, dl_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", help="number of batches sent to model", type=int, default=32)
    parser.add_argument("-l", "--log", help="log results to weights and biases", action='store_true', default=False)
    parser.add_argument("-s", "--save", help="save models during training", action="store_true", default=True)
    parser.add_argument("-e", "--epochs", help="number of epochs to train model", type=int, default=2)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=3e-5)
    parser.add_argument("-wd", "--weight_decay", help="optimizer weight decay", type=float, default=1e-2)
    parser.add_argument("-t", "--tags", help="tags for experiment run", nargs="+", default=None)
    parser.add_argument("-nt", "--notes", help="notes for a specific experiment run", type=str,
                        default="run with base parameters")
    parser.add_argument("-mt", "--model_type", help="specify model type to train", type=str, default="roberta-base")
    parser.add_argument("-f", "--frac", help='fraction of training data to use for training', type=float, default=1.0)
    args = parser.parse_args()
    main(
        call_tqdm=True,
        model_type=args.model_type,
        log=args.log,
        save_model=args.save,
        training_parameters=TrainingParameters(args.learning_rate, args.batch_size, args.epochs, args.weight_decay),
        tags=args.tags,
        notes=args.notes,
    )
