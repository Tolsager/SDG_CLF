import argparse
import os

import pytorch_lightning as pl

import api_key
from sdg_clf import base
from sdg_clf import dataset_utils
from sdg_clf import modelling
from sdg_clf import training
from sdg_clf import utils


def get_save_dirpath(model_type: str):
    return os.path.join("finetuned_models", os.path.dirname(model_type))


def get_save_filename(model_type: str):
    model_number = modelling.get_next_model_number(model_type)
    save_filename = os.path.basename(model_type) + f"_model{model_number}"
    return save_filename


def main(
        debug: bool = False,
        seed: int = 0,
        model_type: str = "roberta-base",
        hparams: base.HParams = base.HParams(),
        tags: list = None,
        notes: str = "Evaluating baseline model",
):
    """
    Train a model on the Twitter dataset.
    Args:
        debug: if True, run on a small subset of the data, log offline, and don't save
        seed: seed used for the model
        model_type: huggingface model type
        hparams: an instance of the HParams class
        tags: tags to log to wandb
        notes: notes to log to wandb

    Returns:
        None

    """
    tags = [model_type] + tags if tags is not None else [model_type]
    # Setup W and B project log
    os.environ["WANDB_API_KEY"] = api_key.key

    utils.seed_everything(seed)
    model = modelling.load_model(model_type=model_type)

    dl_train = dataset_utils.get_dataloader("twitter", model_type, "train")
    dl_val = dataset_utils.get_dataloader("twitter", model_type, "val")

    # set up model checkpoint callback
    save_dirpath = get_save_dirpath(model_type)
    save_filename = get_save_filename(model_type)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dirpath, filename=save_filename,
                                                       monitor="val_micro_f1", mode="max",
                                                       save_top_k=1 if not debug else 0)
    logger = pl.loggers.wandb.WandbLogger(project="sdg_clf", tags=tags, name=save_filename, notes=notes,
                                          offline=debug)
    callbacks = [checkpoint_callback]

    if debug:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision=16,
            limit_train_batches=2,
            limit_val_batches=2,
            max_epochs=1,
            callbacks=callbacks,
            logger=logger,
        )
    else:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision=16,
            limit_train_batches=hparams.frac,
            max_epochs=hparams.max_epochs,
            callbacks=callbacks,
            logger=logger,
        )
    LitModel = training.LitSDG(model=model, hparams=hparams)

    trainer.fit(LitModel, train_dataloaders=dl_train, val_dataloaders=dl_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", help="number of batches sent to model", type=int, default=8)
    parser.add_argument("-d", "--debug", help="debugging doesn't save and logs offline", action="store_true",
                        default=False)
    parser.add_argument("-e", "--epochs", help="number of epochs to train model", type=int, default=2)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=3e-5)
    parser.add_argument("-wd", "--weight_decay", help="optimizer weight decay", type=float, default=1e-2)
    parser.add_argument("-t", "--tags", help="tags for experiment run", nargs="+", default=None)
    parser.add_argument("-nt", "--notes", help="notes for a specific experiment run", type=str,
                        default="run with base parameters")
    parser.add_argument("-mt", "--model_type", help="specify model type to train", type=str, default="roberta-base")
    parser.add_argument("-f", "--frac", help='fraction of training data to use for training', type=float, default=1.0)
    parser.add_argument("-se", "--seed", help="seed for random number generator", type=int, default=0)
    args = parser.parse_args()
    main(
        model_type=args.model_type,
        hparams=base.HParams(batch_size=args.batch_size, max_epochs=args.epochs, lr=args.learning_rate,
                             frac=args.frac, ),
        tags=args.tags,
        notes=args.notes,
        debug=args.debug,
        seed=args.seed
    )
