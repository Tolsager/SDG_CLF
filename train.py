import argparse
import dataclasses
import datetime
import os

import pytorch_lightning as pl

import api_key
from sdg_clf import base, dataset_utils, modelling, training, utils


def get_save_dirpath(model_type: str):
    return os.path.join("finetuned_models", os.path.dirname(model_type))



def main(
        experiment_params: base.ExperimentParams,
        hparams: base.HParams,
):
    """
    Train a model on the Twitter dataset.
    Args:
        experiment_params: parameters for the experiment
        hparams: an instance of the HParams class

    Returns:
        None

    """
    tags = [experiment_params.model_type] + experiment_params.tags if experiment_params.tags is not None else [
        experiment_params.model_type]
    if experiment_params.ckpt_path is not None:
        experiment_params.notes = f"Resuming training from {experiment_params.ckpt_path}" + experiment_params.notes + "\n"
    # Setup W and B project log
    os.environ["WANDB_API_KEY"] = api_key.key

    utils.seed_everything(experiment_params.seed)
    model = modelling.load_model(model_type=experiment_params.model_type)

    dl_train = dataset_utils.get_dataloader("twitter", experiment_params.model_type, "train",
                                            batch_size=hparams.batch_size)
    dl_val = dataset_utils.get_dataloader("twitter", experiment_params.model_type, "val", batch_size=hparams.batch_size)

    # set up model checkpoint callback
    save_dirpath = get_save_dirpath(experiment_params.model_type)
    save_filename = f"{experiment_params.model_type}_{datetime.datetime.now().strftime('%d%m%H%M%S')}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dirpath, filename=save_filename,
                                                       monitor="val_micro_f1", mode="max",
                                                       save_top_k=1 if not experiment_params.debug else 0)
    logger = pl.loggers.wandb.WandbLogger(project="sdg_clf", tags=tags, name=save_filename,
                                          notes=experiment_params.notes,
                                          offline=experiment_params.debug)
    logger.log_hyperparams(dataclasses.asdict(hparams) | {"save_filename": save_filename})
    callbacks = [checkpoint_callback]

    if experiment_params.debug:
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
            limit_train_batches=hparams.frac_train,
            limit_val_batches=hparams.frac_val,
            max_epochs=hparams.max_epochs,
            callbacks=callbacks,
            logger=logger,
        )
    LitModel = training.LitSDG(model=model, hparams=hparams)

    if experiment_params.ckpt_path is not None:
        trainer.fit(LitModel, dl_train, dl_val, ckpt_path=experiment_params.ckpt_path)
    else:
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
                        default="")
    parser.add_argument("-mt", "--model_type", help="specify model type to train", type=str, default="roberta-base")
    parser.add_argument("-ft", "--frac_train", help='fraction of training data to use for training', type=float,
                        default=1.0)
    parser.add_argument("-fv", "--frac_val", help='fraction of training data to use for validation', type=float,
                        default=1.0)
    parser.add_argument("-se", "--seed", help="seed for random number generator", type=int, default=0)
    parser.add_argument("-ck", "--ckpt_path", help="path to checkpoint to load", type=str, default=None)
    args = parser.parse_args()
    main(
        hparams=base.HParams(batch_size=args.batch_size, max_epochs=args.epochs, lr=args.learning_rate,
                             frac_train=args.frac_train, frac_val=args.frac_val, weight_decay=args.weight_decay),
        experiment_params=base.ExperimentParams(seed=args.seed, debug=args.debug, tags=args.tags,
                                                model_type=args.model_type, notes=args.notes, ckpt_path=args.ckpt_path),
    )
