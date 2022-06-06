import torch
import random
import os

import numpy as np

from typing import Callable
from tqdm import tqdm, trange

from datetime import datetime
import datasets
import torchmetrics
import transformers
import wandb
from .dataset_utils import load_ds_dict


class Trainer:
    def __init__(
        self,
        model=None,
        metrics: dict = None,
        save_metric: str = None,
        criterion: Callable = None,
        save_filename: str = "best_model",
        gpu_index: int = None,
        save_model: bool = False,
        call_tqdm: bool = True,
        log: bool = True,
        hypers: dict = {"learning_rate": 3e-5,
                        "epochs": 2,
                        "batch_size": 16},
    ):
        """
        save_filename : str
            name of file without extension. Unique id will be added
            when training
        """
        if gpu_index is not None:
            self.device = f"cuda:{gpu_index}"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.lr = hypers["learning_rate"]
        self.optimizer = self.set_optimizer(self.model)
        self.epochs = hypers["epochs"]
        self.save_filename = save_filename
        self.criterion = criterion
        self.save_model = save_model
        self.call_tqdm = call_tqdm
        self.metrics = metrics
        for k, v in self.metrics.items():
            self.metrics[k]["metric"] = v["metric"].to(self.device)
        """
        metrics = {"accuracy": {"goal": "maximize", "metric": torchmetrics.Accuracy}, ...}
        """
        self.save_metric = save_metric
        self.log = log

    def train_step(self, batch):
        """
        :param batch:
        :return:
        Returns a dictionary that must have the key 'loss'.
        Must have key 'n_correct' to calculate accuracy
        """
        raise NotImplementedError("train_step must be implemented")

    def validation_step(self, batch):
        """
        :param batch:
        :return:
        Returns a dictionary that must have the key 'loss'.
        """
        raise NotImplementedError("validation_step must be implemented")

    def train(self, train_dataloader=None, val_dataloader=None):
        print(f"Training on device: {self.device}")
        self.time = datetime.now().strftime("%m%d%H%M%S")
        # validation check
        print("Validation check")
        self.model.eval()
        val_dataloader_iterator = iter(val_dataloader)
        for i in trange(2):
            with torch.no_grad():
                batch = next(val_dataloader_iterator)
                batch = self.move_to(batch)
                val_step_out = self.validation_step(batch)
                self.update_metrics(val_step_out)
        # check if all metrics can be computed
        self.compute_metrics()

        print("Validation check completed\n")

        # train
        self.model.train()
        print("Training")
        for epoch in trange(self.epochs):
            # training
            self.reset_metrics()
            self.best_val_metrics = {
                k: float("inf") if (v["goal"] == "minimize") else -float("inf")
                for k, v in self.metrics.items()
            }
            total_train_loss = 0
            batches = 0
            for batch in tqdm(train_dataloader, disable=not self.call_tqdm):
                batch = self.move_to(batch)
                train_step_out = self.train_step(batch)
                self.step_optimizer(train_step_out)
                self.update_metrics(train_step_out)
                total_train_loss += train_step_out["loss"].detach()
                batches += 1
            epoch_metrics_train = self.compute_metrics()
            avg_train_loss = total_train_loss / batches

            # validation
            self.reset_metrics()
            self.model.eval()

            total_val_loss = 0
            batches = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, disable=not self.call_tqdm):
                    batch = self.move_to(batch)
                    val_step_out = self.validation_step(batch)
                    self.update_metrics(val_step_out)
                    total_val_loss += val_step_out["loss"]
                    batches += 1
            epoch_metrics_val = self.compute_metrics()
            avg_val_loss = total_val_loss / batches

            if self.save_model:
                new_best = False
                if (
                        self.metrics[self.save_metric]["goal"] == "maximize"
                        and epoch_metrics_val[self.save_metric]
                        > self.best_val_metrics[self.save_metric]
                ):
                    new_best = True
                elif (
                        self.metrics[self.save_metric]["goal"] == "minimize"
                        and epoch_metrics_val[self.save_metric]
                        < self.best_val_metrics[self.save_metric]
                ):
                    new_best = True

                if new_best:
                    torch.save(
                        self.model.state_dict(),
                        self.save_filename + "_" + self.time + ".pt",
                    )

            for k, v in epoch_metrics_val.items():
                new_best = False
                if (
                        self.metrics[k]["goal"] == "maximize"
                        and v > self.best_val_metrics[k]
                ):
                    new_best = True
                elif (
                        self.metrics[k]["goal"] == "minimize"
                        and v < self.best_val_metrics[k]
                ):
                    new_best = True

                if new_best:
                    self.best_val_metrics[k] = v

            print(f"Epoch {epoch}:")
            print("Training metrics")
            print(f"    Train loss: {avg_train_loss}")
            for k in self.metrics.keys():
                print(f"    {k}: {epoch_metrics_train[k]}")
            print("Validation metrics")
            print(f"    Validation loss: {avg_val_loss}")

            for k in self.metrics.keys():
                print(f"    {k}: {epoch_metrics_val[k]}")

            if self.log:
                log_train = {f"train_{k}": v for k, v in epoch_metrics_train.items()}
                log_train["train_loss"] = avg_train_loss
                log_val = {f"val_{k}": v for k, v in epoch_metrics_val.items()}
                log_val["val_loss"] = avg_val_loss
                wandb.log(log_train)
                wandb.log(log_val)
        return self.best_val_metrics

    def test(self, test_dataloader):
        self.model.eval()
        n_samples = 0
        test_step_outputs = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, disable=not self.call_tqdm):
                n_samples += self.get_batch_size(batch)
                batch = self.move_to(batch)
                test_step_out = self.validation_step(batch)
                test_step_out = self.detach_outputs(test_step_out)
                test_step_outputs.append(test_step_out)

        avg_outputs = self.avg_outputs(test_step_outputs, n_samples)
        test_loss = avg_outputs["loss"]
        test_accuracy = avg_outputs.get("n_correct")
        print(f"Test loss: {test_loss}")
        print(f"Test accuracy: {test_accuracy}")

    def update_metrics(self, step_outputs: dict):
        for metric in self.metrics.values():
            metric["metric"].update(
                target=step_outputs["label"], preds=step_outputs["prediction"]
            )

    def compute_metrics(self):
        metric_values = {
            k: v["metric"].compute().cpu() for k, v in self.metrics.items()
        }
        return metric_values

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric["metric"].reset()

    def set_optimizer(self, model: torch.nn.Module):
        """
        should return an initialized optimizer with the parameters from model
        """
        raise NotImplementedError("set_optimizer must be implemented")

    def step_optimizer(self, train_step_out):
        loss = train_step_out["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def move_to(self, obj):
        if torch.is_tensor(obj):
            return obj.to(self.device)
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = self.move_to(v)
            return res
        elif isinstance(obj, list):
            res = []
            for v in obj:
                res.append(self.move_to(v))
            return res
        else:
            raise TypeError("Invalid type for move_to")

    @staticmethod
    def detach_outputs(output):
        for k, i in output.items():
            if type(i) == torch.Tensor:
                output[k] = i.detach()
        return output

    @staticmethod
    def get_batch_size(batch):
        """
        computes the batch size of a batch from a dataloader by recursively
        inspecting the first element no matter the type until a tensor is found
        """
        if type(batch) == torch.Tensor:
            return batch.shape[0]
        elif type(batch) == tuple or type(batch) == list:
            return Trainer.get_batch_size(batch[0])
        elif type(batch) == dict:
            return Trainer.get_batch_size(list(batch.values())[0])


class SDGTrainer(Trainer):
    def __init__(self, tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def train_step(self, batch):
        tokens = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        out = self.model(tokens, attention_mask)["logits"]
        loss = self.criterion(out, label.float())
        return {"loss": loss, "prediction": out, "label": label}

    def validation_step(self, batch):
        tokens = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        out = self.model(tokens, attention_mask)["logits"]
        loss = self.criterion(out, label.float())
        return {"loss": loss, "prediction": out, "label": label}

    def set_optimizer(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        return optimizer

    def update_metrics(self, step_outputs: dict):
        for metric in self.metrics.values():
            metric["metric"].update(
                target=step_outputs["label"],
                preds=step_outputs["prediction"].sigmoid() if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss)
                else step_outputs["prediction"],
            )

    def prepare_long_text_input(self, input_ids, max_length: int = 260):
        input_ids = input_ids.tolist()[0]
        input_ids = [input_ids[x:x + max_length - 2] for x in range(0, len(input_ids), max_length - 2)]
        attention_masks = []
        for i in range(len(input_ids)):
            input_ids[i] = [tokenizer.cls_token_id] + input_ids[i] + [tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids[i])
            while len(input_ids[i]) < max_length:
                input_ids[i] += [tokenizer.pad_token_id]
                attention_mask += [0]
            attention_masks.append(attention_mask)

        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.tensor(attention_masks, device=self.device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def long_text_step(self, model_inputs):
        out = self.model(**model_inputs).logits
        out = torch.mean(out, dim=0)
        return out

    def test_long_text(self, dataloader, max_length: int = 260):
        self.model.eval()
        self.reset_metrics()
        predictions = []
        labels = []
        for sample in dataloader:
            with torch.no_grad():
                input_ids = sample["input_ids"]
                label = torch.squeeze(sample["label"], dim=0)
                labels.append(label)
                model_inputs = self.prepare_long_text_input(input_ids, max_length=max_length)
                prediction = self.long_text_step(model_inputs)
                predictions.append(prediction)
        self.update_metrics({"label": torch.stack(labels, dim=0).to(self.device), "prediction": torch.stack(predictions, dim=0)})
        metrics = self.compute_metrics()
        print(metrics)

        return metrics


if __name__ == "__main__":
    # ds_dict = datasets.load_from_disk("../data/processed/scopus/roberta-base")
    ds_dict = load_ds_dict("roberta-base", tweet=False, path_data="../data")
    test = ds_dict["test"]
    # sample = test["Abstract"][0]
    tokenizer = transformers.AutoTokenizer.from_pretrained("../tokenizers/roberta-base")
    sdg_model = transformers.AutoModelForSequenceClassification.from_pretrained("../pretrained_models/roberta_base", num_labels=17)
    sdg_model.cuda()
    sdg_model.load_state_dict(torch.load("../pretrained_models/best_model_0603141006.pt"))
    # metrics = {
    #     "accuracy": {
    #         "goal": "maximize",
    #         "metric": torchmetrics.Accuracy(subset_accuracy=True),
    #     }
    # }
    multilabel = True
    for i in np.linspace(0, 1, 9):
        metrics = {
            "accuracy": {
                "goal": "maximize",
                "metric": torchmetrics.Accuracy(threshold=i,subset_accuracy=True, multiclass=not multilabel),
            },
            "auroc": {
            "goal": "maximize",
            "metric": torchmetrics.AUROC(num_classes=17),
            },
            "precision": {
                "goal": "maximize",
                "metric": torchmetrics.Precision(threshold=i,num_classes=17, multiclass=not multilabel),
            },
            "recall": {
                "goal": "maximize",
                "metric": torchmetrics.Recall(threshold=i,num_classes=17, multiclass=not multilabel),
            },
            "f1": {
                "goal": "maximize",
                "metric": torchmetrics.F1Score(threshold=i,num_classes=17, multiclass=not multilabel),
            },
        }
        trainer = SDGTrainer(tokenizer=tokenizer, model=sdg_model, metrics=metrics)
        # model_inputs = trainer.prepare_long_text_input(sample)
        # pred =
        test.set_format("pt", columns=["input_ids", "label"])
        dl = torch.utils.data.DataLoader(test)
        trainer.test_long_text(dl)

    # dataset = datasets.load_from_disk("../data/processed/scopus/roberta-base")
    # dataset_train = dataset["train"]
    # dataset_val = dataset["test"]
    # dataset_train.set_format(
    #     type="torch", columns=["input_ids", "attention_mask", "label"]
    # )
    # dataset_val.set_format(
    #     type="torch", columns=["input_ids", "attention_mask", "label"]
    # )
    # dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16)
    # dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
    # metrics = {
    #     "accuracy": {
    #         "goal": "maximize",
    #         "metric": torchmetrics.Accuracy(subset_accuracy=True),
    #     }
    # }
    # sdg_model = model.get_model(pretrained_path="pretrained_models/roberta_base")
    # criterion = torch.nn.BCEWithLogitsLoss()
    # trainer = SDGTrainer(
    #     metrics=metrics, epochs=3, model=sdg_model, criterion=criterion
    # )
    # trainer.long_text_step(dataset_val[0])

    """best_val_metrics = trainer.train(
        train_dataloader=dataloader_train, val_dataloader=dataloader_val
    )
    print(best_val_metrics)"""
