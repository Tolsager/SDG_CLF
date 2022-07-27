import os
from datetime import datetime
from typing import Callable, Union

import numpy as np
import torch
import wandb
from tqdm import tqdm, trange

from sdg_clf.utils import get_metrics


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
                            "batch_size": 16,
                            "weight_decay": 1e-2},
    ):
        """
        Class initializer

                Args:
                    model: pretrained classification model
                    epochs (int, optional): Number of epochs during training. Defaults to None
                    metrics (dict, optional): Dictionary of metrics to measure the performance of the Trainer. Defaults to None
                    save_metric (str, optional): Metrics of interest to optimize. Defaults to None
                    criterion (Callable, optional): Criterion the Trainer learns based on. Defaults to None
                    save_filename (str, optional): Name of file the model will be saved in without extension. Unique id will be added when training. Defaults to "best_model"
                    gpu_index (int, optional): GPU model number to run the Trainer on, if available. Defaults to None
                    save_model (bool, optional): Whether the model should be saved. Defaults to False
                    call_tqdm (bool, optional): Whether loops display progress bars. Defaults to True
                    log (boll, optional): Whether to log metrics in weights and biases. Defaults to True
        """
        if gpu_index is not None:
            self.device = f"cuda:{gpu_index}"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.hypers = hypers
        self.optimizer = self.set_optimizer(self.model)
        self.epochs = hypers["epochs"]
        self.save_file_name = save_filename
        self.criterion = criterion
        self.save_model = save_model
        self.call_tqdm = call_tqdm
        self.metrics = metrics
        """
        metrics = {"accuracy": {"goal": "maximize", "metric": torchmetrics.Accuracy}, ...}
        """
        self.save_metric = save_metric
        self.log = log

    def train_step(self, batch):
        """
        Completes training step

        Args:
            batch: a batch of data

        Returns:
            Raises an implementation error
            In subsequent classes, that inherit from Trainer, the train_step returns a dictionary that must have a key 'loss'
        """
        raise NotImplementedError("train_step must be implemented")

    def validation_step(self, batch):
        """
        Completes validation step

        Args:
            batch: a batch of data

        Returns:
            Raises an implementation error
            In subsequent classes, that inherit from Trainer, the validation_step returns a dictionary that must have a key 'loss'
        """
        raise NotImplementedError("validation_step must be implemented")

    def train(self, train_dataloader=None, val_dataloader=None):
        """
        First runs a validation check over the validation data before training on the training data

        Args:
            train_dataloader: Training data
            val_dataloader: Validation data

        Returns:
            dictionary of best values for each metrics in the self.metrics dictionary
        """
        print(f"Training on device: {self.device}")
        # set metrics to device
        self.set_metrics_to_device()
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
                    os.makedirs(f"finetuned_models", exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        f"finetuned_models/{self.save_file_name}_{self.time}.pt",
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
                wandb.log(log_train | log_val)
        return self.best_val_metrics

    def test(self, test_dataloader):
        """
        Predicts on test data

        Args:
            test_dataloader: test data
        """
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
        """
        Updates the metrics of the self.metrics dictionary based on outputs from the model

        Args:
            step_outputs (dict): Dictionary of metric outputs from the neural network
        """
        for metric in self.metrics.values():
            metric["metric"].update(
                target=step_outputs["label"], preds=step_outputs["prediction"]
            )

    def compute_metrics(self):
        """
        Computes the metrics in the self.metric dictionary using the .compute() torchmetrics method

        Returns:
            Dictionary of metrics computed by torchmetrics
        """
        metric_values = {
            k: v["metric"].compute().cpu() for k, v in self.metrics.items()
        }
        return metric_values

    def reset_metrics(self):
        """
        Resets the values of the self.metrics dictionary
        """
        for metric in self.metrics.values():
            metric["metric"].reset()

    def set_optimizer(self, model: torch.nn.Module):
        """
        Returns an initialized optimizer with the parameters from model

        Args:
            model (torch.nn.Module): pretrained classification model
        """
        raise NotImplementedError("set_optimizer must be implemented")

    def step_optimizer(self, train_step_out):
        """
        Returns a step optimizer based on the train_step_out

        Args:
            train_step_out: output metrics from training the model
        """
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

        Args:
            batch: a batch of data
        """
        if type(batch) == torch.Tensor:
            return batch.shape[0]
        elif type(batch) == tuple or type(batch) == list:
            return Trainer.get_batch_size(batch[0])
        elif type(batch) == dict:
            return Trainer.get_batch_size(list(batch.values())[0])

    def set_metrics_to_device(self):
        for k, v in self.metrics.items():
            self.metrics[k]["metric"] = v["metric"].to(self.device)


class SDGTrainer(Trainer):
    def __init__(self, tokenizer=None, *args, **kwargs):
        """
        Class initializer

        Args:
            tokenizer: A tokenizer (ideally a huggingface tokenizer)
        """
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def train_step(self, batch):
        """
        Completes training step

        Args:
            batch: a batch of data

        Returns:
            In subsequent classes, that inherit from Trainer, the train_step returns a dictionary that must have a key 'loss'
        """
        tokens = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        out = self.model(tokens, attention_mask)["logits"]
        loss = self.criterion(out, label.float())
        return {"loss": loss, "prediction": out, "label": label}

    def validation_step(self, batch):
        """
        Completes validation step

        Args:
            batch: a batch of data

        Returns:
            In subsequent classes, that inherit from Trainer, the validation_step returns a dictionary that must have a key 'loss'
        """
        tokens = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        out = self.model(tokens, attention_mask)["logits"]
        loss = self.criterion(out, label.float())
        return {"loss": loss, "prediction": out, "label": label}

    def set_optimizer(self, model):
        """
        Initialize optimizer based on model parameters

        Args:
            model: pretrained classification model

        Returns:
            Adam optimizer with improved weight decay
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.hypers["learning_rate"],
                                      weight_decay=self.hypers["weight_decay"])
        return optimizer

    def update_metrics(self, step_outputs: dict):
        """
        Updates the metrics of the self.metrics dictionary based on outputs from the model

        Args:
            step_outputs (dict): Dictionary of metric outputs from the neural network
        """
        for metric in self.metrics.values():
            metric["metric"].update(
                target=step_outputs["label"],
                preds=step_outputs["prediction"].sigmoid() if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss)
                else step_outputs["prediction"],
            )

    def prepare_long_text_input(self, input_ids: Union[list[int], torch.Tensor], max_length: int = 260,
                                step_size: int = 260):
        """
        Prepare longer text for classification task by breaking into chunks

        Args:
            input_ids: Tokenized full text
            max_length (int, optional): Max length of each tokenized text chunk

        Returns:
            Dictionary of chunked data ready for classification
        """
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()[0]
        input_ids = [input_ids[x:x + max_length - 2] for x in range(0, len(input_ids), step_size)]
        attention_masks = []
        for i in range(len(input_ids)):
            input_ids[i] = [self.tokenizer.cls_token_id] + input_ids[i] + [self.tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids[i])
            while len(input_ids[i]) < max_length:
                input_ids[i] += [self.tokenizer.pad_token_id]
                attention_mask += [0]
            attention_masks.append(attention_mask)

        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.tensor(attention_masks, device=self.device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def long_text_step(self, model_outputs: torch.Tensor):
        """
        Completes one prediction step for one model input sample

        Args:
            model_inputs: input data for classification

        Returns:
            model output classification results
        """
        prediction = torch.mean(model_outputs, dim=0)
        return prediction

    def test_scopus_mean(self, dataloader, max_length: int = 260, step_size=260, return_all_metrics=False):
        """
        Testing the classification performance of the model based on long texts

        Args:
            dataloader: data for the test
            max_length: max length for chunking of long texts

        Returns:
            Classification metrics of task
        """
        self.model.eval()
        predictions = []
        labels = []
        threshold_results = {}
        best_f1 = (0, 0)
        for sample in dataloader:
            with torch.no_grad():
                input_ids = sample["input_ids"]
                label = torch.squeeze(sample["label"], dim=0)
                labels.append(label)
                model_inputs = self.prepare_long_text_input(input_ids, max_length=max_length, step_size=step_size)
                model_outputs = self.model(**model_inputs).logits.sigmoid()
                prediction = self.long_text_step(model_outputs)
                predictions.append(prediction)
        for threshold in np.linspace(0, 1, 101):
            self.metrics = get_metrics(threshold=threshold)
            self.set_metrics_to_device()
            self.reset_metrics()
            self.update_metrics({"label": torch.stack(labels, dim=0).to(self.device),
                                 "prediction": torch.stack(predictions, dim=0).to(self.device)})
            metrics = self.compute_metrics()
            if metrics["f1"] > best_f1[1]:
                best_f1 = (threshold, metrics["f1"])
                if not return_all_metrics:
                    threshold_results = metrics
            if return_all_metrics:
                threshold_results[threshold] = metrics
        return threshold_results, best_f1

    def test_scopus_any(self, dataloader, max_length: int = 260, step_size=260):
        self.model.eval()
        model_outputs_all = []
        labels = []
        best_f1 = (0, 0)
        for sample in dataloader:
            with torch.no_grad():
                input_ids = sample["input_ids"]
                label = torch.squeeze(sample["label"], dim=0)
                labels.append(label)
                model_inputs = self.prepare_long_text_input(input_ids, max_length=max_length, step_size=step_size)
                model_outputs = self.model(**model_inputs).logits.sigmoid()
                model_outputs_all.append(model_outputs)

        self.metrics = get_metrics(0.5)
        self.set_metrics_to_device()
        for threshold in np.linspace(0, 1, 101):
            predictions = []
            for o in model_outputs_all:
                prediction = torch.any(o > threshold, dim=0).int()
                predictions.append(prediction)

            self.reset_metrics()
            self.update_metrics({"label": torch.stack(labels, dim=0).to(self.device),
                                 "prediction": torch.stack(predictions, dim=0).to(self.device)})
            metrics = self.compute_metrics()
            if metrics["f1"] > best_f1[1]:
                best_f1 = (threshold, metrics["f1"])
                threshold_results = metrics
        return threshold_results, best_f1

    def infer_sample(self, text: str, step_size: int = 260, max_length: int = 260, strategy: str = "mean"):
        text = text.lower()
        input_ids = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        model_inputs = self.prepare_long_text_input(input_ids, max_length=max_length, step_size=step_size)
        model_outputs = self.model(**model_inputs).logits.sigmoid()
        prediction = self.long_text_step(model_outputs)
        return prediction.tolist()
