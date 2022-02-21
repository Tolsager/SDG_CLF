import torch
import random
import os

import numpy as np

from typing import Callable
from tqdm import tqdm, trange


class Trainer:
    def __init__(self, model=None, optimizer: Callable = None, lr: float = None,
                 epochs: int = None, seed: int = 0, criterion: Callable = None):
        self.seed_everything(seed)
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.model = model.to(self.device)
        self.optimizer = optimizer(model.parameters(), lr=lr)
        self.epochs = epochs
        self.criterion = criterion

    def train_step(self, batch):
        tokens, label = batch
        out = self.model(tokens)
        loss = self.criterion(out, label)
        n_correct = torch.sum(torch.argmax(out, dim=1) == label)
        return {'loss': loss, 'n_correct': n_correct}

    def validation_step(self, batch):
        tokens, label = batch
        out = self.model(tokens)
        loss = self.criterion(out, label)
        n_correct = torch.sum(torch.argmax(out, dim=1) == label)
        return {'loss': loss, 'n_correct': n_correct}

    def train(self, train_dataloader=None, val_dataloader=None):
        # validation check
        print("Validation check")
        self.model.eval()
        val_dataloader_iterator = iter(val_dataloader)
        n_samples = 0
        val_step_outputs = []
        for i in trange(2):
            with torch.no_grad():
                batch = next(val_dataloader_iterator)
                n_samples += batch[0].shape[0]
                batch = self.move_to(batch)
                val_step_out = self.validation_step(batch)
                val_step_out = self.detach_outputs(val_step_out)
                val_step_outputs.append(val_step_out)

        self.avg_outputs(val_step_outputs, n_samples)
        print("Validation check completed\n")

        # train
        self.model.train()
        print("Training")
        for epoch in trange(self.epochs):
            # training
            n_samples = 0
            train_step_outputs = []
            for batch in tqdm(train_dataloader):
                n_samples += batch[0].shape[0]
                batch = self.move_to(batch)
                train_step_out = self.train_step(batch)
                self.step_optimizer(train_step_out)
                train_step_out = self.detach_outputs(train_step_out)
                train_step_outputs.append(train_step_out)

            avg_outputs = self.avg_outputs(train_step_outputs, n_samples)
            train_loss = avg_outputs['loss']
            train_accuracy = avg_outputs.get('n_correct')

            # validation
            self.model.eval()
            n_samples = 0
            val_step_outputs = []
            with torch.no_grad():
                for batch in tqdm(train_dataloader):
                    n_samples += batch[0].shape[0]
                    batch = self.move_to(batch)
                    val_step_out = self.validation_step(batch)
                    val_step_out = self.detach_outputs(val_step_out)
                    val_step_outputs.append(val_step_out)

                avg_outputs = self.avg_outputs(val_step_outputs, n_samples)
                val_loss = avg_outputs['loss']
                val_accuracy = avg_outputs.get('n_correct')

            print(f"Epoch {epoch}:")
            print(f"    Train loss: {train_loss}")
            if train_accuracy is not None:
                print(f"    Train Accuracy: {train_accuracy}")

            print(f"    Validation loss: {val_loss}")
            if train_accuracy is not None:
                print(f"    Validation Accuracy: {val_accuracy}")

    def step_optimizer(self, train_step_out):
        loss = train_step_out['loss']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def avg_outputs(self, step_outputs, n_samples):
        metrics_avg = {k: 0 for k in step_outputs[0].keys()}

        for m in metrics_avg.keys():
            total = np.sum([out[m] for out in step_outputs])
            metrics_avg[m] = total / n_samples

        return metrics_avg

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
    def seed_everything(seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

    @staticmethod
    def detach_outputs(output):
        for k, i in output.items():
            if type(i) == torch.Tensor:
                output[k] = i.detach().cpu()
        return output

