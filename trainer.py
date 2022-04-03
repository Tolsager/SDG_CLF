import torch
import random
import os

import numpy as np

from typing import Callable
from tqdm import tqdm, trange

from datetime import datetime


class Trainer:
    def __init__(self, model=None, optimizer: Callable=None,
                 epochs: int = None, seed: int=0, criterion: Callable=None,
                 save_filename: str = 'best_model', gpu_index: int = None):
        """
        save_filename : str
            name of file without extension. Unique id will be added
            when training
        """
        self.seed = seed
        if gpu_index is not None:
            self.device = torch.cuda.device(gpu_index)
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.optimizer = self.set_optimizer(self.model)
        self.epochs = epochs
        self.save_filename = save_filename
        self.criterion=criterion

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
        Must have key 'n_correct' to calculate accuracy 
        """
        raise NotImplementedError("validation_step must be implemented")

    def train(self, train_dataloader=None, val_dataloader=None):
        
        print(f"Training on device: {self.device}")
        self.seed_everything(self.seed)
        best_acc = 0
        self.time = datetime.now().strftime('%m%d%H%M%S')
        # validation check
        print("Validation check")
        self.model.eval()
        val_dataloader_iterator = iter(val_dataloader)
        n_samples = 0
        val_step_outputs = []
        for i in trange(2):
            with torch.no_grad():
                batch = next(val_dataloader_iterator)
                n_samples += self.get_batch_size(batch)
                batch = self.move_to(batch)
                val_step_out = self.validation_step(batch)
                val_step_out = self.detach_outputs(val_step_out)
                val_step_outputs.append(val_step_out)
        
        avg_outputs = self.avg_outputs(val_step_outputs, n_samples)
        print("Validation check completed\n")
    
        # train
        self.model.train()
        print("Training")
        for epoch in trange(self.epochs):
            # training
            n_samples = 0
            train_step_outputs = []
            for batch in tqdm(train_dataloader):
                n_samples += self.get_batch_size(batch)
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
                for batch in tqdm(val_dataloader):
                    n_samples += self.get_batch_size(batch)
                    batch = self.move_to(batch)
                    val_step_out = self.validation_step(batch)
                    val_step_out = self.detach_outputs(val_step_out)
                    val_step_outputs.append(val_step_out)
                
                avg_outputs = self.avg_outputs(val_step_outputs, n_samples)
                val_loss = avg_outputs['loss']
                val_accuracy = avg_outputs.get('n_correct')
                
            if val_accuracy > best_acc:
                torch.save(self.model.state_dict(), self.save_filename + '_' + self.time + '.pt')
                
            print(f"Epoch {epoch}:")
            print(f"    Train loss: {train_loss}")
            if train_accuracy is not None:
                print(f"    Train Accuracy: {train_accuracy}")

            print(f"    Validation loss: {val_loss}")
            if train_accuracy is not None:
                print(f"    Validation Accuracy: {val_accuracy}")

    def test(self, test_dataloader):
        self.model.eval()
        n_samples = 0
        test_step_outputs = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                n_samples += self.get_batch_size(batch)
                batch = self.move_to(batch)
                test_step_out = self.validation_step(batch)
                test_step_out = self.detach_outputs(test_step_out)
                test_step_outputs.append(test_step_out)
        
        avg_outputs = self.avg_outputs(test_step_outputs, n_samples)
        test_loss = avg_outputs['loss']
        test_accuracy = avg_outputs.get('n_correct')
        print(f"Test loss: {test_loss}")
        print(f"Test accuracy: {test_accuracy}")


    def set_optimizer(self, model: torch.nn.Module):
        """
        should return an initialized optimizer with the parameters from model
        """
        raise NotImplementedError("set_optimizer must be implemented")

    def step_optimizer(self, train_step_out):
        loss = train_step_out['loss']
        
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
    def avg_outputs(step_outputs, n_samples):
        metrics_avg = {k: 0 for k in step_outputs[0].keys()}
        
        for m in metrics_avg.keys():
            total = np.sum([out[m] for out in step_outputs])
            metrics_avg[m] = total / n_samples
        
        return metrics_avg

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def train_step(self, batch):
        tokens = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['labels']
        out = self.model(tokens, attention_mask)['logits']
        loss = self.criterion(out, label)
        n_correct = torch.sum(torch.argmax(out, dim=1) == label)
        return {'loss': loss, 'n_correct': n_correct}

    def validation_step(self, batch):
        tokens = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['labels']
        out = self.model(tokens, attention_mask)['logits']
        loss = self.criterion(out, label)
        n_correct = torch.sum(torch.argmax(out, dim=1) == label)
        return {'loss': loss, 'n_correct': n_correct}

    def set_optimizer(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        return optimizer