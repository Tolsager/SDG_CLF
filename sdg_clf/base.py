import torch
import torchmetrics
import transformers
from sdg_clf import utils, modelling
import numpy as np
import dataclasses


# transformer class with a model and a tokenizer
class Transformer:
    def __init__(self, model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, max_length: int = 260):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = model.device

    def prepare_input_ids(self, text: str) -> torch.Tensor:
        input_ids = self.tokenizer(text, add_special_tokens=False)
        input_ids = input_ids["input_ids"]
        input_ids = [input_ids[x:x + self.max_length - 2] for x in range(0, len(input_ids), self.max_length)]
        # add bos and eos tokens to each input_ids
        input_ids = [[self.tokenizer.cls_token_id] + x + [self.tokenizer.eos_token_id] for x in input_ids]
        # pad input_ids to max_length
        input_ids[-1] = input_ids[-1] + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids[-1]))
        return torch.tensor(input_ids)

    def prepare_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows, columns = input_ids.shape
        attention_mask = torch.ones((rows - 1, columns))
        last_mask = torch.tensor([[1 if id != self.tokenizer.pad_token_id else 0 for id in input_ids[-1, :]]])
        attention_mask = torch.concat((attention_mask, last_mask), dim=0)
        return attention_mask

    def prepare_model_inputs(self, text: str) -> dict[str, torch.Tensor]:
        input_ids = self.prepare_input_ids(text)
        attention_mask = self.prepare_attention_mask(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def predict_sample_no_threshold(self, text: str) -> torch.Tensor:
        """
        Predict the sample with the model and return the probabilities of each class.

        The text is split into chunks of 260 tokens.
        There can therefore be several chunks.
        A prediction is made on each chunk and the sigmoid activation function applied to each.
        The output is of shape (n_chunks, 17)

        Args:
            text: the text to predict on

        Returns:
            predictions for each chunk

        """
        if text is np.nan:
            return torch.tensor([0] * 17)
        # get model inputs
        model_inputs = self.prepare_model_inputs(text)
        # get predictions
        with torch.no_grad():
            # set inputs to device
            model_inputs = utils.move_to(model_inputs, self.device)
            # forward pass
            outputs = self.model(**model_inputs).logits
            outputs = torch.sigmoid(outputs).cpu()
        return outputs

    def predict_multiple_samples_no_threshold(self, samples: list[str]) -> list[torch.Tensor]:
        """
        Predict each sample without threshold
        Args:
            samples: texts to predict on

        Returns:
            predictions for each sample

        """
        return [self.predict_sample_no_threshold(text) for text in samples]


def get_transformer(model_type: str, model_weights: str) -> Transformer:
    # get model
    model = modelling.load_model(model_type, model_weights)
    # get tokenizer
    tokenizer = utils.get_tokenizer(model_type)
    # create transformer
    transformer = Transformer(model, tokenizer)
    return transformer


def get_multiple_transformers(model_types: list[str], model_weights: list[str]) -> list[Transformer]:
    return [get_transformer(model_type, model_weight) for model_type, model_weight in zip(model_types, model_weights)]


@dataclasses.dataclass
class HParams:
    lr: float = 3e-6
    weight_decay: float = 1e-2
    max_epochs: int = 10
    batch_size: int = 32
    frac_train: float = 1.0
    frac_val: float = 1.0


@dataclasses.dataclass
class ExperimentParams:
    seed: int = 0
    model_type: str = "roberta-base"
    ckpt_path: str = None
    tags: list[str] = None
    debug: bool = False
    notes: str = None


class Metrics:
    def __init__(self):
        self.metrics = {
            "exact_match_ratio": torchmetrics.Accuracy(num_classes=17, subset_accuracy=True, multiclass=False),
            "precision_micro": torchmetrics.Precision(num_classes=17, multiclass=False, average="micro"),
            "recall_micro": torchmetrics.Recall(num_classes=17, multiclass=False, average="micro"),
            "f1_micro": torchmetrics.F1Score(num_classes=17, multiclass=False, average="micro"),
            "precision_macro": torchmetrics.Precision(num_classes=17, multiclass=False, average="macro"),
            "recall_macro": torchmetrics.Recall(num_classes=17, multiclass=False, average="macro"),
            "f1_macro": torchmetrics.F1Score(num_classes=17, multiclass=False, average="macro"),
        }
        self.values = None

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for metric in self.metrics.values():
            metric.update(preds, target)

    def compute(self):
        values = {metric_name: metric.compute() for metric_name, metric in self.metrics.items()}
        self.values = values

    def print(self):
        print("Metrics")
        print("--------")
        for metric_name, metric_value in self.values.items():
            # print metric values with 4 decimal places
            print(f"{metric_name}: {metric_value:.4f}")
