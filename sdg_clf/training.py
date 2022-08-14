import pytorch_lightning as pl
import torch
import torchmetrics
import transformers

from sdg_clf import base


class LitSDG(pl.LightningModule):
    def __init__(self, model: transformers.AutoModelForSequenceClassification, hparams: base.HParams):
        super().__init__()
        self.model = model
        self.HParams = hparams
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.sigmoid = torch.nn.Sigmoid()
        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(subset_accuracy=True, num_classes=17, multiclass=False)
        self.train_micro_precision = torchmetrics.Precision(num_classes=17, multiclass=False, average="micro")
        self.train_micro_recall = torchmetrics.Recall(num_classes=17, multiclass=False, average="micro")
        self.train_micro_f1 = torchmetrics.F1Score(num_classes=17, multiclass=False, average="micro")
        self.train_macro_precision = torchmetrics.Precision(num_classes=17, multiclass=False, average="macro")
        self.train_macro_recall = torchmetrics.Recall(num_classes=17, multiclass=False, average="macro")
        self.train_macro_f1 = torchmetrics.F1Score(num_classes=17, multiclass=False, average="macro")
        self.val_accuracy = torchmetrics.Accuracy(subset_accuracy=True, num_classes=17, multiclass=False)
        self.val_micro_precision = torchmetrics.Precision(num_classes=17, multiclass=False, average="micro")
        self.val_micro_recall = torchmetrics.Recall(num_classes=17, multiclass=False, average="micro")
        self.val_micro_f1 = torchmetrics.F1Score(num_classes=17, multiclass=False, average="micro")
        self.val_macro_precision = torchmetrics.Precision(num_classes=17, multiclass=False, average="macro")
        self.val_macro_recall = torchmetrics.Recall(num_classes=17, multiclass=False, average="macro")
        self.val_macro_f1 = torchmetrics.F1Score(num_classes=17, multiclass=False, average="macro")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        model_outputs = self(input_ids, attention_mask=attention_mask)
        loss = self.criterion(model_outputs, labels.float())
        preds = self.sigmoid(model_outputs)
        self.train_accuracy(preds, labels)
        self.train_micro_precision(preds, labels)
        self.train_micro_recall(preds, labels)
        self.train_micro_f1(preds, labels)
        self.train_macro_precision(preds, labels)
        self.train_macro_recall(preds, labels)
        self.train_macro_f1(preds, labels)
        metrics_dict = {"train_loss": loss, "train_accuracy": self.train_accuracy,
                        "train_micro_precision": self.train_micro_precision,
                        "train_micro_recall": self.train_micro_recall,
                        "train_micro_f1": self.train_micro_f1, "train_macro_precision": self.train_macro_precision,
                        "train_macro_recall": self.train_macro_recall, "train_macro_f1": self.train_macro_f1}
        self.log_dict(metrics_dict)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        model_outputs = self(input_ids, attention_mask=attention_mask)
        loss = self.criterion(model_outputs, labels.float())
        preds = self.sigmoid(model_outputs)
        self.val_accuracy(preds, labels)
        self.val_micro_precision(preds, labels)
        self.val_micro_recall(preds, labels)
        self.val_micro_f1(preds, labels)
        self.val_macro_precision(preds, labels)
        self.val_macro_recall(preds, labels)
        self.val_macro_f1(preds, labels)
        metrics_dict = {"val_loss": loss, "val_accuracy": self.val_accuracy, "val_micro_precision": self.val_micro_precision,
                        "val_micro_recall": self.val_micro_recall,
                        "val_micro_f1": self.val_micro_f1, "val_macro_precision": self.val_macro_precision,
                        "val_macro_recall": self.val_macro_recall, "val_macro_f1": self.val_macro_f1}
        self.log_dict(metrics_dict)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.HParams.lr, weight_decay=self.HParams.weight_decay)
