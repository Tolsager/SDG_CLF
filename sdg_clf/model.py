# Hugging Face transformers packages
# Why do we get the RobertaForSequenceClassification model, if we don't use it?
import os
import torch

from transformers import (
    AutoModelForSequenceClassification,
    RobertaForSequenceClassification,
    AutoModel,
)

# Why are we installing these?!
import torch.nn as nn
import transformers


class Model(nn.Module):
    def __init__(self, path, n_layers=0, n_labels=17, hidden_size=768):
        super().__init__()
        assert n_layers >= 0

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.path = path
        self.n_labels = n_labels

        self.activation = nn.Tanh()

        self.transformer = AutoModel.from_pretrained(
            self.path,
        )
        self.transformer_o = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                           self.activation,
                                           )

        self.linear = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                    self.activation,
                                    )

        self.dropout = nn.Dropout(p=0.1)

        self.block = nn.Sequential(
            *nn.ModuleList([self.transformer_o] + [self.linear for _ in range(n_layers)])) if n_layers >= 0 else None

        self.head = nn.Linear(self.hidden_size, self.n_labels)

    def forward(self, iids, amask):
        transformer_features = self.transformer(input_ids=iids, attention_mask=amask)["last_hidden_state"]
        x = transformer_features[:, 0, :]
        x = self.dropout(x)
        if self.block is not None:
            x = self.block(x)
            x = self.dropout(x)
        pred = self.head(x)
        out = {"logits": pred}
        return out


# def get_model(
#         num_labels: int = 17,
#         pretrained_path: str = None,
#         n_layers: int = 1,
#         hidden_size: int = 768,
# ):
#     """Function calling the from_pretrained method on Huggingface hosted pretrained models
#
#     Args:
#         num_labels (int, optional): The number of labels to be classified. Defaults to 17.
#         pretrained_path (str, optional): String that mentions the model id to be retrieved. Defaults to None.
#
#     Returns:
#         _type_: _description_
#     """
#     model = Model(path=pretrained_path, n_layers=n_layers, n_labels=num_labels, hidden_size=hidden_size)
#     return model


def load_model(weights: str, model_type: str):
    path_pretrained_model = os.path.join("pretrained_models", model_type)
    if os.path.exists(path_pretrained_model):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(path_pretrained_model, num_labels=17)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=17)
        model.save_pretrained("pretrained_models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load(os.path.join("finetuned_models", weights), map_location=device))
    return model
