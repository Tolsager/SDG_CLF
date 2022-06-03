# Hugging Face transformers packages
# Why do we get the RobertaForSequenceClassification model, if we don't use it?
from transformers import (
    AutoModelForSequenceClassification,
    RobertaForSequenceClassification,
    AutoModel,  
)

# Why are we installing these?!
import torch.nn as nn
import transformers

class Model(nn.Module):
    def __init__(self, path, n_layers=0, n_labels=17, hidden_size=512, batch_size=8, transformer_hidden_size=768):
        super().__init__()
        assert n_layers >= 0


        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.path = path

        self.transformer = AutoModel.from_pretrained(
            self.path,
        )
        self.transformer_o = nn.Sequential(nn.Linear(batch_size*transformer_hidden_size, self.hidden_size),
                                           nn.ReLU(),
                                           )

        self.linear = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.ReLU(),
                                    )

        self.block = nn.ModuleList([self.transformer_o] + [self.linear for _ in range(n_layers)]) if n_layers > 0 else None

        self.head = nn.Linear(self.hidden_size, self.n_labels) if n_layers > 0 else nn.Linear(transformer_hidden_size*batch_size, self.n_labels)
    
    def forward(self, iids, amask):
        transformer_features = self.transformer(input_ids=iids, attention_mask=amask)[1]
        transformer_features = transformer_features[:,0,:].view(-1, 768)
        linear_features = self.block(transformer_features) if self.block is not None else transformer_features

        pred = self.head(linear_features)

        return pred



def get_model(
    num_labels: int =17,
    pretrained_path: str = None,
    n_layers: int = 3,
    hidden_size: int = 768,
    batch_size: int = 8,
    transformer_hidden_size: int = 768,
):
    """Function calling the from_pretrained method on Huggingface hosted pretrained models

    Args:
        num_labels (int, optional): The number of labels to be classified. Defaults to 17.
        pretrained_path (str, optional): String that mentions the model id to be retrieved. Defaults to None.

    Returns:
        _type_: _description_
    """
    model = Model(path=pretrained_path, n_layers=n_layers, n_labels=num_labels, hidden_size=hidden_size, batch_size=batch_size, transformer_hidden_size=transformer_hidden_size)
    return model


"""
Make new model inheriting from AutoModelForSequenceClassification
model.forward_scopus(input_ids):
    split input ids into chunks
    prepend chunk with [CLS]
    append chunk with [EOS]
    pad chunk

    for every chunk:
        preds.append(model(chunk))
    
    combine prediction and output final predictions
"""
