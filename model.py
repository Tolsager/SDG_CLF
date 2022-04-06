from transformers import RobertaForSequenceClassification
import torch.nn as nn

def get_model(num_labels=17, pretrained_path: str=None, multi_class: bool = False):
    if pretrained_path is None:
        pretrained_path = 'roberta-base'
    if multi_class:
        class MultiClassModel(nn.Module):
            def __init__(self, pretrained_path: str = None, num_labels: int = 17):
                super().__init__()
                self.roberta = RobertaForSequenceClassification.from_pretrained(pretrained_path, num_labels=num_labels)
                self.sigmoid = nn.Sigmoid()

            def forward(self, tokens, attention_mask):
                x = self.roberta(tokens, attention_mask)['logits']
                return x
            
            def predict(self, x):
                return self.sigmoid(self(x))
            
        model = MultiClassModel(pretrained_path, num_labels)
    else:
        model = RobertaForSequenceClassification.from_pretrained(pretrained_path, num_labels=num_labels)
    return model
