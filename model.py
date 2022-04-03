from transformers import RobertaForSequenceClassification


def get_model(num_labels=17, pretrained_path: str=None):
    if pretrained_path is None:
        pretrained_path = 'roberta-base'
    model = RobertaForSequenceClassification.from_pretrained(pretrained_path, num_labels=num_labels)
    return model
