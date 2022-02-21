from transformers import RobertaForSequenceClassification


def get_model(num_labels=17):
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
    return model
