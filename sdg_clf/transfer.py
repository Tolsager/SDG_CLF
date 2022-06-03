"""Ideas for splitting the Scopus data into smaller sentences
    1. Use the average (or max) length of the tweets and split regardless of sentence ending
    2. Use the max character length of a tweet and round down to work with full words (280 characters)
    3. 


Process:
    1. Tokenize
    2. 

model.forward_scopus(input_ids):
    split input ids into chunks
    prepend chunk with [CLS]
    append chunk with [EOS]
    pad chunk

    for every chunk:
        preds.append(model(chunk))
    
    combine prediction and output final predictions
"""
import pandas as pd
import transformers


def align_labels_with_tokens(labels, chunk_ids):
    new_labels = []
    current_chunk = None
    for chunk_id in chunk_ids:
        if chunk_id != current_chunk:
            # Start of a new chunk!
            current_chunk = chunk_id
            label = -100 if chunk_id is None else labels[chunk_id]
            new_labels.append(label)
        elif chunk_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same chunk as previous token
            label = labels[chunk_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def forward_longtext(
    model,
    tokenized_data: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    input_ids: list = None,
    max_length: int = 260,
):
    for id in input_ids:
        for i in range(len(tokenized_data["input_ids"][1:-1]) // max_length + 1):
            tokenizer.pad(
                {
                    "input_ids": tokenized_data["input_ids"][
                        1 + i * max_length * (i + 1) - 1
                    ],
                    "attention_mask": tokenized_data["attenteion_mask"][
                        1 + i * max_length * (i + 1) - 1, max_length * (i + 1) - 1
                    ],
                },
                max_length=max_length,
                padding="max_lenght",
            )


tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)
