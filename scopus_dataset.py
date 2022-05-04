import torch
import pandas as pd
import transformers
import os
import re
import datasets

from sdg_clf.tweet_dataset import preprocess

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


def load_abstracts(
    file: str = "data/raw/scopus_ready_to_use.csv",
    seed: int = 0,
    nrows: int = None,
    multi_label: bool = True,
    tokenizer_type: str = "roberta-base",
):
    """Loads the abstracts CSV into a huggingface dataset and apply the preprocessing

    Args:
        file (str, optional): path to csv file. Defaults to "data/raw/allSDGtweets.csv".
        seed (int, optional): seed used for shuffling. Defaults to 0.

    Returns:
        datasets.Dataset: a preprocessed dataset
    """
    # load the csv file into a huggingface dataset
    # Set the encodign to latin to be able to read special characters such as ñ
    abstracts_df = pd.read_csv(file, encoding="latin", nrows=nrows)

    abstracts_df = abstracts_df.drop_duplicates("text")
    abstracts_dataset = datasets.Dataset.from_pandas(abstracts_df)
    if not multi_label:
        abstracts_dataset = abstracts_dataset.filter(
            lambda sample: sample["nclasses"] == 1
        )

    # remove unused columns
    abstracts_dataset = abstracts_dataset.remove_columns(
        ["Title", "Year", "Link", "Author.Keywords", "Index.Keywords", "EID", "text"]
    )

    # Chunking into
    print(f"Length of dataset before chunking: {tweet_dataset.num_rows}")

    # apply the preprocessing function to every sample
    tokenizer_path = "tokenizers/" + tokenizer_type.replace("-", "_")
    if not os.path.exists(tokenizer_path):
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_type)
        os.makedirs(tokenizer_path)
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_type)
    tweet_dataset = tweet_dataset.map(
        preprocess,
        num_proc=6,
        fn_kwargs={"tokenizer": tokenizer, "tweet": False, "textname": "Abstract"},
    )

    # UPDATE LATER?
    tweet_dataset = tweet_dataset.shuffle(seed=seed)

    # tweet_dataset = tweet_dataset.cast_column("label", datasets.Sequence(datasets.Value("float32")))
    tweet_dataset = tweet_dataset.train_test_split(test_size=0.1)
    return tweet_dataset


if __name__ == "__main__":
    abstracts_dataset = load_abstracts(nrows=10)
    # tweet_dataset = load_dataset(nrows=10, multi_label=False)
    # tweet_dataset = load_dataset()
    abstracts_dataset.set_format(
        "torch", columns=["input_ids", "label", "attention_mask"]
    )
    # print(tweet_dataset)
    print(type(abstracts_dataset["train"][0]["input_ids"]))
    print()
