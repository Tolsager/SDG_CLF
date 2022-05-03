import torch
import numpy as np
import pandas as pd
import transformers
import os
import re
import datasets


def remove_with_regex(sample: dict, pattern: re.Pattern = None):
    """Deletes every match with the "pattern".
    Sample must have a 'text' feature.
    Is used for the 'map' dataset method

    Args:
        sample (dict): a huggingface dataset sample
        pattern (re.pattern): compiled regex pattern

    returns:
        sample_processed: sample with all regex matches removed
    """

    sample["text"] = pattern.subn("", sample["text"])[0]
    return sample


def preprocess(sample: dict, tokenizer: transformers.PreTrainedTokenizer):
    """preprocess a sample of the dataset

    Args:
        sample (dict): dataset sample
        tokenizer (transformers.PreTrainedTokenizer): a pretrained tokenizer

    Returns:
        dict: the preprocessed sample
    """
    sample["text"] = sample["text"].lower()

    # remove labels from the tweet
    sdg_prog = re.compile(r"#(sdg)s?(\s+)?(\d+)?")
    sample = remove_with_regex(sample, pattern=sdg_prog)

    # remove ekstra whitespace
    sample["text"] = " ".join(sample["text"].split())

    # create a label vector
    label = [int(sample[f"#sdg{i}"]) for i in range(1, 18)]
    sample["label"] = label

    # tokenize text
    encoding = tokenizer(
        sample["text"], max_length=260, padding="max_length", truncation=True
    )
    sample["input_ids"] = encoding.input_ids
    sample["attention_mask"] = encoding.attention_mask
    return sample


def load_dataset(
    file: str = "data/raw/allSDGtweets.csv",
    seed: int = 0,
    nrows: int = None,
    multi_class: bool = True,
    tokenizer_type: str = "roberta-base",
    split: bool = True,
):
    """Loads the tweet CSV into a huggingface dataset and apply the preprocessing

    Args:
        file (str, optional): path to csv file. Defaults to "data/raw/allSDGtweets.csv".
        seed (int, optional): seed used for shuffling. Defaults to 0.

    Returns:
        datasets.Dataset: a preprocessed dataset
    """
    # load the csv file into a huggingface dataset
    # Set the encodign to latin to be able to read special characters such as ñ
    tweet_df = pd.read_csv(file, encoding="latin", nrows=nrows)

    tweet_df = tweet_df.drop_duplicates("text")
    tweet_dataset = datasets.Dataset.from_pandas(tweet_df)
    if not multi_class:
        tweet_dataset = tweet_dataset.filter(lambda sample: sample["nclasses"] == 1)

    # remove unused columns
    tweet_dataset = tweet_dataset.remove_columns(
        ["Unnamed: 0", "id", "created_at", "category"]
    )
    print(
        f"Length of dataset before removing non-english tweets: {tweet_dataset.num_rows}"
    )

    # remove non-english text
    tweet_dataset = tweet_dataset.filter(lambda sample: sample["lang"] == "en")
    print(
        f"Length of dataset after removing non-english tweets: {tweet_dataset.num_rows}"
    )

    # apply the preprocessing function to every sample
    tokenizer_path = "tokenizers/" + tokenizer_type
    if not os.path.exists(tokenizer_path):
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_type)
        os.makedirs(tokenizer_path)
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_type)
    tweet_dataset = tweet_dataset.map(
        preprocess, num_proc=6, fn_kwargs={"tokenizer": tokenizer}
    )

    # remove redundant columns
    tweet_dataset = tweet_dataset.remove_columns(
        [f"#sdg{i}" for i in range(1, 18)] + ["lang"] + ["__index_level_0__"]
    )

    if split:
        tweet_dataset = tweet_dataset.shuffle(seed=seed)

        # tweet_dataset = tweet_dataset.cast_column("label", datasets.Sequence(datasets.Value("float32")))
        tweet_dataset = tweet_dataset.train_test_split(test_size=0.1)
    return tweet_dataset


def create_processed_dataset(
    path: str, tokenizer_type: str = "roberta-base", nrows: int = None
):

    tweet_dataset = load_dataset(
        path, split=False, tokenizer_type=tokenizer_type, nrows=nrows
    )
    tweet_dataset = tweet_dataset.shuffle(seed=0)
    splits = 10
    dataset_splits = {
        "train": datasets.concatenate_datasets(
            [tweet_dataset.shard(splits, i) for i in range(2, splits)]
        ),
        "validation": tweet_dataset.shard(splits, 0),
        "test": tweet_dataset.shard(splits, 1),
    }
    dataset_dict = datasets.DatasetDict(dataset_splits)
    save_path = "data/processed"
    save_path += f"/{tokenizer_type}"
    dataset_dict.save_to_disk(save_path)


def get_dataset(tokenizer_type: str, path_csv: str = "data/raw/allSDGtweets.csv"):
    """either loads the processed dataset if it exists and otherwise
    creates the dataset and saves it to disk.

    Args:
        tokenizer_type (str): a huggingface tokenizer type such as "roberta-base"
        path_csv (str, optional): path to raw tweet csv. Is only used if there is no processed dataset. Defaults to "data/raw/allSDGtweets.csv".

    Returns:
        datasets.DatasetDict: the processed dataset
    """
    path_ds = f"data/processed/{tokenizer_type}"
    if not os.path.exists(path_ds):
        create_processed_dataset(path_csv, tokenizer_type=tokenizer_type)
    ds = datasets.load_from_disk(path_ds)
    return ds


if __name__ == "__main__":
    # tweet_dataset = load_dataset(nrows=10)
    # tweet_dataset = load_dataset(nrows=10, multi_class=False)
    # tweet_dataset = load_dataset()
    # tweet_dataset.set_format("torch", columns=["input_ids", "label", "attention_mask"])
    # print(tweet_dataset)
    # print(type(tweet_dataset['train'][0]["input_ids"]))
    # create_processed_dataset("data/allSDGtweets.csv", nrows=20)
    # ds = datasets.load_from_disk("sodif")
    ds_dict = get_dataset("roberta-base")
    print()
    # print
