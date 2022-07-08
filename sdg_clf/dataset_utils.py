import pandas as pd
import datasets
import transformers
import os
import re
import numpy as np
import torch

from .utils import get_tokenizer


def remove_with_regex(sample: dict, pattern: re.Pattern = None):
    """
    Deletes every match with the "pattern".
    Sample must have a 'text' feature.
    Is used for the 'map' dataset method

    Args:
        sample (dict): a huggingface dataset sample
        pattern (re.pattern, optional): compiled regex pattern. Defaults to None
        textname (str, optional): the column name of the text column. Defaults to "text"

    returns:
        sample_processed: sample with all regex matches removed
    """

    sample["text"] = pattern.subn("", sample["text"])[0]
    return sample


def preprocess_sample(
        sample: dict,
        tweet: bool = True,
):
    """
    Preprocess a sample of the dataset by editing the text and collecting the labels in a list

    Args:
        sample (dict): dataset sample
        tweet (bool, optional): whether the data are tweets or abstracts. Defaults to True

    Returns:
        dict: the preprocessed sample
    """
    sample["text"] = sample["text"].lower()

    # remove labels from the tweet
    sdg_prog1 = re.compile(r"#(?:sdg)s?(\s+)?(\d+)?")
    sample = remove_with_regex(sample, pattern=sdg_prog1)
    sdg_prog2 = re.compile(r"(?:sdg)s?(\s?)(\d+)?")
    sample = remove_with_regex(sample, pattern=sdg_prog2)
    sdg_prog3 = re.compile(r"(sustainable development goals?\s?)(\d+)?")
    sample = remove_with_regex(sample, pattern=sdg_prog3)
    sdg_prog4 = re.compile(r"© \d\d(\d?)\d")
    sample = remove_with_regex(sample, pattern=sdg_prog4)
    sdg_prog5 = re.compile(r"elsevier\s+Ltd")
    sample = remove_with_regex(sample, pattern=sdg_prog5)

    # remove extra whitespace
    sample["text"] = " ".join(sample["text"].split())

    # create a label vector (only applicable for tweets)
    label_name = "sdg"
    if tweet:
        label_name = "#" + label_name
    label = [int(sample[f"{label_name}{i}"]) for i in range(1, 18)]
    sample["label"] = label

    return sample


def preprocess_dataset(
        file: str = "data/raw/allSDGtweets.csv",
        nrows: int = None,
        multi_label: bool = True,
        tweet: bool = True,
):
    """
    Loads the tweet CSV into a huggingface dataset and apply the preprocessing

    Args:
        file (str, optional): path to csv file. Defaults to "data/raw/allSDGtweets.csv".
        nrows (int, optional): only used for debugging the preprocessing. Defaults to None
        multi_label (bool, optional): if true only load samples with nclasses==1. Defaults to True
        tweet (bool, optional): whether the data are tweets or abstracts. Defaults to True

    Returns:
        datasets.Dataset: a preprocessed dataset
    """
    # load the csv file into a huggingface dataset
    # Set the encoding to latin to be able to read special characters such as ñ
    df = pd.read_csv(file, encoding="latin", nrows=nrows)

    if not tweet:
        df = df.drop(columns=["text"])
        df.rename(columns={"Abstract": "text"}, inplace=True)
    df = df.drop_duplicates("text")
    ds = datasets.Dataset.from_pandas(df)
    if not multi_label:
        ds = ds.filter(lambda sample: sample["nclasses"] == 1)

    # remove non-english text
    if tweet:
        ds = ds.filter(lambda sample: sample["lang"] == "en")
    ds = ds.map(
        preprocess_sample, num_proc=1, fn_kwargs={"tweet": tweet}
    )

    # remove redundant columns
    if tweet:
        ds = ds.remove_columns(
            ["Unnamed: 0", "id", "created_at", "category", "__index_level_0__", "lang"] + [f"#sdg{i}" for i in
                                                                                           range(1, 18)]
        )
    else:
        ds = ds.remove_columns(
            [
                "Unnamed: 0",
                "Title",
                "Year",
                "Link",
                "Author.Keywords",
                "Index.Keywords",
                "EID",
                "__index_level_0__"
            ] + [f"sdg{i}" for i in range(1, 18)]
        )

    return ds


def osdg_mapping(sample: dict):
    sample["text"] = sample["text"].lower()

    # remove labels from the tweet
    sdg_prog1 = re.compile(r"#(?:sdg)s?(\s+)?(\d+)?")
    sample = remove_with_regex(sample, pattern=sdg_prog1)
    sdg_prog2 = re.compile(r"(?:sdg)s?(\s?)(\d+)?")
    sample = remove_with_regex(sample, pattern=sdg_prog2)
    sdg_prog3 = re.compile(r"(sustainable development goals?\s?)(\d+)?")
    sample = remove_with_regex(sample, pattern=sdg_prog3)
    sdg_prog4 = re.compile(r"© \d\d(\d?)\d")
    sample = remove_with_regex(sample, pattern=sdg_prog4)
    sdg_prog5 = re.compile(r"elsevier\s+Ltd")
    sample = remove_with_regex(sample, pattern=sdg_prog5)

    # remove extra whitespace
    sample["text"] = " ".join(sample["text"].split())

    # create a label vector (only applicable for tweets)
    label = [0] * 17
    label[sample["sdg"] - 1] = 1
    sample["label"] = label

    return sample


def preprocess_osdg():
    # load the csv file into a huggingface dataset
    # Set the encoding to latin to be able to read special characters such as ñ
    df = pd.read_csv("data/raw/osdg.csv", encoding="latin", delimiter="\t")

    df = df.drop_duplicates("text")
    ds = datasets.Dataset.from_pandas(df)
    # remove non-english text
    ds = ds.filter(lambda sample: sample["labels_positive"] > sample["labels_negative"] and sample["agreement"] > 0.5)
    ds = ds.map(
        osdg_mapping, num_proc=1)

    # remove redundant columns
    ds = ds.remove_columns(
        [
            "doi",
            "text_id",
            "sdg",
            "labels_positive",
            "labels_negative",
            "agreement", ]
    )
    return ds


def split_dataset(
        ds: datasets.Dataset, tweet: bool = True
):
    """
    Splits the huggingface dataset into a test, training and validation set for tweets and a test and validation set for Scopus Abtracts

    Args:
        ds (datasets.Dataset): a dataset.
        tweet (bool, optional): whether the data are tweets or abstracts. Defaults to True

    Returns:
        dict of datasets.Dataset: Dictionary of the splitted dataset
    """
    ds = ds.shuffle(seed=0)
    if tweet:
        splits = 10
        dataset_splits = {
            "train": datasets.concatenate_datasets(
                [ds.shard(splits, i) for i in range(2, splits)]
            ),
            "validation": ds.shard(splits, 0),
            "test": ds.shard(splits, 1),
        }
        dataset_dict = datasets.DatasetDict(dataset_splits)
    else:
        dataset_dict = ds.train_test_split(test_size=0.5, shuffle=False)

    return dataset_dict


def create_base_dataset(tweet: bool = True, nrows: int = None):
    """
    Preprocesses the text of the two datasets

    Args:
        tweet (bool, optional): whether the data are tweets or abstracts. Defaults to True
        nrows (int, optional): only used for debugging. Defaults to None
        path_data (str, optional): path to data directory. Defaults to "data"

    Returns:
        None

    """
    path_data = "data"
    path = os.path.join(path_data, "raw")
    if tweet:
        path = os.path.join(path, "twitter.csv")
    else:
        path = os.path.join(path, "scopus.csv")

    ds = preprocess_dataset(file=path, nrows=nrows, tweet=tweet)
    ds_dict = split_dataset(ds, tweet=tweet)
    path_save = os.path.join(path_data, "processed")
    if tweet:
        path_save = os.path.join(path_save, "twitter/base")
    else:
        path_save = os.path.join(path_save, "scopus/base")
    ds_dict.save_to_disk(path_save)


def tokenize_dataset(tokenizer: transformers.PreTrainedTokenizer, tweet: bool = True, max_length: int = 260,
                     ):
    """
    Tokenizes the dataset

    Args:
        tokenizer (transformers.PreTrainedTokenizer): an instantiated huggingface tokenizer
        tweet (bool, optional): tweet dataset or scopus. Defaults to True
        max_length (int, optional): maximum token length used during training. Defaults to 260
        path_data (str, optional): path to data directory. Defaults to "data"

    Returns:
        dataset dict with the output from the tokenizer

    """
    path_data = "data/processed"
    if tweet:
        ds_dict = datasets.load_from_disk(os.path.join(path_data, "twitter/base"))
    else:
        ds_dict = datasets.load_from_disk(os.path.join(path_data, "scopus/base"))

    for split in ds_dict.keys():
        if tweet:
            ds_dict[split] = ds_dict[split].map(
                lambda samples: tokenizer(samples["text"], padding="max_length", max_length=max_length,
                                          truncation=True, return_token_type_ids=False), batched=True, num_proc=1)
        else:
            ds_dict[split] = ds_dict[split].map(
                lambda samples: tokenizer(samples["text"], add_special_tokens=False, truncation=False,
                                          return_attention_mask=False, return_token_type_ids=False), num_proc=1)
        ds_dict[split] = ds_dict[split].remove_columns(["text", "nclasses", "label"])

    return ds_dict


def get_dataset(tokenizer_type: str, tweet: bool = True, sample_data: bool = False, max_length: int = 260,
                overwrite: bool = False):
    """
    Creates the dataset if it doesn't already exist otherwise it loads it from the correct folder

    Args:
        tokenizer_type (str): name of a huggingface tokenizer e.g. bert-base-uncased
        tweet (bool, optional): if scopus or tweet dataset. Defaults to True
        sample_data (bool, optional): if true only selects 20 samples. Used for debugging. Defaults to False
        max_length (int, optional): max token length used for training the transformer. Defaults to 260
        overwrite (bool, optional): if an existing dataset should be overwritten. Defaults to False
        path_data (str, optional): path to data directory. Defaults to "data"
        path_tokenizers (str, optional): path to tokenizer. Defaults to "tokenizers"

    Returns:
        processed dataset for training with transformer outputs and labels
    """
    # load tokenized dataset if exists
    path_data = "data/processed"
    if tweet:
        path_data = os.path.join(path_data, "twitter")
    else:
        path_data = os.path.join(path_data, "scopus")
    path_base = os.path.join(path_data, "base")

    path_ds_dict_tokens = os.path.join(path_data, tokenizer_type)
    if os.path.exists(path_ds_dict_tokens) and not overwrite:
        ds_dict_tokens = datasets.load_from_disk(path_ds_dict_tokens)
        ds_dict_base = datasets.load_from_disk(path_base)
        for split in ds_dict_base.keys():
            ds_dict_tokens[split] = ds_dict_tokens[split].add_column("label", ds_dict_base[split]["label"])
            if sample_data:
                ds_dict_tokens[split] = ds_dict_tokens[split].select(np.arange(20))
        return ds_dict_tokens

    # else create dataset
    if not os.path.exists(path_base):
        create_base_dataset(tweet=tweet)
    ds_dict_base = datasets.load_from_disk(path_base)

    tokenizer = get_tokenizer(tokenizer_type)
    ds_dict_tokens = tokenize_dataset(tokenizer, tweet=tweet, max_length=max_length)
    path_save = path_data + f"/{tokenizer_type}"
    ds_dict_tokens.save_to_disk(path_save)

    for split in ds_dict_base.keys():
        ds_dict_tokens[split] = ds_dict_tokens[split].add_column("label", ds_dict_base[split]["label"])
        if sample_data:
            ds_dict_tokens[split] = ds_dict_tokens[split].select(np.arange(20))
    return ds_dict_tokens


def load_ds_dict(tokenizer_type: str, tweet: bool = True):
    """
    Load an existing dataset from the files

        Args:
            tokenizer_type (str): name of a huggingface tokenizer e.g. bert-base-uncased
            tweet (bool, optional): whether the data are tweets or abstracts. Defaults to True

        Returns:
            dict of datasets.Dataset: Dictionary of the splitted dataset
        """
    path_data = "data"
    path_ds = os.path.join(path_data, "processed", "twitter" if tweet else "scopus")
    path_ds_dict_tokens = os.path.join(path_ds, tokenizer_type)
    ds_dict_tokens = datasets.load_from_disk(path_ds_dict_tokens)

    path_ds_dict_base = os.path.join(path_ds, "base")
    ds_dict_base = datasets.load_from_disk(path_ds_dict_base)

    for split in ds_dict_base.keys():
        ds_dict_tokens[split] = ds_dict_tokens[split].add_column("label", ds_dict_base[split]["label"])

    return ds_dict_tokens


def get_dataloader(split: str, tokenizer_type, batch_size: int = 20, tweet: bool = True, n_samples: int = None):
    ds_dict = get_dataset(tokenizer_type, tweet=tweet)
    ds = ds_dict[split][:n_samples]
    if tweet:
        ds.set_format("pt", ["input_ids", "attention_mask"])
    else:
        ds.set_format("pt", ["input_ids"])
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    return dl


def load_preprocessed_dataset(dataset_name: str, split: str) -> datasets.Dataset:
    return datasets.load_from_disk(f"data/processed/{dataset_name}/base")[split]


if __name__ == "__main__":
    ds_dict_tweets = get_dataset("roberta-base")
    ds_dict_scopus = get_dataset("roberta-base", tweet=False)
