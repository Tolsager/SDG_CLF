import pandas as pd
import datasets
import transformers
import os
import re
import numpy as np
import torch

from sdg_clf.utils import get_tokenizer
from typing import Union


def preprocess_scopus_df() -> pd.DataFrame:
    """
    Preprocess the scopus dataset
    """
    df = pd.read_csv("data/raw/scopus_ready_to_use.csv")
    # lower Abstract
    df["Abstract"] = df["Abstract"].str.lower()
    # remove duplicate abstracts
    df = df.drop_duplicates("Abstract")
    # remove rows with no abstract
    df = df[df["Abstract"] != "[no abstract available]"]
    # remove rows with nclasses of 0
    df = df[df["nclasses"] != 0]
    # remove rows with the Chinese archaeological shuidonggou which abbreviates to SDG
    df = df[~df["Abstract"].str.contains("shuidonggou")]
    # correct mislabelled sdg1s
    sdg1_pattern = re.compile(r"(sdg\s?1)(\D|$)")
    df["sdg1"] = df["Abstract"].apply(lambda x: True if sdg1_pattern.search(x) is not None else False)

    # add the sdg columns
    df_clean = df[[f"sdg{i}" for i in range(1, 18)]].copy()
    df_clean["text"] = df["Abstract"]

    df_clean.reset_index(drop=True, inplace=True)
    return df_clean


def preprocess_twitter_df() -> pd.DataFrame:
    """
    Preprocess the twitter dataset
    """
    df = pd.read_csv("data/raw/allSDGtweets.csv", index_col=0, encoding="latin")
    # lower text
    df["text"] = df["text"].str.lower()
    # remove duplicate text
    df = df.drop_duplicates("text")

    # remove non english rows
    df = df[df["lang"] == "en"]

    # remove the hashtags from the #sdg columns
    df.rename(columns={f"#sdg{i}": f"sdg{i}" for i in range(1, 18)}, inplace=True)

    df_clean = df[[f"sdg{i}" for i in range(1, 18)]].copy()
    df_clean["text"] = df["text"]
    # reset index
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean


def preprocess_osdg_df() -> pd.DataFrame:
    """
    Preprocess the osdg dataset from https://zenodo.org/record/6831287
    """
    df = pd.read_csv("data/raw/osdg-community-data-v2022-07-01.csv", encoding="latin", delimiter="\t")
    df["text"] = df["text"].str.lower()
    # remove duplicate text
    df = df.drop_duplicates("text")

    # remove rows where labels_negative is larger than labels_positive
    df = df[df["labels_negative"] <= df["labels_positive"]]

    # remove rows where agreement is less than 0.5
    df = df[df["agreement"] >= 0.5]

    def create_label(sdg: int) -> np.array:
        label = np.zeros(17)
        label[sdg - 1] = 1
        return label

    # create the label vector
    labels = []
    for i, row in df.iterrows():
        sdg = row["sdg"]
        label = create_label(sdg)
        labels.append(label)
    labels = np.stack(labels, axis=0)
    df[[f"sdg{i}" for i in range(1, 18)]] = labels

    df_clean = df[[f"sdg{i}" for i in range(1, 18)]].copy()
    df_clean["text"] = df["text"]

    # reset index
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean


def remove_with_regex(text: str, patterns: list[str]) -> str:
    """
    Remove text with regex patterns.
    Args:
        text: text to be edited
        patterns: patterns to indicate what to substitute

    Returns:


    """
    for pattern in patterns:
        text = re.sub(pattern, "", text)

    return text


def preprocess_text(text: str) -> str:
    """
    Preprocess the text by removing labels and noise.

    Args:
        text: text to preprocess

    Returns:
        preprocessed text
    """
    # lower text
    text = text.lower()
    # remove extra whitespace
    text = " ".join(text.split())

    # remove labels from the tweet
    sdg_prog1 = r"#(?:sdg)s?(\s+)?(\d+)?"
    sdg_prog2 = r"(?:sdg)s?(\s?)(\d+)?"
    sdg_prog3 = r"(sustainable development goals?\s?)(\d+)?"
    copyright_prog = r"© \d\d(\d?)\d"
    elsevier_prog = r"elsevier\s+Ltd"
    url_prog = r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
    patterns = [sdg_prog1, sdg_prog2, sdg_prog3, copyright_prog, elsevier_prog, url_prog]

    text = remove_with_regex(text, patterns)
    return text


def preprocess_df(dataset_name: str = "twitter") -> pd.DataFrame:
    """
    Preprocess the dataframe.
    Args:
        dataset_name:  {twitter, scopus, osdg}

    Returns:
    preprocessed dataframe

    """
    if dataset_name == "twitter":
        df = preprocess_twitter_df()
    elif dataset_name == "scopus":
        df = preprocess_scopus_df()
    elif dataset_name == "osdg":
        df = preprocess_osdg_df()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    df = df.sample(frac=1, random_state=42)
    df["preprocessed_text"] = df["text"].apply(preprocess_text)
    return df


def get_split_indices(n_elements: int, fractions: list[float]) -> list[tuple[int, int]]:
    """
    Get the indices of to split on
    Args:
        n_elements: the total number of samples
        fractions: fraction of n_elements to be in each split

    Returns:
        indices to split on
    """
    cum_fractions = np.cumsum(fractions)
    cum_fractions[-1] = 1
    upper_bounds = [int(np.round(n_elements * cum_fraction)) for cum_fraction in cum_fractions]
    lower_bounds = [0] + upper_bounds[:-1]
    split_indices = list(zip(lower_bounds, upper_bounds))
    return split_indices


def split_df(df: pd.DataFrame, fractions: list[float], split_names: list[str]) -> dict[str, pd.DataFrame]:
    """
    Split the dataframe into multiple dataframes.
    Args:
        df: dataframe to split
        fractions: fractions of the dataframe to split into
        split_names: names of the dataframes

    Returns:
        list of dataframes
    """
    if len(fractions) != len(split_names):
        raise ValueError("fractions and split_names must be the same length")

    split_indices = get_split_indices(len(df), fractions)
    split_dfs = {name: df.iloc[lower_bound:upper_bound] for name, (lower_bound, upper_bound) in
                 zip(split_names, split_indices)}

    return split_dfs


def create_processed_dfs(dataset_name: str, fractions: list[float], split_names: list[str]) -> dict[str, pd.DataFrame]:
    """
    Create the processed dataframe.
    Args:
        dataset_name:  {twitter, scopus, osdg}
        fractions: fractions of the dataframe to split into
        split_names: names of the splits

    Returns:
        dict of dataframes

    """
    df = preprocess_df(dataset_name)
    split_dfs = split_df(df, fractions, split_names)
    return split_dfs


def create_scopus_processed_dfs():
    fractions = [0.5, 0.5]
    split_names = ["val", "test"]
    return create_processed_dfs("scopus", fractions, split_names)


def create_twitter_processed_dfs():
    fractions = [0.8, 0.1, 0.1]
    split_names = ["train", "val", "test"]
    return create_processed_dfs("twitter", fractions, split_names)


def create_osdg_processed_dfs():
    fractions = [1]
    split_names = ["test"]
    return create_processed_dfs("osdg", fractions, split_names)


def save_processed_dfs(dataset_name: str, split_dfs: dict[str, pd.DataFrame]) -> None:
    """
    Save the processed dataframes.
    Args:
        dataset_name:  {twitter, scopus, osdg}
        split_dfs: dict of dataframes to save
    """
    save_dir = f"data/processed/{dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)
    for name, df in split_dfs.items():
        save_path = os.path.join(save_dir, f"{name}.csv")
        df.to_csv(save_path, index=False)


def load_processed_df(dataset_name: str, split_name: str) -> Union[pd.DataFrame, None]:
    """
    Load the processed dataframe.
    Args:
        dataset_name:  {twitter, scopus, osdg}
        split_name: name of the split

    Returns:
        dataframe if found, else None
    """
    load_dir = f"data/processed/{dataset_name}/"
    load_path = os.path.join(load_dir, f"{split_name}.csv")
    try:
        df = pd.read_csv(load_path)
    except FileNotFoundError:
        return None
    return df


def get_processed_df(dataset_name: str, split_name: str, overwrite: bool = False):
    """
    Get the processed dataframe.
    Args:
        dataset_name:  {twitter, scopus, osdg}
        split_name: name of the split
        overwrite: whether to overwrite the dataframe if it already exists

    Returns:
        dataframe
    """
    if overwrite:
        df = None
    else:
        df = load_processed_df(dataset_name, split_name)
    if df is None:
        if dataset_name == "twitter":
            dfs = create_twitter_processed_dfs()
        elif dataset_name == "scopus":
            dfs = create_scopus_processed_dfs()
        elif dataset_name == "osdg":
            dfs = create_osdg_processed_dfs()
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        # save the dataframes
        save_processed_dfs(dataset_name, dfs)
    df = load_processed_df(dataset_name, split_name)
    return df


def get_labels_tensor(df: pd.DataFrame) -> torch.Tensor:
    """
    Gets the labels tensor from a processed dataframe
    Args:
        df: processed dataframe

    Returns:
        labels tensor
    """
    # extract the sdgx columns
    labels = df[[f"sdg{i}" for i in range(1, 18)]].values
    labels_tensor = torch.tensor(labels)

    return labels_tensor


def get_base_dataset(dataset_name: str, split: str) -> datasets.Dataset:
    dataset_path = f"data/processed/{dataset_name}/base"
    if os.path.exists(dataset_path):
        return datasets.load_from_disk(dataset_path)[split]
    # else:
    #     if dataset_name == "scopus":
    #         df = preprocess_scopus()
    #     elif

    return datasets.load_from_disk(f"data/raw/{dataset_name}")[split]


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
