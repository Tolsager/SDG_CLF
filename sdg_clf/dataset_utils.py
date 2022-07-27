import pandas as pd
import datasets
import transformers
import os
import re
import numpy as np
import torch

from sdg_clf.utils import get_tokenizer
from typing import Union


def process_scopus_df() -> pd.DataFrame:
    """
    process the scopus dataset
    """
    df = pd.read_csv("data/raw/scopus_ready_to_use.csv")

    # lower Abstract
    df["Abstract"] = df["Abstract"].str.lower()

    # remove extra whitespace in the "Abstract" column
    df["Abstract"] = df["Abstract"].str.replace(r"\s+", " ")

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
    df_clean = df[[f"sdg{i}" for i in range(1, 18)]].astype(int).copy()
    df_clean["text"] = df["Abstract"]

    df_clean.reset_index(drop=True, inplace=True)
    return df_clean


def process_twitter_df() -> pd.DataFrame:
    """
    process the twitter dataset
    """
    df = pd.read_csv("data/raw/allSDGtweets.csv", index_col=0, encoding="latin")

    # lower text
    df["text"] = df["text"].str.lower()

    # remove duplicate text
    df = df.drop_duplicates("text")

    # remove rows with nans in any column
    df = df.dropna(axis=0, how="any")

    # remove extra whitespace in the "text" column
    df["text"] = df["text"].str.replace(r"\s+", " ")

    # remove non english rows
    df = df[df["lang"] == "en"]

    # remove the hashtags from the #sdg columns
    df.rename(columns={f"#sdg{i}": f"sdg{i}" for i in range(1, 18)}, inplace=True)

    df_clean = df[[f"sdg{i}" for i in range(1, 18)]].astype(int).copy()
    df_clean["text"] = df["text"]
    # reset index
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean


def process_osdg_df() -> pd.DataFrame:
    """
    process the osdg dataset from https://zenodo.org/record/6831287
    """
    df = pd.read_csv("data/raw/osdg-community-data-v2022-07-01.csv", encoding="latin", delimiter="\t")
    df["text"] = df["text"].str.lower()
    # remove duplicate text
    df = df.drop_duplicates("text")

    # remove extra whitespace in the "text" column
    df["text"] = df["text"].str.replace(r"\s+", " ")

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

    df_clean = df[[f"sdg{i}" for i in range(1, 18)]].astype(int).copy()
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


def process_text(text: str) -> str:
    """
    process the text by removing labels and noise.

    Args:
        text: text to process

    Returns:
        processed text
    """
    # lower text
    text = text.lower()

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


def process_df(dataset_name: str = "twitter") -> pd.DataFrame:
    """
    process the dataframe and creates the new column "processed_text". This might contain nans
    Args:
        dataset_name:  {twitter, scopus, osdg}

    Returns:
    processed dataframe

    """
    if dataset_name == "twitter":
        df = process_twitter_df()
    elif dataset_name == "scopus":
        df = process_scopus_df()
    elif dataset_name == "osdg":
        df = process_osdg_df()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    df = df.sample(frac=1, random_state=42)
    df["processed_text"] = df["text"].apply(process_text)
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
    df = process_df(dataset_name)
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


def process_dataset(dataset: datasets.Dataset, tokenizer: transformers.PreTrainedTokenizer, max_length: int = 260):
    """
    Creates the label tensor from the SDG columns. Tokenizes the dataset and removes all columns but label and label
    and model inputs.
    This is an efficient way of storing the data during training.

    Args:
        dataset: dataset with the columns ["text", "processed_text"] and the SDG columns
        tokenizer (transformers.PreTrainedTokenizer): an instantiated huggingface tokenizer
        max_length (int, optional): maximum token length used during training. Defaults to 260

    Returns:
        dataset with the output from the tokenizer along with the processed label

    """

    def processing_func(sample):
        text = sample["processed_text"]
        # tokenize text
        tokenizer_output = tokenizer(text, max_length=max_length, padding="max_length", truncation=True)

        label = [sample[f"sdg{i}"] for i in range(1, 18)]
        return {"input_ids": tokenizer_output["input_ids"], "attention_mask": tokenizer_output["attention_mask"],
                "label": label}

    ds = dataset.map(processing_func, batched=False, num_proc=1)

    # remove unused columns
    ds = ds.remove_columns(["processed_text"] + [f"sdg{i}" for i in range(1, 18)])
    return ds


def get_processed_ds(dataset_name: str, model_type: str, split: str) -> datasets.Dataset:
    """
    Get the processed dataset.
    Args:
        dataset_name:  {twitter, scopus, osdg}
        model_type: a model from the Huggingface Hub with an AutoModelForSequenceClassification class
        split: data split to use

    Returns:
        dataset
    """
    ds_load_path = f"data/processed/{dataset_name}/{split}/{model_type}"
    # if the dataset already exists, load it
    if os.path.exists(ds_load_path):
        ds = datasets.load_from_disk(ds_load_path)
    else:
        df = get_processed_df(dataset_name, split)
        # remove columns with nans in "processed_text" as we can't train on these and the dataset is only used
        # during training.
        df = df.dropna()
        features = datasets.Features(
            {f"sdg{i}": datasets.Value("int8") for i in range(1, 18)} | {"processed_text": datasets.Value("string")})
        ds = datasets.Dataset.from_pandas(df, features=features)
        # ds = datasets.Dataset.from_pandas(df)
        tokenizer = get_tokenizer(model_type)
        ds = process_dataset(ds, tokenizer)
        ds.save_to_disk(ds_load_path)
    return ds


def get_dataloader(dataset_name: str, model_type: str, split: str, batch_size: int = 20):
    ds = get_processed_ds(dataset_name, model_type, split)
    ds.set_format("pt")
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    return dl


def load_processed_dataset(dataset_name: str, split: str) -> datasets.Dataset:
    return datasets.load_from_disk(f"data/processed/{dataset_name}/base")[split]
