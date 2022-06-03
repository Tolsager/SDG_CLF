import pandas as pd
import datasets
import transformers
import os
import re
import numpy as np


def remove_with_regex(sample: dict, pattern: re.Pattern = None, textname: str = "text"):
    """Deletes every match with the "pattern".
    Sample must have a 'text' feature.
    Is used for the 'map' dataset method

    Args:
        sample (dict): a huggingface dataset sample
        pattern (re.pattern): compiled regex pattern

    returns:
        sample_processed: sample with all regex matches removed
    """

    sample[textname] = pattern.subn("", sample[textname])[0]
    return sample


def preprocess_sample(
        sample: dict,
        tweet: bool = True,
):
    """preprocess a sample of the dataset

    Args:
        sample (dict): dataset sample
        tokenizer (transformers.PreTrainedTokenizer): a pretrained tokenizer
        tweet (bool): whether the data are tweets or abstracts

    Returns:
        dict: the preprocessed sample
    """
    if tweet:
        textname = 'text'
    else:
        textname = 'Abstract'

    sample[textname] = sample[textname].lower()

    # remove labels from the tweet
    sdg_prog1 = re.compile(r"#(?:sdg)s?(\s+)?(\d+)?")
    sample = remove_with_regex(sample, pattern=sdg_prog1, textname=textname)
    sdg_prog2 = re.compile(r"(?:sdg)s?(\s?)(\d+)?")
    sample = remove_with_regex(sample, pattern=sdg_prog2, textname=textname)
    sdg_prog3 = re.compile(r"(sustainable development goals?\s?)(\d+)?")
    sample = remove_with_regex(sample, pattern=sdg_prog3, textname=textname)
    sdg_prog4 = re.compile(r"© \d\d(\d?)\d")
    sample = remove_with_regex(sample, pattern=sdg_prog4, textname=textname)
    sdg_prog5 = re.compile(r"elsevier\s+Ltd")
    sample = remove_with_regex(sample, pattern=sdg_prog5, textname=textname)

    # remove ekstra whitespace
    sample[textname] = " ".join(sample[textname].split())

    # create a label vector (only applicable for tweets)
    label_name = "sdg"
    if tweet:
        label_name = "#" + label_name
    label = [int(sample[f"{label_name}{i}"]) for i in range(1, 18)]
    sample["label"] = label

    # # tokenize text
    # encoding = tokenizer(
    #     sample[textname], max_length=260, padding="max_length", truncation=True
    # )
    # sample["input_ids"] = encoding.input_ids
    # sample["attention_mask"] = encoding.attention_mask
    return sample


def preprocess_dataset(
        file: str = "data/raw/allSDGtweets.csv",
        nrows: int = None,
        multi_label: bool = True,
        tweet: bool = True,
):
    """Loads the tweet CSV into a huggingface dataset and apply the preprocessing

    Args:
        file (str, optional): path to csv file. Defaults to "data/raw/allSDGtweets.csv".
        seed (int, optional): seed used for shuffling. Defaults to 0.
        tweet (bool): whether the data are tweets or abstracts

    Returns:
        datasets.Dataset: a preprocessed dataset
    """
    # load the csv file into a huggingface dataset
    # Set the encoding to latin to be able to read special characters such as ñ
    df = pd.read_csv(file, encoding="latin", nrows=nrows)

    if tweet:
        textname = 'text'
    else:
        textname = 'Abstract'
    df = df.drop_duplicates(textname)
    ds = datasets.Dataset.from_pandas(df)
    if not multi_label:
        ds = ds.filter(lambda sample: sample["nclasses"] == 1)

    # remove non-english text
    if tweet:
        ds = ds.filter(lambda sample: sample["lang"] == "en")
    ds = ds.map(
        preprocess_sample, num_proc=6, fn_kwargs={"tweet": tweet}
    )

    # remove redundant columns
    if tweet:
        ds = ds.remove_columns(
            ["Unnamed: 0", "id", "created_at", "category", "__index_level_0__", "lang"] + [f"#sdg{i}" for i in range(1, 18)]
        )
    else:
        # remove unused columns
        ds = ds.remove_columns(
            [
                "Unnamed: 0",
                "Title",
                "Year",
                "Link",
                "Author.Keywords",
                "Index.Keywords",
                "EID",
                "text",
                "__index_level_0__"
            ]
        )
    # print(
    #     f"Length of dataset before removing non-english tweets: {ds.num_rows}"
    # )

    # remove non-english text
    if tweet:
        ds = ds.filter(lambda sample: sample["lang"] == "en")
    # print(
    #     f"Length of dataset after removing non-english tweets: {ds.num_rows}"
    # )

    # apply the preprocessing function to every sample
    # tokenizer_path = "tokenizers/" + tokenizer_type
    # if not os.path.exists(tokenizer_path):
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_type)
    #     os.makedirs(tokenizer_path)
    #     tokenizer.save_pretrained(tokenizer_path)
    # else:
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_type)
    ds = ds.map(
        preprocess_sample, num_proc=6, fn_kwargs={"tweet": tweet}
        # preprocess_sample, num_proc=1, fn_kwargs={"tweet": tweet}
    )

    # remove redundant columns
    if tweet:
        ds = ds.remove_columns(
            [f"#sdg{i}" for i in range(1, 18)] + ["lang"])
    else:
        ds = ds.remove_columns([f"sdg{i}" for i in range(1, 18)])
    return ds

def split_dataset(
        ds: datasets.Dataset, tweet: bool = True
):
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
    if tweet:
        path = "data/raw/allSDGtweets.csv"
    else:
        path = "data/raw/scopus_ready_to_use.csv"

    ds = preprocess_dataset(file=path, nrows=nrows, tweet=tweet)
    ds_dict = split_dataset(ds, tweet=tweet)
    save_path = "data/processed"
    if tweet:
        save_path += "/tweets/base"
    else:
        save_path += "/scopus/base"
    ds_dict.save_to_disk(save_path)


def tokenize_dataset(tokenizer: transformers.PreTrainedTokenizer, tweet: bool = True, max_length: int = 260):
    if tweet:
        ds_dict = datasets.load_from_disk("data/processed/tweets/base")
        textname = "text"
    else:
        ds_dict = datasets.load_from_disk("data/processed/scopus/base")
        textname = "Abstract"

    for split in ds_dict.keys():
        if tweet:
            ds_dict[split] = ds_dict[split].map(lambda samples: tokenizer(samples[textname], padding="max_length", max_length=max_length, truncation=True), batched=True, num_proc=1)
        else:
            ds_dict[split] = ds_dict[split].map(
                lambda samples: tokenizer(samples[textname], add_special_tokens=False, truncation=False, return_attention_mask=False), num_proc=1)
        ds_dict[split] = ds_dict[split].remove_columns([textname, "nclasses", "label"])

    return ds_dict

def get_dataset(tokenizer_type: str, tweet: bool = True, sample_data: bool = False, max_length: int = 260, overwrite: bool = False):
    # load tokenized dataset if exists
    path_data = "data/processed"
    if tweet:
        path_data += "/tweets"
    else:
        path_data += "/scopus"
    path_base = path_data + "/base"

    path_ds_dict_tokens = path_data + f"/{tokenizer_type}"
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

    tokenizer_path = "tokenizers/" + tokenizer_type
    if not os.path.exists(tokenizer_path):
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_type)
        os.makedirs(tokenizer_path)
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    ds_dict_tokens = tokenize_dataset(tokenizer, tweet=tweet, max_length=max_length)
    path_save = path_data + f"/{tokenizer_type}"
    ds_dict_tokens.save_to_disk(path_save)

    for split in ds_dict_base.keys():
        ds_dict_tokens[split] = ds_dict_tokens[split].add_column("label", ds_dict_base[split]["label"])
        if sample_data:
            ds_dict_tokens[split] = ds_dict_tokens[split].select(np.arange(20))
    return ds_dict_tokens

def load_ds_dict(tokenizer_type: str, tweet: bool = True, path_data="data"):
    path_ds = os.path.join(path_data, "processed", "tweets" if tweet else "scopus")
    path_ds_dict_tokens = os.path.join(path_ds, tokenizer_type)
    ds_dict_tokens = datasets.load_from_disk(path_ds_dict_tokens)

    path_ds_dict_base = os.path.join(path_ds, "base")
    ds_dict_base = datasets.load_from_disk(path_ds_dict_base)

    for split in ds_dict_base.keys():
        ds_dict_tokens[split] = ds_dict_tokens[split].add_column("label", ds_dict_base[split]["label"])

    return ds_dict_tokens


if __name__ == "__main__":
    os.chdir("..")
    # create_base_dataset(tweet=True)
    # create_base_dataset(tweet=False)
    # ds_dict = datasets.load_from_disk("data/processed/tweets/base")
    # print()
    # tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    # ds_dict_tokens = tokenize_dataset(tokenizer)
    # print()
    # get_dataset("roberta-base")
    # get_dataset("roberta-base")
    # ds_dict = datasets.load_from_disk("data/processed/tweets/roberta-base")
    #ds_dict = get_dataset("roberta-base", sample_data=True)
    #print()
    # get_dataset("roberta-base", tweet=False)
    ds_dict = datasets.load_from_disk("data/processed/scopus/roberta-base")
    print()
