import pandas as pd
import datasets
import transformers
import os
import re


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

    # remove unused columns
    if tweet:
        ds = ds.remove_columns(
            ["Unnamed: 0", "id", "created_at", "category"]
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
        preprocess_sample, num_proc=1, fn_kwargs={"tweet": tweet}
        # preprocess_sample, num_proc=1, fn_kwargs={"tweet": tweet}
    )

    # remove redundant columns
    if tweet:
        ds = ds.remove_columns(
            [f"#sdg{i}" for i in range(1, 18)] + ["lang"] + ["__index_level_0__"]
        )
    else:
        ds = ds.remove_columns(
            [f"sdg{i}" for i in range(1, 18)]
        )

    # if split:
    #     ds = ds.shuffle(seed=seed)
    #
    #     # ds = ds.cast_column("label", datasets.Sequence(datasets.Value("float32")))
    #     ds = ds.train_test_split(test_size=0.1)
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

    ds = preprocess_dataset(file=path, nrows=nrows)
    ds_dict = split_dataset(ds)
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
            ds_dict[split] = ds_dict[split].map(lambda samples: tokenizer(samples[textname], padding=True, max_length=max_length, truncation=True), batched=True, num_proc=1)
        else:
            ds_dict[split] = ds_dict[split].map(
                lambda samples: tokenizer(samples[textname]), num_proc=1)
        ds_dict[split] = ds_dict[split].remove_columns([textname, "nclasses", "label"])

    return ds_dict

def get_dataset(tokenizer_type: str, tweet: bool = True, nrows: int = None, max_length: int = 260):
    # load tokenized dataset if exists
    if tweet:
        path_ds_dict = f"data/processed/tweets/{tokenizer_type}"
        textname = "text"
    else:
        path_ds_dict = f"data/processed/scopus/{tokenizer_type}"
        textname = "Abstract"
    if os.path.exists(path_ds_dict):
        if nrows is not None:
            ds_dict = datasets.load_from_disk(path_ds_dict)
        else:
            ds_dict = datasets.load_from_disk(path_ds_dict)
        return ds_dict

    # else create dataset
    if tweet:
        path_base = "data/processed/tweets/base"
    else:
        path_base = "data/processed/scopus/base"

    if not os.path.exists(path_base):
        create_base_dataset(tweet=tweet)
    ds_dict = datasets.load_from_disk(path_base)

    tokenizer_path = "tokenizers/" + tokenizer_type
    if not os.path.exists(tokenizer_path):
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_type)
        os.makedirs(tokenizer_path)
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    ds_dict_tokens = tokenize_dataset(tokenizer, tweet=tweet, max_length=max_length)

    for split in ds_dict.keys():
        df = ds_dict[split].to_pandas()
        for column_name in ds_dict_tokens["train"].features.keys():
            df[column_name] = ds_dict_tokens[split][column_name]
        ds_dict[split] = datasets.Dataset.from_pandas(df)
        # ds_dict[split] = ds_dict[split].add_column(column_name, ds_dict_tokens[split][column_name])
        ds_dict[split] = datasets.concatenate_datasets([ds_dict[split], ds_dict_tokens[split]], axis=1)

    if tweet:
        ds_dict.save_to_disk(f"data/processed/tweets/{tokenizer_type}")
    else:
        ds_dict.save_to_disk(f"data/processed/scopus/{tokenizer_type}")

    return ds_dict


# def get_dataset(tokenizer_type: str, path_csv: str = "data/raw/allSDGtweets.csv", nrows: int = None):
#     """either loads the processed dataset if it exists and otherwise
#     creates the dataset and saves it to disk.
#
#     Args:
#         tokenizer_type (str): a huggingface tokenizer type such as "roberta-base"
#         path_csv (str, optional): path to raw tweet csv. Is only used if there is no processed dataset. Defaults to "data/raw/allSDGtweets.csv".
#
#     Returns:
#         datasets.DatasetDict: the processed dataset
#     """
#     path_ds = f"data/processed/{tokenizer_type}"
#     if not os.path.exists(path_ds):
#         split_dataset(path_csv, tokenizer_type=tokenizer_type, nrows=nrows)
#     ds = datasets.load_from_disk(path_ds)
#     return ds

if __name__ == "__main__":
    os.chdir("..")
    # create_base_dataset(tweet=True, nrows=20)
    # ds = datasets.load_from_disk("data/processed/tweets/base")
    # tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    # ds1 = tokenize_dataset(tokenizer)
    # print()
    get_dataset("roberta-base")