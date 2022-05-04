import torch
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


def preprocess(
    sample: dict, tokenizer: transformers.PreTrainedTokenizer, tweet: bool = True, textname: str = 'text'
):
    """preprocess a sample of the dataset

    Args:
        sample (dict): dataset sample
        tokenizer (transformers.PreTrainedTokenizer): a pretrained tokenizer
        tweet (bool): whether the data are tweets or abstracts
        textname (str): name of text column in 

    Returns:
        dict: the preprocessed sample
    """
    sample[textname] = sample[textname].lower()

    # remove labels from the tweet
    sdg_prog1 = re.compile(r"#(sdg)s?(\s+)?(\d+)?")
    sample = remove_with_regex(sample, pattern=sdg_prog1)
    sdg_prog2 = re.compile(r"(sdg)s?(\s?)(\d+)?")
    sample = remove_with_regex(sample, pattern=sdg_prog2)
    sdg_prog3 = re.compile(r"(sustainable development goals?\s?)(\d+)?")
    sample = remove_with_regex(sample, pattern=sdg_prog3)
    sdg_prog4 = re.compile(r"(© \d\d(\d?)\d)?\s")
    sample = remove_with_regex(sample, pattern=sdg_prog4)
    sdg_prog5 = re.compile(r"(Elsevier\sLtd)")
    sample = remove_with_regex(sample, pattern=sdg_prog5)

    # remove ekstra whitespace
    sample[textname] = " ".join(sample[textname].split())

    # create a label vector (only applicable for tweets)
    if tweet:
        label = [int(sample[f"#sdg{i}"]) for i in range(1, 18)]
        sample["label"] = label

    # tokenize text
    encoding = tokenizer(
        sample[textname], max_length=260, padding="max_length", truncation=True
    )
    sample["input_ids"] = encoding.input_ids
    sample["attention_mask"] = encoding.attention_mask
    return sample


def load_dataset(
    file: str = "data/raw/allSDGtweets.csv",
    seed: int = 0,
    nrows: int = None,
    multi_label: bool = True,
    tokenizer_type: str = "roberta-base",
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
    # Set the encodign to latin to be able to read special characters such as ñ
    tweet_df = pd.read_csv(file, encoding="latin", nrows=nrows)

    tweet_df = tweet_df.drop_duplicates("text")
    tweet_dataset = datasets.Dataset.from_pandas(tweet_df)
    if not multi_label:
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
    tokenizer_path = "tokenizers/" + tokenizer_type.replace("-", "_")
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

    tweet_dataset = tweet_dataset.shuffle(seed=seed)

    # tweet_dataset = tweet_dataset.cast_column("label", datasets.Sequence(datasets.Value("float32")))
    tweet_dataset = tweet_dataset.train_test_split(test_size=0.1)
    return tweet_dataset


if __name__ == "__main__":
    tweet_dataset = load_dataset(nrows=10)
    # tweet_dataset = load_dataset(nrows=10, multi_label=False)
    # tweet_dataset = load_dataset()
    tweet_dataset.set_format("torch", columns=["input_ids", "label", "attention_mask"])
    # print(tweet_dataset)
    print(type(tweet_dataset["train"][0]["input_ids"]))
    print()
