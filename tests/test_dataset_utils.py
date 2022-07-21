import numpy as np
import transformers
import datasets

from sdg_clf import dataset_utils
import pandas as pd
import pytest
import os


def test_process_scopus_df():
    os.chdir("..")
    df = dataset_utils.process_scopus_df()
    assert len(df.columns) == 17 + 1


def test_process_twitter_df():
    os.chdir("..")
    df = dataset_utils.process_twitter_df()
    assert len(df.columns) == 17 + 1


def test_process_osdg_df():
    os.chdir("..")
    df = dataset_utils.process_twitter_df()
    assert len(df.columns) == 17 + 1


def test_remove_with_regex():
    sdg_prog1 = r"#(?:sdg)s?(\s+)?(\d+)?"
    sdg_prog2 = r"(?:sdg)s?(\s?)(\d+)?"
    patterns = [sdg_prog1, sdg_prog2]
    text = "1: #sdgs 1 2:#sdgs 10 3: sdgs 3"
    text_clean = dataset_utils.remove_with_regex(text, patterns)
    assert text_clean == "1:  2: 3: "


def test_process_text():
    text = "1: www.Google.com 2: #sdg1"
    text_clean = dataset_utils.process_text(text)
    assert text_clean == "1:  2: "


def test_process_df():
    os.chdir("..")
    dataset_name = "scopus"
    df = dataset_utils.process_df(dataset_name)
    assert len(df.columns) == 17 + 2
    with pytest.raises(ValueError):
        invalid_dataset_name = "invalid_dataset_name"
        dataset_utils.process_df(invalid_dataset_name)


def test_get_split_indices():
    n_elements = 100
    fractions = [0.33, 0.33, 0.33]
    indices = dataset_utils.get_split_indices(n_elements, fractions)
    assert len(indices) == 3
    assert indices[0][1] == 33
    assert type(indices[0][1]) == int
    assert indices[1][1] == 66
    assert indices[2][1] == 100


def test_split_df():
    os.chdir("..")
    df = pd.read_csv("data/raw/scopus_ready_to_use.csv")
    fractions = [0.8, 0.2]
    dfs = dataset_utils.split_df(df, fractions, ["train", "test"])
    assert len(dfs) == 2
    df_train = dfs["train"]
    df_test = dfs["test"]
    assert len(df_train) == int(np.round(0.8 * len(df)))
    assert len(df_test) == int(np.round(0.2 * len(df)))


def test_create_processed_dfs():
    os.chdir("..")
    dataset_name = "scopus"
    fractions = [0.5, 0.5]
    split_names = ["train", "test"]
    dfs = dataset_utils.create_processed_dfs(dataset_name, fractions, split_names)
    assert len(dfs) == 2
    df_train = dfs["train"]
    df_test = dfs["test"]
    assert len(df_train) == pytest.approx(len(df_test), 1)


def test_create_scopus_processed_dfs():
    os.chdir("..")
    dfs = dataset_utils.create_scopus_processed_dfs()
    assert len(dfs) == 2
    df_train = dfs["val"]
    df_test = dfs["test"]
    assert len(df_train) == pytest.approx(len(df_test), 1)


def test_create_twitter_processed_dfs():
    os.chdir("..")
    dfs = dataset_utils.create_twitter_processed_dfs()
    assert len(dfs) == 3
    df_train = dfs["train"]
    df_val = dfs["val"]
    df_test = dfs["test"]
    assert len(df_test) == pytest.approx(len(df_train) / 8, 2)


def test_create_osdg_processed_dfs():
    os.chdir("..")
    dfs = dataset_utils.create_osdg_processed_dfs()
    assert len(dfs) == 1
    df_test = dfs["test"]


def test_save_processed_dfs():
    os.chdir("..")
    dataset_name = "test_dataset"
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})
    dfs = {"train": df1, "test": df2}
    dataset_utils.save_processed_dfs(dataset_name, dfs)
    assert os.path.exists("data/processed/test_dataset/train.csv")
    assert os.path.exists("data/processed/test_dataset/test.csv")


def test_load_processed_df():
    os.chdir("..")
    dataset_name = "test_dataset"
    df = dataset_utils.load_processed_df(dataset_name, "train")
    assert len(df) == 3


def test_get_processed_df():
    os.chdir("..")
    with pytest.raises(ValueError):
        dataset_utils.get_processed_df("woeirj", "aoweijrf")
    df_scopus = dataset_utils.get_processed_df("scopus", "val")
    df_twitter = dataset_utils.get_processed_df("twitter", "train")
    df_osdg = dataset_utils.get_processed_df("osdg", "test")
    # check that all dfs have the same columns
    assert set(df_scopus.columns) == set(df_twitter.columns) == set(df_osdg.columns)


def test_get_labels_tensor():
    os.chdir("..")
    df = pd.read_csv("data/processed/scopus/test.csv")
    labels = dataset_utils.get_labels_tensor(df)
    assert labels.shape == (len(df), 17)


def test_process_dataset():
    os.chdir("..")
    df = pd.read_csv("data/processed/scopus/test.csv")
    ds = datasets.Dataset.from_pandas(df)
    tokenizer = transformers.AutoTokenizer.from_pretrained("albert-base-v2")
    processed_ds = dataset_utils.process_dataset(ds, tokenizer)
    assert set(processed_ds.format["columns"]) == {"input_ids", "attention_mask", "label"}
