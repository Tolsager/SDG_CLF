import argparse

from sdg_clf import dataset_utils
import datasets


def main(dataset_name: str):
    if dataset_name == "twitter":
        tweet = True
    elif dataset_name == "scopus":
        tweet = False
    if dataset_name == "twitter" or dataset_name == "scopus":
        dataset_utils.create_base_dataset(tweet=tweet)
    elif dataset_name == "osdg":
        ds = dataset_utils.preprocess_osdg()
        ds_dict = datasets.DatasetDict()
        ds_dict["test"] = ds
        ds_dict.save_to_disk(f"data/processed/{dataset_name}/base")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="twitter or scopus")
    args = parser.parse_args()
    main(args.dataset_name)
