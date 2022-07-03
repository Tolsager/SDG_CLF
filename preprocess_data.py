import argparse

from sdg_clf import dataset_utils


def main(dataset_name: str):
    if dataset_name == "twitter":
        tweet = True
    elif dataset_name == "scopus":
        tweet = False
    dataset_utils.create_base_dataset(tweet=tweet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="twitter or scopus")
    args = parser.parse_args()
    main(args.dataset_name)
