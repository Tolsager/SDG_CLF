import torch
import pandas as pd
import transformers
import os
import re
import datasets

class DS(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: transformers.PreTrainedTokenizer, multi_class: bool = False):
        self.df = df
        self.multi_class = multi_class
        self.tokenizer = tokenizer
        self.sdgs = [f'#sdg{i}' for i in range(1, 18)]
        self.sdg_prog = re.compile(r'#(sdg)s?(\s+)?(\d+)?')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tweet = self.df.loc[idx, 'text'].lower()
        labels = self.df.loc[idx, self.sdgs]
        num_labels = self.df.loc[idx, 'nclasses']

        tweet = self.sdg_prog.subn('', tweet)[0]
        tweet = ' '.join(tweet.split())
    
        if not self.multi_class:
            labels = torch.tensor(labels, dtype=torch.uint8).argmax(dim=0)
        else:
            labels = torch.tensor(labels, dtype=torch.float32)

        encoding = self.tokenizer(tweet, padding='max_length', max_length=252, return_tensors='pt', truncation=True)

        ids = torch.squeeze(encoding['input_ids'])
        mask = torch.squeeze(encoding['attention_mask'])

        return {'input_ids': ids.long(), 'attention_mask': mask.int(), 'labels': labels, "num_labels": num_labels}

        
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

    sample['text'] =  pattern.subn('', sample['text'])[0]
    return sample

def preprocess(sample: dict, tokenizer: transformers.PreTrainedTokenizer):
    """preprocess a sample of the dataset

    Args:
        sample (dict): dataset sample
        tokenizer (transformers.PreTrainedTokenizer): a pretrained tokenizer

    Returns:
        dict: the preprocessed sample
    """
    sample["text"] = sample['text'].lower()

    # remove labels from the tweet
    sdg_prog = re.compile(r'#(sdg)s?(\s+)?(\d+)?')
    sample = remove_with_regex(sample, pattern=sdg_prog) 

    # remove ekstra whitespace
    sample["text"] = " ".join(sample['text'].split())

    # create a label vector
    label = [int(sample[f"#sdg{i}"]) for i in range(1, 18)]
    sample["label"] = label

    # tokenize text
    encoding = tokenizer(sample['text'], max_length=260, padding="max_length", truncation=True, return_tensors="pt") 
    sample["input_ids"] = encoding.input_ids
    sample['attention_mask'] = encoding.attention_mask
    return sample

def load_dataset(file: str="data/raw/allSDGtweets.csv", seed: int = 0):
    """Loads the tweet CSV into a huggingface dataset and apply the preprocessing

    Args:
        file (str, optional): path to csv file. Defaults to "data/raw/allSDGtweets.csv".
        seed (int, optional): seed used for shuffling. Defaults to 0.

    Returns:
        datasets.Dataset: a preprocessed dataset
    """
    # load the csv file into a huggingface dataset
    # Set the encodign to latin to be able to read special characters such as ñ
    tweet_dataset = datasets.load_dataset("csv", data_files=file, encoding='latin')
    
    # remove unused columns
    tweet_dataset = tweet_dataset.remove_columns(['Unnamed: 0', 'id', 'created_at', 'category']) 
    print(f"Length of dataset before removing non-english tweets: {tweet_dataset.num_rows}")

    # remove non-english text
    tweet_dataset = tweet_dataset.filter(lambda sample: sample['lang'] == 'en') 
    print(f"Length of dataset after removing non-english tweets: {tweet_dataset.num_rows}")
    
    # apply the preprocessing function to every sample 
    tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-base')
    tweet_dataset = tweet_dataset.map(preprocess, num_proc=6, fn_kwargs={"tokenizer": tokenizer})

    # remove redundant columns
    tweet_dataset = tweet_dataset.remove_columns([f"#sdg{i}" for i in range(1, 18)] + ["lang"])
    
    tweet_dataset = tweet_dataset.shuffle(seed=seed)

    tweet_dataset = tweet_dataset["train"].train_test_split(test_size=0.1)
    return tweet_dataset

if __name__ == '__main__':
    tweet_dataset = load_dataset()
    print(tweet_dataset)
    print(tweet_dataset['train'][0])
