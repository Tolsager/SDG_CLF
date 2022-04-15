import torch
import pandas as pd
from transformers import AutoTokenizer
import transformers
from torch.utils.data import Dataset, DataLoader
import os
import re
import datasets

class DS(Dataset):
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

def preprocess(sample: dict):
    sample["text"] = sample['text'].lower()

    # remove labels from the tweet
    sdg_prog = re.compile(r'#(sdg)s?(\s+)?(\d+)?')
    sample = remove_with_regex(sample, pattern=sdg_prog) 

    # remove ekstra whitespace
    sample["text"] = " ".join(sample['text'].split())

    
    label = [int(sample[f"#sdg{i}"]) for i in range(1, 18)]
    sample["label"] = label
    return sample

def load_dataset(file: str="data/raw/allSDGtweets.csv"):
    # load the csv file into a huggingface dataset
    # Set the encodign to latin to be able to read special characters such as ñ
    tweet_dataset = datasets.load_dataset("csv", data_files=file, encoding='latin')
    
    # remove unused columns
    tweet_dataset = tweet_dataset.remove_columns(['Unnamed: 0', 'id', 'created_at', 'category']) 

    tweet_dataset = tweet_dataset.map(preprocess, num_proc=6)

    
    return tweet_dataset

if __name__ == '__main__':
    tweet_dataset = load_dataset()
    print(tweet_dataset)
    print(tweet_dataset['train'][0])


# 10-fold cross-validation (see also next section on rounding behavior):
# The validation datasets are each going to be 10%:
# [0%:10%], [10%:20%], ..., [90%:100%].
# And the training datasets are each going to be the complementary 90%:
# [10%:100%] (for a corresponding validation set of [0%:10%]),
# [0%:10%] + [20%:100%] (for a validation set of [10%:20%]), ...,
# [0%:90%] (for a validation set of [90%:100%]).
# vals_ds = datasets.load_dataset('bookcorpus', split=[f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)])
# trains_ds = datasets.load_dataset('bookcorpus', split=[f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)])