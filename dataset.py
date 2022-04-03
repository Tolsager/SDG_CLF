import torch
import pandas as pd
from transformers import AutoTokenizer
import transformers
from torch.utils.data import Dataset, DataLoader
import os
import re


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


if __name__ == '__main__':
    # if not os.path.exists(os.getcwd() + '\pretrained'):
    #     os.mkdir(os.getcwd() + '\pretrained')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    # tokenizer.save_pretrained('./pretrained/')
    testing = DS(pd.read_csv('data/raw/allSDGtweets.csv', encoding='latin1'), tokenizer)
    testing_dl = DataLoader(testing, batch_size=10)

    for batch in testing_dl:
        print(batch)
        assert False
