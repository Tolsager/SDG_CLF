import torch
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import os
import re


class DS(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tweet = self.df.iloc[idx, 2]
        labels = self.df.iloc[idx, 4:-2]
        type = self.df.iloc[idx, -2]
        num_labels = self.df.iloc[idx, -1]

        prog = re.compile(r'#\S+')
        tweet = prog.sub('', tweet)[0]
        tweet = ' '.join(tweet.split())

        encoding = self.tokenizer.encode_plus(tweet, padding='max_length', max_length=252, return_tensors='pt', truncation=True)

        ids = torch.squeeze(encoding['input_ids'])
        mask = torch.squeeze(encoding['attention_mask'])

        return {'input_ids': ids.long(), 'attention_mask': mask.int(), 'labels': torch.tensor(labels, dtype=torch.float32), "num_labels": num_labels, 'type': type}


if __name__ == '__main__':
    if not os.path.exists(os.getcwd() + '\pretrained'):
        os.mkdir(os.getcwd() + '\pretrained')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    tokenizer.save_pretrained('./pretrained/')
    testing = DS(pd.read_csv('data/allSDGtweets.csv', encoding='latin1'), tokenizer)
    testing_dl = DataLoader(testing, batch_size=10)

    for batch in testing_dl:
        print(batch)
        assert False
