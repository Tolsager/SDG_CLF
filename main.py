import pandas as pd
import torch
import os

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import DS
from sklearn.model_selection import train_test_split

# our scripts
from trainer import SDGTrainer
from model import get_model


def main(batch_size: int=16, csv_path: str='data/raw/allSDGtweets.csv', epochs: int=2, multi_class: bool = False):
    df = pd.read_csv(csv_path, encoding='latin1')
    df = df[df['lang'] == 'en']
    if not multi_class:
        df = df[df['nclasses'] == 1]
    df = df.drop_duplicates('text')
    df = df.sample(frac=1)
    # df = df.sample(frac=0.001)
    df_train, df_test = train_test_split(df, train_size=0.9)
    df_train, df_cv = train_test_split(df_train, train_size=0.9)
    df_train.reset_index(drop=True, inplace=True)
    df_cv.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    if not os.path.exists('tokenizers/roberta_base'):
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        os.makedirs('tokenizers/roberta_base')
        tokenizer.save_pretrained('tokenizers/roberta_base')
    else:
        tokenizer = AutoTokenizer.from_pretrained('tokenizers/roberta_base')

    ds_train = DS(df_train, tokenizer)
    ds_cv = DS(df_cv, tokenizer)
    ds_test = DS(df_test, tokenizer)

    dl_train = DataLoader(ds_train, batch_size=batch_size)
    dl_cv = DataLoader(ds_cv, batch_size=batch_size)
    dl_test = DataLoader(ds_test, batch_size=batch_size)

    if not os.path.exists('pretrained_models/roberta_base'):
        model = get_model()
        os.makedirs('pretrained_models/roberta_base')
        model.save_pretrained('pretrained_models/roberta_base')
    else:
        model = get_model(pretrained_path='pretrained_models/roberta_base')

    if multi_class:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    trainer = SDGTrainer(model, epochs=epochs, criterion=criterion)
    trainer.train(dl_train, dl_cv)
    trainer.test(dl_test)


if __name__ == '__main__':
    main(batch_size=20, epochs=10)
