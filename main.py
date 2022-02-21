import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import DS
from sklearn.model_selection import train_test_split

# our scripts
from trainer import Trainer
from model import get_model

csv_path = './data/allSDGtweets.csv'


def main(batch_size=16, csv_path=csv_path, epochs=2):
    df = pd.read_csv(csv_path, encoding='latin1')
    df = df[df['nclasses'] == 1]
    df = df.sample(frac=1)
    df = df.sample(frac=0.02)
    df_train, df_test = train_test_split(df, train_size=0.8)

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    ds_train = DS(df_train, tokenizer)
    ds_test = DS(df_test, tokenizer)

    dl_train = DataLoader(ds_train, batch_size=batch_size)
    dl_test = DataLoader(ds_test, batch_size=batch_size)

    model = get_model()
    optimizer = torch.optim.AdamW

    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer=optimizer, lr=3e-5, epochs=epochs, criterion=criterion)
    trainer.train(dl_train, dl_test)


main()
