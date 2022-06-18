# IMPORTS
import glob

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from nltk import tokenize
from tensorflow import convert_to_tensor
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel
import pandas as pd
from transformers import BertConfig, BertTokenizer
from nltk import tokenize
from sklearn.model_selection import train_test_split
from tensorflow import convert_to_tensor
from transformers import TFBertModel, BertConfig
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.metrics import BinaryAccuracy, Precision, Recall
import time

nltk.download("punkt")

#our files
import datasets

# def create_model(label=None):
#     config=BertConfig.from_pretrained(
#                                     "bert-base-multilingual-uncased",
#                                      num_labels=2,
#                                      hidden_dropout_prob=0.2,
#                                      attention_probs_dropout_prob=0.2)
#     bert=TFBertModel.from_pretrained(
#                                     "bert-base-multilingual-uncased",
#                                     config=config)
#     bert_layer=bert.layers[0]
#     input_ids_layer=Input(
#                         shape=(512),
#                         name="input_ids",
#                         dtype="int32")
#     input_attention_masks_layer=Input(
#                                     shape=(512),
#                                     name="attention_masks",
#                                     dtype="int32")
#     bert_model=bert_layer(
#                         input_ids_layer,
#                         input_attention_masks_layer)
#     target_layer=Dense(
#                     units=1,
#                     kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
#                     name="target_layer",
#                     activation="sigmoid")(bert_model[1])
#     model=Model(
#                 inputs=[input_ids_layer, input_attention_masks_layer],
#                 outputs=target_layer,)
#                 # name="model_"+label.replace(".", "_"))
#     # optimizer=Adam(
#     #             learning_rate=5e-05,
#     #             epsilon=1e-08,
#     #             decay=0.01,
#     #             clipnorm=1.0)
#     # model.compile(
#     #             optimizer=optimizer,
#     #             loss="binary_crossentropy",
#     #             metrics=[BinaryAccuracy(), Precision(), Recall()])
#     return model

def tokenize_abstracts(abstracts):
    """For a given texts, adds '[CLS]' and '[SEP]' tokens
    at the beginning and the end of each sentence, respectively.
    """
    t_abstracts=[]
    for abstract in abstracts:
        t_abstract="[CLS] "
        for sentence in tokenize.sent_tokenize(abstract):
            t_abstract=t_abstract + sentence + " [SEP] "
        t_abstracts.append(t_abstract)
    return t_abstracts


tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-uncased')


def b_tokenize_abstracts(t_abstracts, max_len=512):
    """Tokenizes sentences with the help
    of a 'bert-base-multilingual-uncased' tokenizer.
    """
    b_t_abstracts=[tokenizer.tokenize(_)[:max_len] for _ in t_abstracts]
    return b_t_abstracts


def convert_to_ids(b_t_abstracts):
    """Converts tokens to its specific
    IDs in a bert vocabulary.
    """
    input_ids=[tokenizer.convert_tokens_to_ids(_) for _ in b_t_abstracts]
    return input_ids


def abstracts_to_ids(abstracts):
    """Tokenizes abstracts and converts
    tokens to their specific IDs
    in a bert vocabulary.
    """
    tokenized_abstracts=tokenize_abstracts(abstracts)
    b_tokenized_abstracts=b_tokenize_abstracts(tokenized_abstracts)
    ids=convert_to_ids(b_tokenized_abstracts)
    return ids


def pad_ids(input_ids, max_len=512):
    """Padds sequences of a given IDs.
    """
    p_input_ids=pad_sequences(input_ids,
                              maxlen=max_len,
                              dtype="long",
                              truncating="post",
                              padding="post")
    return p_input_ids


def create_attention_masks(inputs):
    """Creates attention masks
    for a given seuquences.
    """
    masks=[]
    for seq in inputs:
        seq_mask=[float(i>0) for i in seq]
        masks.append(seq_mask)
    return masks


# PREDICT

def float_to_percents(float, decimal=3):
    """Takes a float from range 0. to 0.9... as an input
    and converts it to a percentage with specified decimal places.
    """
    return str(float*100)[:(decimal+3)]+"%"

  
def models_predict(directory, inputs, attention_masks, float_to_percent=False):
    """Loads separate .h5 models from a given directory.
    For predictions, inputs are expected to be:
    tensors of token's ids (bert vocab) and tensors of attention masks.
    Output is of format:
    {'model/target N': [the probability of a text N dealing with the target N , ...], ...}
    """
    models=glob.glob(f"{directory}*.h5")
    predictions_dict={}
    # model = create_model()
    for _ in models:
        # model.load_weights(_)
        model=tf.keras.models.load_model(_)
        #predictions=model.predict_step([inputs, attention_masks])
        predictions = model.predict([inputs, attention_masks])
        predictions=[float(_) for _ in predictions]
        if float_to_percent==True:
            predictions=[float_to_percents(_) for _ in predictions]
        predictions_dict[model.name]=predictions
        del predictions, model
        # del predictions
    return predictions_dict

  
def predictions_dict_to_df(predictions_dictionary):
    """Converts model's predictions of format:
    {'model/target N': [the probability of a text N dealing with the target N , ...], ...}
    to a dataframe of format:
    | text N | the probability of the text N dealing with the target N | ... |
    """
    predictions_df=pd.DataFrame(predictions_dictionary)
    predictions_df.columns=[_.replace("model_", "").replace("_", ".") for _ in predictions_df.columns]
    predictions_df.insert(0, column="text", value=[_ for _ in range(len(predictions_df))])
    return predictions_df


def create_aurora_predictions(tweet: bool = False, split: str = "test", n_samples: int = 1400):
    if tweet:
        name_ds = "twitter"
        name_text = "text"
    else:
        name_ds = "scopus"
        name_text = "Abstract"
    ds_dict = datasets.load_from_disk(f"data/processed/{name_ds}/base")
    ds = ds_dict[split]
    texts = ds[name_text][:n_samples]
    abstracts=texts
    ids=abstracts_to_ids(abstracts)
    padded_ids=pad_ids(ids)
    masks=create_attention_masks(padded_ids)
    masks=convert_to_tensor(masks)
    inputs=convert_to_tensor(padded_ids)
    predictions=models_predict(directory="pretrained_models/mbert/", inputs=inputs, attention_masks=masks)
    predictions=predictions_dict_to_df(predictions)
    predictions = predictions[[str(i) for i in range(1,18)]]
    predictions = predictions.values
    predictions = np.where(predictions > 0.95, 1, 0)
    predictions = torch.tensor(predictions)
    return predictions




