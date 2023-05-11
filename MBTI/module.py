import pickle
import gc
import os
import random

import torch
import pandas as pd
import numpy as np
from transformers import  AutoModel, AutoTokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Load train & test data
def load_saved_data(path):
    with open(path, 'rb') as handle:
        df = pickle.load(handle)
    return df

# Convert string to boolean for argparser
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        raise ValueError("!!! Wrong input for freezing argument !!!")

# Divide train dataset into train & valid based on MBTI
def divide_train_valid(train_df, number_per_MBTI, seed):

    valid_df = train_df.groupby('MBTI').apply(          \
        lambda x: x.sample(                             \
            n=number_per_MBTI, random_state=seed        \
            )                                           \
        ).reset_index(drop = True)
    train_df = train_df[~train_df['Data_ID'].isin(valid_df['Data_ID'].tolist())].reset_index(drop=True)        # Exclue valid data from trian data based on 'Data_ID' column (primary key)

    print(f"!!! len of train_df : {len(train_df)}, len of valid_df : {len(valid_df)}, you seleceted {number_per_MBTI} rows per MBTI !!!")
    return train_df, valid_df

# Encode main answer into one-hot embedded vector
def encode_one_hot(df):
    answer_list = ['<그렇다>', '<중립>', '<아니다>']
    count = 0
    for answer in answer_list:
        df[answer] = np.where(df.Answer.str.startswith(answer), 1, 0)
        count += df[answer].value_counts()[1]
    assert len(df) == count
    print("!!! one-hot encoded !!!")
    print(df.head())
    return df

# Tokenize input in order to feed pretrained model
def tokenize(pretrained_url, df):

    # Tokenize question and answer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_url)

    # Convert data type for tokenizer (which accepts string only)
    question_ = [str(i) for i in df['Question'].values]             # Deprecated(현재는 Question을 사전 학습에 넣지 않고 따로 Embedding Layer 로 처리 중)
    answer_ = [str(i) for i in df['Answer'].values]
    encoding = tokenizer(
        answer_,
        padding="max_length",
        max_length=80,
        truncation=True
    )

    print(tokenizer.decode(encoding['input_ids'][0]))
    return encoding

# Load pretrained model from higgingface (or from local dir if there is saved one)
def load_pretrained_model(pretrained_url):
    saved_path = f"./models/pretrained/{pretrained_url}"
    if os.path.exists(saved_path):
        base_model = AutoModel.from_pretrained(saved_path)
    else:
        base_model = AutoModel.from_pretrained(pretrained_url)  
        os.makedirs(saved_path, exist_ok=True)
        base_model.save_pretrained(saved_path)
    print(f"!!! selected model : {pretrained_url} !!!")
    return base_model

# Garbage collect
def collect_garbage():
    gc.collect()
    torch.cuda.empty_cache()

# Device preparation
def prepare_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using CPU instead.')
    return device

# Freeze Encoder, use head's parameters only
def freeze_encoder(base_model, freeze = True):
    if freeze:
        print("!!! Freeze Encoder !!!")
        for param in base_model.base_model.parameters():
            param.requires_grad = False
    else:
        print("!!! Don't freeze Encoder !!!")

def load_ckp(checkpoint_fpath, model, optim):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    print(f"!!! Load checkpoint from {checkpoint_fpath} !!!")
    return model, optim