import pickle
import gc
import os

import torch
from transformers import  AutoModel, AutoTokenizer

# Load train & test data
def load_saved_data(path):
    with open(path, 'rb') as handle:
        df = pickle.load(handle)
    print(df.head())
    return df

# Divide train dataset into train & valid
def divide_train_valid(train_df, ratio, seed):
    valid_df = train_df.sample(frac=ratio, random_state=seed)
    train_df = train_df.drop(valid_df.index)

    # reset index
    valid_df.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    print(f"len of train_df : {len(train_df)}, lend of valid_df : {len(valid_df)}")
    return train_df, valid_df

# Tokenize input in order to feed pretrained model
def tokenize(pretrained_url, df):

    # Tokenize question and answer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_url)

    # Convert data type for tokenizer (which accepts string only)
    question_ = [str(i) for i in df['Question'].values]             # Deprecated(현재는 Question을 사전 학습에 넣지 않고 따로 Embedding Layer 로 처리 중)
    answer_ = [str(i) for i in df['Answer'].values]
    encoding = tokenizer(
        answer_,
        padding=True,
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
        for param in base_model.base_model.parameters():
            param.requires_grad = False