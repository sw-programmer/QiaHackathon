import random
import os
import re
from tqdm import tqdm
from tqdm import trange
import pprint

import pandas as pd
import numpy as np
from pykospacing import Spacing
from hanspell import spell_checker
# import nltk
import wandb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from transformers import  AutoModel, AutoTokenizer, BertForSequenceClassification, BertConfig, AdamW

from mbti_dataset import MBTIDataset
from classifier import MLPClassifier
import module

# Random seed
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# MBTI
MBTI = ['I/E', 'S/N', 'T/F', 'J/P']

#############################################
#
#   Definition of train, valid, and test
#         
#############################################

def train(model, loader, criterion, optim, device):
  model.train()
  loss_all, acc_all = 0, 0

  # Train loop
  for _, batch in tqdm(enumerate(loader)):
    input_ids       = batch['input_ids'].to(device)
    attention_mask  = batch['attention_mask'].to(device)
    gender  = batch['gender'].to(device)
    age     = batch['age'].to(device)
    q_num   = batch['q_num'].to(device)
    label   = batch['label'].to(device)
    output  = model(input_ids,
                    attention_mask=attention_mask,
                    gender=gender,
                    age=age,
                    q_num=q_num)

    loss = criterion(output, label)
    optim.zero_grad()
    loss.backward()
    optim.step()

    acc = (output.argmax(axis=1) == label).sum() / len(label)
    loss_all += loss.item()
    acc_all += acc.item()

  loss = loss_all / len(loader)
  acc = acc_all / len(loader)
  return loss, acc

def valid(model, loader, criterion, device):
  model.eval()
  loss_all, acc_all = 0, 0

  for _, batch in tqdm(enumerate(loader)):
    input_ids       = batch['input_ids'].to(device)
    attention_mask  = batch['attention_mask'].to(device)
    gender  = batch['gender'].to(device)
    age     = batch['age'].to(device)
    q_num   = batch['q_num'].to(device)
    label   = batch['label'].to(device)
    output  = model(input_ids,
                    attention_mask=attention_mask,
                    gender=gender,
                    age=age,
                    q_num=q_num)
    loss = criterion(output, label)

    acc = (output.argmax(axis=1) == label).sum() / len(label)

    loss_all += loss.item()
    acc_all += acc.item()

  loss = loss_all / len(loader)
  acc = acc_all / len(loader)
  return loss, acc

def runner(config,
           train_encoding,
           valid_encoding,
           train_df,
           valid_df,
           base_model,
           device):
  # MENDATORY
  user = 'sw'
  test_name = 'test_1'

  # Train 4 models for each of the MBTI attribute
  FINAL_RESULT = {}
  for target in ['T/F', 'J/P']:          # MBTI = ['I/E', 'S/N', 'T/F', 'J/P']

    print(f"##############   Target : {target}  ################")

    target_dir = "_".join(target.split('/'))

    # Make dataset
    train_set = MBTIDataset(train_encoding, train_df, target)
    valid_set = MBTIDataset(valid_encoding, valid_df, target)

    # Dataloader
    train_loader = DataLoader(train_set, batch_size = config['batch_size'], shuffle = True)
    valid_loader = DataLoader(valid_set, batch_size = config['batch_size'], shuffle = True)

    # Model
    model = MLPClassifier(base_model, config['hidden_dim'])
    model.to(device)

    # Model parameters w/ optimizer
    optim = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    # Train/Valid Loop
    train_final = []
    valid_final = []
    for epoch in tqdm(range(config['epoch'])):
      train_loss, train_acc = train(model, train_loader, criterion, optim, device)
      valid_loss, valid_acc = valid(model, valid_loader, criterion, device)

      wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'epoch': epoch})
      wandb.log({'valid_loss': valid_loss, 'valid_acc': valid_acc, 'epoch': epoch})

      train_final.append([train_loss, train_acc])
      valid_final.append([valid_loss, valid_acc])
      
      scheduler.step()

      # Save model for every 10 epochs or last model
      if epoch == 30 or epoch == config['epoch'] - 1:
        model_path = f'./models/{user}/{test_name}/{target_dir}'
        os.makedirs(model_path, exist_ok=True)
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer' : optim.state_dict(),
        }, f"{model_path}/epoch_{epoch}.pth.tar")

    FINAL_RESULT[target] = (train_final, valid_final)

  return FINAL_RESULT


if __name__ == "__main__":
    train_path  = './data/sw/' + 'train_data_spacing_fixed.pickle'
    train_df  = module.load_saved_data(train_path)
    train_df, valid_df  = module.divide_train_valid(train_df, 0.1, seed)

    pretrained_url = "xlm-roberta-large"
    train_encoding = module.tokenize(pretrained_url, train_df)
    valid_encoding = module.tokenize(pretrained_url, valid_df)
    base_model = module.load_pretrained_model(pretrained_url)
    module.collect_garbage()
    device = module.prepare_gpu()
    module.freeze_encoder(base_model)

    #train/valid
    config = {
      'batch_size' : 64,
      'hidden_dim' : [256, 32],
      'lr' : 2e-3,
      'momentum' : 0.9,
      'epoch' : 40
    }

    wandb.init(project='sw_test_2')
    wandb.config = config

    result = runner(config=config,
                    train_encoding=train_encoding,
                    valid_encoding=valid_encoding,
                    train_df=train_df,
                    valid_df=valid_df,
                    base_model=base_model,
                    device=device)
    
    
    
    
