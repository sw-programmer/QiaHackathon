import random
import os
import re
from tqdm import tqdm
from tqdm import trange
import pprint
import argparse

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
    others  = batch['Others'].to(device)
    q_num   = batch['Q_number'].to(device)
    label   = batch['label'].to(device)
    output  = model(input_ids,
                    attention_mask=attention_mask,
                    others=others,
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
    others  = batch['Others'].to(device)
    q_num   = batch['Q_number'].to(device)
    label   = batch['label'].to(device)
    output  = model(input_ids,
                    attention_mask=attention_mask,
                    others=others,
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
           resume,
           device,
           test_name):
  # MENDATORY
  user = 'sw'

  # Train 4 models for each of the MBTI attribute
  FINAL_RESULT = {}
  for target in MBTI:          # MBTI = ['I/E', 'S/N', 'T/F', 'J/P']

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

    start_epoch = 0
    # Load checkpoint when resumption
    if resume:
      target_epoch = 100
      checkpoint_fpath = f'./models/sw/{test_name}/{target_dir}/epoch_{target_epoch}.pth.tar'
      model, optim = module.load_ckp(checkpoint_fpath, model, optim)
      start_epoch = target_epoch
      config['epoch'] += target_epoch + 1

    # Train/Valid Loop
    train_final = []
    valid_final = []
    for epoch in tqdm(range(start_epoch, config['epoch'])):
      train_loss, train_acc = train(model, train_loader, criterion, optim, device)
      valid_loss, valid_acc = valid(model, valid_loader, criterion, device)

      try:
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'epoch': epoch})
        wandb.log({'valid_loss': valid_loss, 'valid_acc': valid_acc, 'epoch': epoch})
        wandb.log({'lr' : scheduler.get_last_lr()[0]})
        print({'lr' : scheduler.get_last_lr()})
      except:     # wandb 연결 끊어질 경우 방지
        print({'train_loss': train_loss, 'train_acc': train_acc, 'epoch': epoch})
        print({'valid_loss': valid_loss, 'valid_acc': valid_acc, 'epoch': epoch})

      train_final.append([train_loss, train_acc])
      valid_final.append([valid_loss, valid_acc])

      scheduler.step()

      # Save model for every 50 epochs or last model
      if epoch != 0:
        if epoch % 50 == 0 or epoch == config['epoch'] - 1:
          model_path = f'./models/{user}/{test_name}/{target_dir}'
          os.makedirs(model_path, exist_ok=True)
          torch.save({
              'state_dict': model.state_dict(),
              'optimizer' : optim.state_dict(),
          }, f"{model_path}/epoch_{epoch}.pth.tar")

    FINAL_RESULT[target] = (train_final, valid_final)

  return FINAL_RESULT


#############################################
#
#               Main function
#         
#############################################


if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=False, type=module.str2bool)    # Whether train from scratch or from saved checkpoint
    parser.add_argument('--epoch', type=int, default=50)                    # Epoch
    parser.add_argument('--batch_size', type=int, default=64)               # Batch Size
    parser.add_argument('--lr', type=float, default=1e-3)                   # Learning rate
    parser.add_argument('--momentum', type=float, default=0.9)              # Momentum
    parser.add_argument('--freeze', default=True, type=module.str2bool)     # Whether freeze pretrained model's parameters or not
    parser.add_argument('--proj', type=str, default='sw_test_nonfreeze')    # Project name for wandb
    parser.add_argument('--hf_url', type=str, default="xlm-roberta-base")  # Pretrained url for huggingface
    args = parser.parse_args()
    print(args)

    # Training preparation
    seed = 1234
    module.set_seed(seed)
    train_path  = './data/sw/' + 'train_data_augmented_v1.pickle'
    train_df  = module.load_saved_data(train_path)
    train_df, valid_df  = module.divide_train_valid(train_df, 150, seed)
    train_df = module.encode_one_hot(train_df)
    valid_df = module.encode_one_hot(valid_df)

    pretrained_url = args.hf_url
    train_encoding = module.tokenize(pretrained_url, train_df)
    valid_encoding = module.tokenize(pretrained_url, valid_df)
    base_model = module.load_pretrained_model(pretrained_url)
    module.collect_garbage()
    device = module.prepare_gpu()
    module.freeze_encoder(base_model, args.freeze)

    # Training configuration
    config = {
      'batch_size' : args.batch_size,
      'hidden_dim' : [256, 32],
      'lr' : args.lr,
      'momentum' : args.momentum,
      'epoch' : args.epoch
    }

    wandb.init(project=args.proj)
    wandb.config = config

    result = runner(config=config,
                    train_encoding=train_encoding,
                    valid_encoding=valid_encoding,
                    train_df=train_df,
                    valid_df=valid_df,
                    base_model=base_model,
                    resume=args.resume,
                    device=device,
                    test_name=args.proj)
    
    
    
    
