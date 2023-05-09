import tqdm

import torch
from torch.utils.data import DataLoader

from mbti_dataset import MBTIDataset
from classifier import MLPClassifier
import module

# MBTI
MBTI = ['I/E', 'S/N', 'T/F', 'J/P']

def test(model, config, encoding, df, ):

  model.eval()
  loss_all, acc_all = 0, 0

  with torch.no_grad():
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
  

  # model.eval()
  pass

if __name__ == "__main__":
  test_path   = './data/sw/' + 'test_data_spacing_fixed.pickle'
  test_df = module.load_saved_data(test_path)
  pretrained_url = "xlm-roberta-large"
  test_encoding = module.tokenize(pretrained_url, test_df)

  for target in MBTI:
    test_set = MBTIDataset(test_encoding, test_df)
    test_loader = DataLoader(test_set, batch_size = 64, shuffle = False)
    
    model_path = './models/'
    model = torch.load(model_path)
  
  
