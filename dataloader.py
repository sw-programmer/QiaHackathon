import os
import re

import torch
import pandas as pd
from torch.utils.data import Dataset
from pykospacing import Spacing
from hanspell import spell_checker

#TODO:
# * train / text 다른 전처리 필요한지 고민
# * 다른 Text preprocess 방식 도입 검토

class QiaDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        question_path: str,
        txt_preprocess: bool = True,
        normalize: bool = True,
        is_train: bool = True
        ):

        data = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_parquet(data_path)
        question = pd.read_csv(question_path)

        # preprocess data
        if txt_preprocess:
            data['Answer'] = data['Answer'].apply(self.fix_grammar)
            data['Answer'] = data['Answer'].apply(self.fix_spacing)
            data['Answer'] = data['Answer'].apply(self.remove_punctuation)
        if normalize:
            data['Age'] = data['Age'].apply(lambda x: (x-x.mean())/ x.std(), axis=0)

        self.data = data

        # prepare for language model
        #TODO:
        # 1. 문장에 [CLS], [SEP] Token 붙여주기
        # 2. Tokenizer
        # 3. getitem --> BERT input dimension 확인 후 변환 & 다른 features 들과 붙여서 input instance 생성


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass

    def fix_grammar(answer: str):
        answer = spell_checker.check(answer)
        return answer.checked

    def fix_spacing(answer: str):
        answer = answer.replace(" ", '')
        spacing = Spacing()
        return spacing(answer)

    def remove_punctuation(answer: str):
        #FIXME: remove punctuation
        answer = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '', answer)
        answer = re.sub(r"^\s+", '', answer)                    # remove space from start
        answer = re.sub(r'\s+$', '', answer)                    # remove space from the end

        return answer







