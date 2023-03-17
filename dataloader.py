import os
import re

import torch
import pandas as pd
from torch.utils.data import Dataset
from pykospacing import Spacing
from hanspell import spell_checker

#TODO:
# * train / text 다른 로직이 필요 (데이터 형식이 조금 다름)
# * 다른 Text preprocess 방식 도입 검토

class QiaDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        question_path: str,
        target_mbti: str,
        txt_preprocess: bool = True,
        normalize: bool = True,
        is_bert: bool = True,
        is_binary_classification: bool = True,
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


        # make dataset for binary classification
        if is_binary_classification:
            self.prepare_binary_classification(data)


        # prepare for language model
        #TODO:
        # 1. 문장에 [CLS], [SEP] Token 붙여주기 --> QA 는 이어진 문장이므로 [SEP], [CLS] 잘 붙여줘야 함
        # 2. Tokenizer
        # 3. getitem --> BERT input dimension 확인 후 변환 & 다른 features 들과 붙여서 input instance 생성

        self.data = data

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

    def prepare_binary_classification(data: pd.DataFrame):
        t_value, f_value = 1, 0
        target_mbti = target_mbti.upper()
        if target_mbti not in ['E', 'N', 'F', 'P']:
            if target_mbti in ['I', 'S', 'T', 'J']:
                t_value, f_value = 0, 1
            else:
                raise ValueError ("Wrong mbti type. Try different type instead.")

        data['MBTI'] = data['MBTI'].str     \
            .contains(target_mbti)          \
                .replace({True: t_value, False: f_value})

        col_name = None
        if target_mbti in ('E', 'I'):
            col_name = 'I/E'
        elif target_mbti in ('N', 'S'):
            col_name = 'S/N'
        elif target_mbti in ('F', 'T'):
            col_name = 'T/F'
        else:
            col_name = 'J/P'
        data.rename(columns = {'MBTI':col_name}, inplace=True)

    def tokenize():
        pass







