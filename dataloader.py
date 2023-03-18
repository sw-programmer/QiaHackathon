import os
import re

import torch
import pandas as pd
from torch.utils.data import Dataset
from pykospacing import Spacing                     #FIXME: 이 패키지가 m1 mac 에서 작동을 안 함...
from hanspell import spell_checker
from transformers import AutoTokenizer

#TODO:
# * train / text 다른 로직이 필요 (데이터 형식이 조금 다름)
# * 다른 Text preprocess 방식 도입 검토

class MBTIDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        question_path: str,
        target_mbti: str = None,
        txt_preprocess: bool = True,
        normalize: bool = True,
        is_binary_classification: bool = True,
        is_bert: bool = True,
        is_train: bool = True
        ):
        """DataLoader for MBTI dataset

        Args:
            data_path (str): Data file path. Both csv and parguet files are allowed.
            question_path (str): Question file path. Both csv and parguet files are allowed.
            target_mbti (str): Target mbti for binary classification.
            txt_preprocess (bool, optional): Text preprocessing pipeline. (e.g. fixing grammar, removing punctuations). Defaults to True.
            normalize (bool, optional): Normalize numeric attribute. Defaults to True.
            is_binary_classification (bool, optional): Target of task. You can choose btw Multi-class classificaiton
                and 4 binary classification problem. Defaults to True.
            is_bert (bool, optional): Using BERT for language model or not. Defaults to True.
            is_train (bool, optional): Whether given data is for training or not. Defaults to True.
        """

        data = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_parquet(data_path)
        self.question_data = pd.read_csv(question_path)

        # preprocess data
        if txt_preprocess:
            data['Answer'] = data['Answer'].apply(self.fix_grammar)
            data['Answer'] = data['Answer'].apply(self.fix_spacing)
            data['Answer'] = data['Answer'].apply(self.remove_punctuation)
        if normalize:
            data['Age'] = (data['Age'] - data['Age'].mean()) / data['Age'].std()

        # align dataset with binary classification (only for training data - test data doesn't contain 'MBTI' field)
        label_col = None
        if is_train and is_binary_classification:
            label_col = self.prepare_binary_classification(data, target_mbti)

        # prepare for language model
        #TODO: tokenizer class 인자로 넣어야 함 & tokenizer 만으로 [CLS], [SEP] 잘 붙는지 확인 필요
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.tokenize(data)

        # select columns for both training and inference
        #TODO: 테스트 데이터를 고려해서 유저 정보를 학습에 활용하지 않는 상황. 필요시 고쳐야 함.
        self.selected_cols = ['Gender', 'Age', 'QandA']
        self.label_col = label_col
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #TODO:
        # * getitem --> BERT input dimension 확인 후 변환 & 다른 features 들과 붙여서 input instance 생성
        pass

    # ======================
    #    Helper Functions
    # ======================

    def fix_grammar(self, answer: str) -> str:
        answer = spell_checker.check(answer)
        return answer.checked

    def fix_spacing(self, answer: str) -> str:
        answer = answer.replace(" ", '')
        spacing = Spacing()
        return spacing(answer)

    def remove_punctuation(self, answer: str) -> str:
        #FIXME: remove punctuation 검증 필요
        answer = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '', answer)
        answer = re.sub(r"^\s+", '', answer)                    # remove space from start
        answer = re.sub(r'\s+$', '', answer)                    # remove space from the end
        return answer

    def prepare_binary_classification(self, data: pd.DataFrame, target_mbti: str) -> str:
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
        
        return col_name

    def tokenize(self, data: pd.DataFrame):
        data['QandA'] =  data.apply(lambda row : self.tokenizer(                    \
            self.question_data.iloc[row['Q_number'] - 1].Question, row['Answer']    \
            ))







