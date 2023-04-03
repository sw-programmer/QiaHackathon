import os
import re
from tqdm import tqdm
from typing import Union

import torch
import pandas as pd
from torch.utils.data import Dataset
from pykospacing import Spacing
from hanspell import spell_checker
from transformers import AutoTokenizer

#TODO:
# 1. train / text 다른 로직이 필요 (데이터 형식이 조금 다름)
# 2. 다른 Text preprocess 방식 도입 검토
#       ref : https://ebbnflow.tistory.com/246

class MBTIDataset(Dataset):
    def __init__(
        self,
        data_path     : Union[str, pd.DataFrame],
        question_path : Union[str, pd.DataFrame],
        txt_preprocess: bool            = True,
        normalize     : bool            = True,
        pretrained_url: str             = "klue/bert-base",
        padding_per_batch               = True,
        is_binary_classification: bool  = True,
        is_bert       : bool            = True,
        is_train      : bool            = True
        ):
        """DataLoader for MBTI dataset

        Args:
            data_path (str): Data file path. Both csv and parguet files are allowed.
            question_path (str): Question file path. Both csv and parguet files are allowed.
            txt_preprocess (bool, optional): Text preprocessing pipeline. (e.g. fixing grammar, removing punctuations). Defaults to True.
            normalize (bool, optional): Normalize numeric attribute. Defaults to True.
            is_binary_classification (bool, optional): Target of task. You can choose btw Multi-class classificaiton
                and 4 binary classification problem. Defaults to True.
            is_bert (bool, optional): Using BERT for language model or not. Defaults to True.
            is_train (bool, optional): Whether given data is for training or not. Defaults to True.
        """

        def resolve_path(path:str)->pd.DataFrame:
            if path.endswith('.csv'):
                try:
                    df = pd.read_csv(path)
                except:
                    df = pd.read_csv(path, encoding='cp949')
            else:
                df = pd.read_parquet(path)
            return df

        data = None
        question_data = None
        label_cols = ['I/E', 'S/N', 'T/F', 'J/P']
        # if given data_path is pd.Dataframe, we assume preprocessing is already applied to given Dataframe
        # so that it can skip all the processes below
        if not isinstance(data_path, pd.DataFrame):
            data = resolve_path(data_path)
            question_data = resolve_path(question_path)

            self.question_data = question_data

            # preprocess data
            if txt_preprocess:
                self.preprocess_txt(data)
            if normalize:
                data['Age'] = (data['Age'] - data['Age'].mean()) / data['Age'].std()

            # make dataset suitable for binary classification (only for training data - test data doesn't contain 'MBTI' field)
            if is_train and is_binary_classification:
                self.prepare_binary_classification(data)
                # if method right above works successfully, then data should contain same # 0 and 1.
                for col in label_cols:
                    value_counted = data[col].value_counts()
                    assert value_counted[0] == value_counted[1]

            # prepare for language model
            #FIXME: df 를 넣으면 tokenizer 인식 못함. data 랑 같이 저장해줘야 할 듯.
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_url)
            self.padding_per_batch = padding_per_batch
            self.tokenize(data)

        else:
            data = data_path

        # set columns for both training and inference
        #TODO: 테스트 데이터를 고려해서 유저 정보를 학습에 활용하지 않는 상황. 필요시 고쳐야 함
        self.cat_col    = ['Gender']
        self.num_col    = ['Age']
        self.label_cols = label_cols
        self.is_train   = is_train
        self.data       = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        selected_data = self.data.iloc[idx]

        cat_input = torch.tensor(selected_data[self.cat_col])                       # [batch size   x   # categorical features]
        num_input = torch.tensor(selected_data[self.num_col])                       # [batch size   x   # numerical features]

        sample              = selected_data['QandA']                                # [batch size   x   sequence length]
        #FIXME: Dataframe 으로 넣으면 여기서 에러가 남
        #TypeError: 'str' object does not support item assignment
        sample['cat_input'] = cat_input
        sample['num_input'] = num_input

        # Include label only for training cases
        if self.is_train:
            for col in self.label_cols:
                label = torch.tensor(selected_data[col])                            # [batch size   x   1]
                sample[col] = label

        return sample

    # ======================
    #    Helper Functions
    # ======================

    def fix_grammar(self, answer: str) -> str:
        answer = spell_checker.check(answer)
        return answer.checked

    def fix_spacing(self, answer: str) -> str:
        answer  = answer.replace(" ", '')
        spacing = Spacing()
        return spacing(answer)

    def remove_punctuation(self, answer: str) -> str:
        answer = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '', answer)
        answer = re.sub(r'\s+', ' ', answer)        # remove extra space
        answer = re.sub(r"^\s+", '', answer)        # remove space from start
        answer = re.sub(r'\s+$', '', answer)        # remove space from the end
        return answer

    def preprocess_txt(self, data: pd.DataFrame):
        try:
            data['Answer'] = data['Answer'].apply(self.fix_grammar)         #FIXME: 해당 패키지의 서버가 가끔 응답 오류가 남...
        except:
            pass
        print('===============    fix_spacing     ===============')
        tqdm.pandas()
        data['Answer'] = data['Answer'].progress_apply(self.fix_spacing)
        print('=============== remove_punctuation ===============')
        tqdm.pandas()   # TODO:  필요 없으면 버리기
        data['Answer'] = data['Answer'].progress_apply(self.remove_punctuation)

    def prepare_binary_classification(self, data: pd.DataFrame):
        one_list = ['E', 'N', 'F', 'P']
        zero_list = ['I', 'S', 'T', 'J']

        for idx, mbti in enumerate(one_list):
            data[mbti] = data['MBTI'].str               \
                .contains(mbti)                         \
                .replace({True: 1, False: 0})

            new_name = zero_list[idx] + '/' + mbti
            data.rename(columns = {mbti:new_name}, inplace=True)

    def tokenize(self, data: pd.DataFrame):

        def tokenize_per_sentence(series: pd.Series) -> str:
            selected_question = self.question_data.iloc[series['Q_number'] - 1].Question
            selected_answer = series['Answer']

            padding = False if self.padding_per_batch else 'longest'
            #TODO: 필요시 max_length 조절 필요
            return self.tokenizer(selected_question,
                                  selected_answer,
                                  padding=padding)

        print('===============    tokenize    ===============')
        tqdm.pandas()
        data['QandA'] =  data.progress_apply(tokenize_per_sentence, axis=1)
