import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# model importing used for classifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from model_mlp import MLPClassifier


# TODO: input 형식 정하기
class Classifier():
    """ MBTI Classifier """
    def __init__(self, model, config):
        """
        Args:
            model (str): 'mlp', 'catboost', 'xgboost' ...
            config (json): 
                ├── mlp
                │   ├── input_dim
                │   ├── hidden_dim
                │   ├── dropout
                │   └── ...
                ├── catboost
                │   ├── iterations
                │   ├── depth
                │   ├── learning_rate
                │   ├── loss_function
                │   ├── verbose
                │   └── ...
                ├── XGBoost
                │   ├── num_rounds
                │   ├── max_depth
                │   ├── max_leaves
                │   ├── alpha
                │   ├── eta
                │   └── ...
                ├── 
                └── 
        """
        super(Classifier, self).__init__()
        
        classifier = None

        if model == 'mlp':
            classifier = MLPClassifier(config['mlp']['input_dim'], 
                          config['mlp']['hidden_dim'],
                          config['mlp']['num_classes'], 
                          config['mlp']['dropout'])

        elif model == 'catboost':
            classifier = CatBoostClassifier(iterations=config['catboost']['iterations'], 
                                            depth=config['catboost']['depth'],
                                            learning_rate=config['catboost']['learning_rate'],
                                            loss_function=config['catboost']['loss_function'],
                                            verbose=config['catboost']['verbose'])

        elif model == 'xgboost':
            classifer = XGBClassifier()

        else:
            raise NotImplementedError
        
        assert classifier != None
        return classifier
