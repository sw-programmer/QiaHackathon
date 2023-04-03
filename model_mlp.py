import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# TODO: input 형식 정하기
class MLPClassifier(nn.Module):
    """ MBTI Binary Classifier """
    def __init__(self, input_dim, hidden_dim, num_classes = 2, dropout = 0.5):
        super(MLPClassifier, self).__init__()

        assert type(hidden_dim) == list, ValueError("hidden_dim should be list type")

        self.hidden_dim = hidden_dim        # hidden dimension
        self.input_dim = input_dim          # input dimension
        self.num_classes = num_classes      # number of classes
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        
        # dimension list for all layers 
        self.dimensions = [self.input_dim] + self.hidden_dim + [self.num_classes]

        # layer stacks
        self.layers = nn.ModuleList(
            [nn.Linear(self.dimensions[i - 1], self.dimensions[i]) for i in range(1, len(self.dimensions))])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # If layer is not the last layer
            if i != len(self.layers) - 1: 
                x = self.dropout(F.relu(x))
        
        return x
