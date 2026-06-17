"""
File adapted from https://github.com/SJTUzhanglj/FCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d,self).__init__()

        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=-1)

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)

