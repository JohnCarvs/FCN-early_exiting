import torch
import torchvision
import re

from models.FCN_8 import FCN8s
from models.FCN_16 import FCN16s
from models.FCN_32 import FCN32s

import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='FCN8', help='desired model to inspect')
opt = parser.parse_args()
print(opt)


"""nets"""
model = opt.model
match model:
    case "FCN8":
        model = FCN8s()
    case "FCN16":
        model = FCN16s()
    case "FCN32":
        model = FCN32s()

print(model)