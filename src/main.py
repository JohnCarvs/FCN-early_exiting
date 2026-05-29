"""
File adapted from https://github.com/SJTUzhanglj/FCN
"""

import torch
from torch.utils.data import DataLoader
import torchvision

from data.data import SBDClassSeg, MyTestData
from utils.transform import Colorize
from utils.criterion import CrossEntropyLoss2d
from models.FCN_8 import FCN8s
from utils.imsave import imsave

#import visdom
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--param', type=str, default=None, help='path to pre-trained parameters')
parser.add_argument('--data', type=str, default='./train', help='path to input data')
parser.add_argument('--out', type=str, default='./out', help='path to output data')
opt = parser.parse_args()
print(opt)


color_transform = Colorize()
"""parameters"""
iterNum = 30

"""data loader"""
# dataRoot = '/media/xyz/Files/data/datasets'
# checkRoot = '/media/xyz/Files/fcn8s-deconv'
dataRoot = opt.data
if not os.path.exists(opt.out):
    os.mkdir(opt.out)
if opt.phase == 'train':
    checkRoot = opt.out
    train_loader = torch.utils.data.DataLoader(
        SBDClassSeg(dataRoot, split='train', transform=True),
        batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        SBDClassSeg(dataRoot, split='seg11valid', transform=True),
        batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
else:
    outputRoot = opt.out
    loader = torch.utils.data.DataLoader(
        MyTestData(dataRoot, transform=True),
        batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

"""nets"""
model = FCN8s()
if opt.param is None:
    vgg16 = torchvision.models.vgg16(pretrained=True)
    model.copy_params_from_vgg16(vgg16, copy_fc8=False, init_upscore=True)
else:
    model.load_state_dict(torch.load(opt.param))

criterion = CrossEntropyLoss2d()
optimizer = torch.optim.Adam(model.parameters(), 0.0001, betas=(0.5, 0.999))

model = model.cuda()

if opt.phase == 'train':
    """train"""
    best_loss = float('inf')
    best_epoch = 0

    # iterate epochs
    for it in range(iterNum):

        train_epoch_loss = []
        val_epoch_loss = []
        
        # iterate batches (train)
        model.train()
        for ib, data in enumerate(train_loader):
            inputs = data[0].cuda()
            targets = data[1].cuda()
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            if ib % 2 == 0:
                image = inputs[0].detach().cpu()
                # desfaz normalização
                image[0] = image[0] + 122.67891434
                image[1] = image[1] + 116.66876762
                image[2] = image[2] + 104.00698793
                title = 'input (epoch: %d)' % (it)
                title = 'output (epoch: %d)' % (it)
                title = 'target (epoch: %d)' % (it)
                average = sum(train_epoch_loss) / len(train_epoch_loss)
                print('loss: %.4f (epoch: %d, train)' % (loss.item(), it))
                train_epoch_loss.append(average)
                title = 'loss (epoch: %d)' % (it)

        # iterate batches (validation)
        model.eval()
        with torch.no_grad():
            for ib, data in enumerate(val_loader):
                inputs = data[0].cuda()
                targets = data[1].cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_epoch_loss.append(loss.item())
                if ib % 2 == 0:
                    image = inputs[0].detach().cpu()
                    image[0] = image[0] + 122.67891434
                    image[1] = image[1] + 116.66876762
                    image[2] = image[2] + 104.00698793
                    title = 'input (epoch: %d)' % (it)
                    title = 'output (epoch: %d)' % (it)
                    title = 'target (epoch: %d)' % (it)
                    average = sum(val_epoch_loss) / len(val_epoch_loss)
                    print('loss: %.4f (epoch: %d, val)' % (loss.item(), it))
                    val_epoch_loss.append(average)
                    title = 'loss (epoch: %d)' % (it)

        if average < best_loss:
            best_loss = average
            best_epoch = it


        filename = ('%s/FCN-epoch-%d.pth' \
                    % (checkRoot, it))
        torch.save(model.state_dict(), filename)
        print('save: (epoch: %d)' % (it))
        
        with open(os.path.join(checkRoot, 'best_epoch.txt'), 'w') as f:
            f.write('Best epoch: %d with loss: %.4f' % (best_epoch, best_loss))
else:
    for ib, data in enumerate(loader):
        print('testing batch %d' % ib)
        inputs = data[0].cuda()
        outputs = model(inputs)
        hhh = color_transform(outputs[0].detach().cpu().max(0)[1])
        imsave(os.path.join(outputRoot, data[1][0] + '.png'), hhh)
