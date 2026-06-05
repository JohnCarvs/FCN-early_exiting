"""
File adapted from https://github.com/SJTUzhanglj/FCN
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import re

from data.data import SBDClassSeg, MyTestData
from utils.transform import Colorize
from utils.criterion import CrossEntropyLoss2d
from utils.val_metrics import ConfusionMatrix

from models.FCN_8 import FCN8s
from models.FCN_16 import FCN16s
from models.FCN_32 import FCN32s
from utils.imsave import imsave

#import visdom
import numpy as np
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--param', type=str, default=None, help='path to pre-trained parameters')
parser.add_argument('--data', type=str, default='./train', help='path to input data')
parser.add_argument('--out', type=str, default='./out', help='path to output data')
parser.add_argument('--epochs', type=int, default=90, help='total number of training epochs')
parser.add_argument('--model', type=str, default="FCN8", help='name of the model to run')
opt = parser.parse_args()
print(opt)


color_transform = Colorize()
"""parameters"""
iterNum = opt.epochs

"""data loader"""
# dataRoot = '/media/xyz/Files/data/datasets'
# checkRoot = '/media/xyz/Files/fcn8s-deconv'
dataRoot = opt.data
os.makedirs(opt.out, exist_ok=True)
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
n_class = len(SBDClassSeg.class_names)
print(f"Predicting {n_class} classes")

"""nets"""
model = opt.model
match model:
    case "FCN8":
        model = FCN8s(n_class)
    case "FCN16":
        model = FCN16s(n_class)
    case "FCN32":
        model = FCN32s(n_class)

"""load checkpoint"""
if opt.param is None:
    vgg16 = torchvision.models.vgg16(pretrained=True)
    model.copy_params_from_vgg16(vgg16, copy_fc8=False, init_upscore=True)
else:
    checkpoint = torch.load(opt.param, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

criterion = CrossEntropyLoss2d()
optimizer = torch.optim.Adam(model.parameters(), 0.0001, betas=(0.5, 0.999))
conf_matrix = ConfusionMatrix(n_class)

model = model.cuda()

if opt.phase == 'train':
    """train"""
    best_loss = float('inf')
    best_epoch = 0
    start_epoch = 0

    if opt.param is not None and isinstance(checkpoint, dict):
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'best_loss' in checkpoint:
            best_loss = checkpoint['best_loss']
        if 'best_epoch' in checkpoint:
            best_epoch = checkpoint['best_epoch']
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        else:
            match = re.search(r'epoch-(\d+)\.pth$', os.path.basename(opt.param))
            if match:
                start_epoch = int(match.group(1)) + 1

    if start_epoch == 0 or not os.path.exists(os.path.join(checkRoot, 'losses.csv')):
        with open(os.path.join(checkRoot, 'metrics.csv'), 'w') as f:
            f.write('epoch,train_loss,val_loss,mean_pixel_acc,miou\n')

    # iterate epochs
    for it in range(start_epoch, iterNum):

        train_epoch_loss = []
        val_epoch_loss = []
        
        #"""
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
                running_avg = sum(train_epoch_loss) / len(train_epoch_loss)
                print('loss: %.4f (epoch: %d, train) running_avg: %.4f' % (loss.item(), it, running_avg))
        #"""

        

        # iterate batches (validation)
        model.eval()
        conf_matrix.reset()
        with torch.no_grad():
            for ib, data in enumerate(val_loader):
                inputs = data[0].cuda()
                targets = data[1].cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                preds = outputs.argmax(dim=1)
                conf_matrix.update(preds, targets)

                val_epoch_loss.append(loss.item())
                if ib % 2 == 0:
                    image = inputs[0].detach().cpu()
                    image[0] = image[0] + 122.67891434
                    image[1] = image[1] + 116.66876762
                    image[2] = image[2] + 104.00698793
                    title = 'input (epoch: %d)' % (it)
                    title = 'output (epoch: %d)' % (it)
                    title = 'target (epoch: %d)' % (it)
                    running_val_avg = sum(val_epoch_loss) / len(val_epoch_loss)
                    print('loss: %.4f (epoch: %d, val) running_avg: %.4f' % (loss.item(), it, running_val_avg))
                    title = 'loss (epoch: %d)' % (it)

        # compute epoch averages
        average_train_loss = float(np.mean(train_epoch_loss)) if len(train_epoch_loss) > 0 else 0.0
        average_val_loss = float(np.mean(val_epoch_loss)) if len(val_epoch_loss) > 0 else 0.0

        print(conf_matrix.matrix)
        print("shape: ", conf_matrix.matrix.shape)
        print("total pixels: ", conf_matrix.matrix.sum())
        print("total diag: ", np.diag(conf_matrix.matrix).sum())
        print("GT distribuition: ", conf_matrix.matrix.sum(axis=1))
        print("pred distribuition: ", conf_matrix.matrix.sum(axis=0))

        print("mean pixel accuracy", conf_matrix.mean_pixel_acc())
        print("iou per class", conf_matrix.iou_per_class())
        print("mean miou", conf_matrix.miou())
        #sys.exit()
        

        mean_pixel_acc = conf_matrix.mean_pixel_acc()
        iou_per_class = conf_matrix.iou_per_class()
        miou = conf_matrix.miou()



        if average_val_loss < best_loss:
            best_loss = average_val_loss
            best_epoch = it


        filename = ('%s/FCN-epoch-%d.pth' \
                    % (checkRoot, it))
        torch.save({
            'epoch': it,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'best_epoch': best_epoch,
        }, filename)
        print('save: (epoch: %d)' % (it))

        with open(os.path.join(checkRoot, 'best_epoch.txt'), 'w') as f:
            f.write('Best epoch: %d with loss: %.4f' % (best_epoch, best_loss))

        # write losses to csv
        with open(os.path.join(checkRoot, 'metrics.csv'), 'a') as f:
            f.write('%d,%.4f,%.4f,%.4f,%.4f\n' % (it, average_train_loss, average_val_loss, mean_pixel_acc, miou))
else:
    for ib, data in enumerate(loader):
        print('testing batch %d' % ib)
        inputs = data[0].cuda()
        outputs = model(inputs)
        hhh = color_transform(outputs[0].detach().cpu().max(0)[1])
        imsave(os.path.join(outputRoot, data[1][0] + '.png'), hhh)
