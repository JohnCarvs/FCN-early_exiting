"""
File adapted from https://github.com/SJTUzhanglj/FCN
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import re
import random

from data.data import SBDClassSeg, MyTestData, VOC2011ClassSeg
from utils.transform import Colorize
from utils.criterion import CrossEntropyLoss2d
from utils.val_metrics import ConfusionMatrix
from utils.class_weights import compute_class_weights

from models.FCN_8 import FCN8s
from models.FCN_16 import FCN16s
from models.FCN_32 import FCN32s
from utils.imsave import imsave

#import visdom
import numpy as np
import argparse
import os
import sys

#AUX_WEIGHTS = {'s32': 0.4, 's16': 0.4}    # weights for auxiliary losses
AUX_WEIGHTS = {'s16': 0.4}    # weights for auxiliary losses

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--param', type=str, default=None, help='path to pre-trained parameters')
parser.add_argument('--data', type=str, default='./data/datasets', help='path to input data')
parser.add_argument('--dataset', type=str, default='SBD', help='dataset to be used (SBD or VOC)')
parser.add_argument('--finetune', action='store_true', default=False, help='Enables finetuning mode')

parser.add_argument('--out', type=str, default='./out', help='path to output data')
parser.add_argument('--epochs', type=int, default=200, help='total number of training epochs')
parser.add_argument('--model', type=str, default="FCN8", help='name of the model to run')
parser.add_argument('--aux', action='store_true', default=False, help='use auxiliary loss')
parser.add_argument('--class_weights', action='store_true', default=False, help='compute and cache class weights')
parser.add_argument('--aux_weights', type=str, default=None,
                    help="auxiliary weights, format 's16:0.4,s32:0.3' (overrides default AUX_WEIGHTS)")
parser.add_argument('--no_skip', action='store_true', default=False, help='disable skip connections')
parser.add_argument('--aux_upsample', action='store_true', default=False, help='add upsampling layers for auxiliary outputs')
parser.add_argument('--aux_interp', action='store_true', default=False, help='aux loss em resolucao cheia via F.interpolate (sem params); use SEM --aux_upsample')
parser.add_argument('--seed', type=int, default=None, help='random seed for training and data shuffling')
parser.add_argument('--deterministic', action='store_true', default=False, help='force deterministic cudnn behavior when using a seed')
parser.add_argument('--log_grad_align', action='store_true', default=False,
                    help='mede o alinhamento grad(L_aux) x grad(L_main) dos exits s16/s32 durante o treino '
                         '(medicao apenas; NAO adiciona as losses auxiliares ao treino). Requer FCN8.')
parser.add_argument('--align_every', type=int, default=40,
                    help='mede o alinhamento a cada N batches de treino (~213 amostras/epoca no SBD com N=40)')
opt = parser.parse_args()
print(opt)

if opt.log_grad_align:
    if opt.model != 'FCN8':
        raise SystemExit('--log_grad_align so suporta FCN8 (exits s16/s32)')
    if opt.aux_upsample:
        raise SystemExit('--log_grad_align nao suporta --aux_upsample (mediria um tap que nao e o treinado)')
    if opt.no_skip:
        raise SystemExit('--log_grad_align nao suporta --no_skip (sem a fusao, o tap s16 vira o proprio '
                         'upscore5 e a comparacao pre/pos-fusao perde o sentido)')
    if opt.align_every < 1:
        raise SystemExit('--align_every deve ser >= 1')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def set_seed(seed, deterministic=False):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(opt.seed, deterministic=opt.deterministic)
if opt.seed is not None:
    print(f"Seed set to: {opt.seed}")
    if opt.deterministic:
        print("Deterministic cuDNN enabled")


color_transform = Colorize()
"""parameters"""
iterNum = opt.epochs

# if provided via CLI, parse and override AUX_WEIGHTS (format: name:weight,name:weight)
if opt.aux_weights:
    parsed = {}
    for token in opt.aux_weights.split(','):
        token = token.strip()
        if not token:
            continue
        if ':' in token:
            name, val = token.split(':', 1)
        elif '=' in token:
            name, val = token.split('=', 1)
        else:
            raise ValueError(f"Invalid --aux_weights token: {token}. Use name:weight")
        try:
            parsed[name.strip()] = float(val)
        except Exception:
            raise ValueError(f"Invalid weight for {name}: {val}")
    AUX_WEIGHTS = parsed
    print('AUX_WEIGHTS set from CLI:', AUX_WEIGHTS)

"""data loader"""
# dataRoot = '/media/xyz/Files/data/datasets'
# checkRoot = '/media/xyz/Files/fcn8s-deconv'
dataRoot = opt.data
os.makedirs(opt.out, exist_ok=True)
if opt.phase in ['train', 'val']:
    checkRoot = opt.out

    if opt.dataset == "SBD":
        train_dataset = SBDClassSeg(dataRoot, split='train', transform=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            SBDClassSeg(dataRoot, split='seg11valid', transform=True),
            batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        n_class = len(SBDClassSeg.class_names)
    
    elif opt.dataset == "VOC":
        print("Loading train dataset...")
        train_dataset = VOC2011ClassSeg(dataRoot, split='train', transform=True)
        print("Train dataset size:", len(train_dataset))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        
        print("Loading val dataset...")
        val_dataset = VOC2011ClassSeg(dataRoot, split='val', transform=True)
        print("Val dataset size:", len(val_dataset))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        n_class = len(VOC2011ClassSeg.class_names)
    else:
        raise SystemExit('unknown --dataset %s (use SBD or VOC)' % opt.dataset)
else:
    outputRoot = opt.out
    loader = torch.utils.data.DataLoader(
        MyTestData(dataRoot, transform=True),
        batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    n_class = len(SBDClassSeg.class_names)  # 21 classes PASCAL (fase test nao carrega dataset rotulado)

print(f"Predicting {n_class} classes")

"""nets"""
model = opt.model
match model:
    case "FCN8":
        # aux=True tambem quando so medimos alinhamento: expoe os score maps
        # (exits sem parametros); as losses auxiliares SO entram no treino se opt.aux
        model = FCN8s(n_class, aux=(opt.aux or opt.log_grad_align), no_skip=opt.no_skip, aux_upsample=opt.aux_upsample)
    case "FCN16":
        model = FCN16s(n_class, aux=opt.aux, no_skip=opt.no_skip, aux_upsample=opt.aux_upsample)
    case "FCN32":
        model = FCN32s(n_class, aux=opt.aux, no_skip=opt.no_skip, aux_upsample=opt.aux_upsample)

"""load checkpoint"""
if opt.param is None:
    try:
        from torchvision.models import VGG16_Weights
        vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    except ImportError:
        vgg16 = torchvision.models.vgg16(pretrained=True)
    model.copy_params_from_vgg16(vgg16, copy_fc8=False, init_upscore=True)
else:
    import glob
    from torch.serialization import add_safe_globals

    def try_load(path):
        # try default load first
        try:
            return torch.load(path, map_location='cpu')
        except RuntimeError as re:
            # corrupted zip archive
            msg = str(re)
            if 'failed finding central directory' in msg or 'PytorchStreamReader failed' in msg:
                raise
            # otherwise fallthrough to other attempts
        except Exception:
            pass

        # try allowing unsafe globals and loading fully
        try:
            add_safe_globals([np._core.multiarray.scalar])
        except Exception:
            pass
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            pass

        # last resort: try weights-only True with added safe globals
        try:
            add_safe_globals([np._core.multiarray.scalar])
            return torch.load(path, map_location='cpu')
        except Exception:
            raise

    # attempt to load requested checkpoint; on zip-corruption, try other checkpoints in same dir
    try:
        checkpoint = try_load(opt.param)
    except RuntimeError as e:
        print(f"Checkpoint appears corrupted: {e}\nSearching for alternate checkpoints in the same directory...", file=sys.stderr)
        chk_dir = os.path.dirname(opt.param) or checkRoot
        candidates = glob.glob(os.path.join(chk_dir, 'FCN-epoch-*.pth'))
        # sort by epoch number descending
        def epoch_key(p):
            m = re.search(r'epoch-(\d+)\.pth$', p)
            return int(m.group(1)) if m else -1
        candidates = sorted(candidates, key=epoch_key, reverse=True)
        checkpoint = None
        for c in candidates:
            if os.path.abspath(c) == os.path.abspath(opt.param):
                continue
            try:
                print(f"Trying candidate checkpoint: {c}", file=sys.stderr)
                checkpoint = try_load(c)
                print(f"Loaded fallback checkpoint: {c}", file=sys.stderr)
                break
            except RuntimeError:
                print(f"Candidate corrupted, skipping: {c}", file=sys.stderr)
            except Exception as e2:
                print(f"Failed loading candidate {c}: {e2}", file=sys.stderr)
        if checkpoint is None:
            raise RuntimeError(f"No valid checkpoints found in {chk_dir}; original load failed: {e}")
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

class_weights = None
if opt.phase == 'train' and opt.class_weights:
    cache_path = os.path.join(dataRoot, 'class_weights.npy')
    class_weights = compute_class_weights(train_dataset, n_class, cache_path).to(device)
    print('class weights:', np.round(class_weights.cpu().numpy(), 3))
criterion = CrossEntropyLoss2d(weight=class_weights)

if opt.finetune:
    lr = 0.00001
else: 
    lr = 0.0001


optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.5, 0.999))
conf_matrix = ConfusionMatrix(n_class)

model = model.to(device)

align_logger = None
if opt.log_grad_align:
    from grad_alignment import AlignmentLogger
    align_logger = AlignmentLogger(model, aux_interp=opt.aux_interp)
    print('grad-alignment logging: 1 a cada %d batches -> %s'
          % (opt.align_every, os.path.join(opt.out, 'grad_align.csv')))

if opt.phase == 'train':
    """train"""
    best_loss = float('inf')
    best_epoch = 0
    best_miou = -1.0
    start_epoch = 0
    if opt.param is not None and isinstance(checkpoint, dict):

        if (not opt.finetune) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if opt.finetune:
            start_epoch = 0
            best_miou = -1
            best_epoch = 0
        else:
            if 'best_loss' in checkpoint:
                best_loss = checkpoint['best_loss']
            if 'best_epoch' in checkpoint:
                best_epoch = checkpoint['best_epoch']
            if 'best_miou' in checkpoint:
                best_miou = checkpoint['best_miou']
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            else:
                match = re.search(r'epoch-(\d+)\.pth$', os.path.basename(opt.param))
                if match:
                    start_epoch = int(match.group(1)) + 1

    # cabecalho apenas em comeco de treino (ou se o csv nunca existiu); um resume
    # (start_epoch > 0) preserva o historico e continua appendando
    if start_epoch == 0 or not os.path.exists(os.path.join(checkRoot, 'metrics.csv')):
        with open(os.path.join(checkRoot, 'metrics.csv'), 'w') as f:
            f.write('epoch,train_loss,val_loss,mean_pixel_acc,miou\n')
    if start_epoch == 0:
        # treino comecando do zero: um grad_align.csv antigo neste out dir nao
        # corresponde mais a esta run (remove mesmo sem --log_grad_align)
        align_csv = os.path.join(checkRoot, 'grad_align.csv')
        if os.path.exists(align_csv):
            os.remove(align_csv)

    # iterate epochs
    for it in range(start_epoch, iterNum):

        train_epoch_loss = []
        val_epoch_loss = []
        
        #"""
        # iterate batches (train)
        model.train()
        for ib, data in enumerate(train_loader):
            inputs = data[0].to(device)
            targets = data[1].to(device)
            model.zero_grad()
            out = model(inputs)
            outputs, aux_dict = (out[0], out[1]) if isinstance(out, tuple) else (out, {})
            loss = criterion(outputs, targets)
            loss_main = loss
            for name, w in AUX_WEIGHTS.items():
                # gate em opt.aux: com --log_grad_align o aux_dict existe mesmo em
                # runs baseline, e as losses auxiliares NAO podem entrar no treino
                if not opt.aux or name not in aux_dict:
                    continue
                if opt.aux_interp:
                    aux_up = F.interpolate(aux_dict[name], size=targets.shape[-2:],
                                           mode='bilinear', align_corners=False)
                    loss = loss + w * criterion(aux_up, targets)
                else:
                    gt_low = F.interpolate(targets.unsqueeze(1).float(), size = aux_dict[name].shape[-2:],
                                       mode='nearest').squeeze(1).long()
                    loss = loss + w * criterion(aux_dict[name], gt_low)
            if align_logger is not None and ib % opt.align_every == 0:
                # mede ANTES do backward (grafo vivo); observe() retem o grafo,
                # entao o backward do treino segue intocado
                align_logger.observe(criterion, loss_main, aux_dict, targets)
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
                inputs = data[0].to(device)
                targets = data[1].to(device)
                out = model(inputs)
                if isinstance(out, tuple):
                    outputs, aux_dict = out
                else:
                    outputs = out
                    aux_dict = {}
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

        mean_pixel_acc = conf_matrix.mean_pixel_acc()
        iou_per_class = conf_matrix.iou_per_class()         # how can we save this to use later?
        miou = conf_matrix.miou()

        print(conf_matrix.matrix)
        print("shape: ", conf_matrix.matrix.shape)
        print("total pixels: ", conf_matrix.matrix.sum())
        print("total diag: ", np.diag(conf_matrix.matrix).sum())
        print("GT distribuition: ", conf_matrix.matrix.sum(axis=1))
        print("pred distribuition: ", conf_matrix.matrix.sum(axis=0))

        print("mean pixel accuracy", mean_pixel_acc)
        print("iou per class", iou_per_class)
        print("mean miou legacy", miou)



        # save if mean IoU improved OR periodically every 25 epochs
        improved = False
        if miou > best_miou:
            best_miou = miou
            best_epoch = it
            improved = True

        periodic_save = (it % 25 == 0)

        if improved or periodic_save:
            filename = ('%s/FCN-epoch-%d.pth' \
                        % (checkRoot, it))
            torch.save({
                'epoch': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'best_miou': best_miou,
            }, filename)
            if improved:
                print('saved checkpoint (epoch: %d) with new best mIoU: %.4f' % (it, best_miou))
                with open(os.path.join(checkRoot, 'best_epoch.txt'), 'w') as f:
                    f.write('Best epoch: %d with mIoU: %.4f' % (best_epoch, best_miou))
            else:
                print('periodic save: checkpoint (epoch: %d) saved (mIoU=%.4f)' % (it, miou))
        else:
            print('no improvement in mIoU (epoch: %d: mIoU=%.4f), checkpoint not saved' % (it, miou))

        # write losses to csv
        with open(os.path.join(checkRoot, 'metrics.csv'), 'a') as f:
            f.write('%d,%.4f,%.4f,%.4f,%.4f\n' % (it, average_train_loss, average_val_loss, mean_pixel_acc, miou))

        # write per-epoch gradient-alignment row (and reset for the next epoch)
        if align_logger is not None:
            align_row = align_logger.summary()
            AlignmentLogger.write_row(os.path.join(checkRoot, 'grad_align.csv'), it, align_row)
            print('grad-align: cos16 %+.4f | cos32 %+.4f | diff %+.4f | act16 %+.4f | act32 %+.4f (n=%d)'
                  % (align_row['cos16_mean'], align_row['cos32_mean'], align_row['paired_diff_mean'],
                     align_row['cos_act16_mean'], align_row['cos_act32_mean'], align_row['batches']))

elif opt.phase == 'val':
    model.eval()
    conf_matrix.reset() 
    val_epoch_loss = [] 
    print("dataset size =", len(val_loader.dataset)) 
    print("loader size =", len(val_loader)) 
    
    with torch.no_grad(): 
        for ib, data in enumerate(val_loader):
            inputs = data[0].to(device) 
            targets = data[1].to(device) 
            out = model(inputs)
            if isinstance(out, tuple):
                outputs, aux_dict = out
            else:
                outputs = out
                aux_dict = {}
            loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)
            conf_matrix.update(preds, targets) 
            val_epoch_loss.append(loss.item()) 
            
            if ib % 20 == 0: 
                print( f"[{ib}/{len(val_loader)}] " f"loss={loss.item():.4f}" ) 
            
        average_val_loss = float(np.mean(val_epoch_loss)) 
        mean_pixel_acc = conf_matrix.mean_pixel_acc() 
        pixel_acc_global = conf_matrix.pixel_acc_global() 
        iou_per_class = conf_matrix.iou_per_class() 
        miou = conf_matrix.miou() 
        
        print() 
        print("=" * 80) 
        print("VALIDATION RESULTS") 
        print("=" * 80) 
        print(f"Val Loss : {average_val_loss:.6f}") 
        print(f"Mean Pixel Acc : {mean_pixel_acc:.6f}") 
        print(f"Global Pixel Acc : {pixel_acc_global:.6f}") 
        print(f"mIoU: {miou:.6f}") 
        print() 
        print("IoU per class") 
        
        for cls_name, cls_iou in zip(train_dataset.class_names, iou_per_class ):
            print(f"{cls_name:20s}: {cls_iou:.4f}") 
            print()
        
        valid_classes = ( 
            ( 
                np.diag(conf_matrix.matrix) + 
                (conf_matrix.matrix.sum(axis=0) - 
                np.diag(conf_matrix.matrix)) + 
                (conf_matrix.matrix.sum(axis=1) - 
                np.diag(conf_matrix.matrix)) 
            ) > 0 
        ) 
        
        print( f"Classes present in validation: " f"{valid_classes.sum()}/{len(valid_classes)}" )

else:
    model.eval()
    for ib, data in enumerate(loader):
        print('testing batch %d' % ib)
        inputs = data[0].to(device)
        out = model(inputs)
        if isinstance(out, tuple):
            outputs, aux_dict = out
        else:
            outputs = out
            aux_dict = {}
        hhh = color_transform(outputs[0].detach().cpu().max(0)[1])
        imsave(os.path.join(outputRoot, data[1][0] + '.png'), hhh)
        
