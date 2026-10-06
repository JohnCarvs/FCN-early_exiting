#!/usr/bin/env python3
"""
test_teamwork_unetpp.py - Decision-Centered Teamwork for UNet++ with Deep Supervision

Self-contained script: includes UNet++ architecture (ResNet34 encoder, 4 deep
supervision exits), auto-downloading Kvasir-SEG medical dataset (polyp
segmentation), training, and Decision-Centered Teamwork evaluation.

Key hypothesis: UNet++ exits are all post-fusion (each decoder node receives
dense skip connections), so they have more balanced quality than backbone-only
exits (e.g., ADP-C's 34%->80%). This should make teamwork aggregation
effective rather than harmful.

Usage:
  Self-test (no data needed):
    python test_teamwork_unetpp.py --selftest

  Train UNet++ on Kvasir-SEG (auto-downloads ~46MB):
    python test_teamwork_unetpp.py --train

  Evaluate teamwork on trained model:
    python test_teamwork_unetpp.py --model_file checkpoints_unetpp/unetpp_best.pth

Dependencies: torch, torchvision, numpy, Pillow (all standard in any DL environment)
"""

import argparse
import csv
import hashlib
import io
import os
import sys
import time
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# ==============================================================================
# Constants
# ==============================================================================

# Kvasir-SEG: binary segmentation (background=0, polyp=1)
KVASIR_NUM_CLASSES = 2
KVASIR_CLASS_NAMES = ['background', 'polyp']
KVASIR_IGNORE_LABEL = 255
KVASIR_URL = 'https://datasets.simula.no/downloads/kvasir-seg.zip'

# Cityscapes (optional, if user wants multi-class)
CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_IGNORE_LABEL = 255
CITYSCAPES_CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]
CITYSCAPES_LABEL_MAPPING = np.full(256, CITYSCAPES_IGNORE_LABEL, dtype=np.uint8)
for _tid, _lid in enumerate([7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]):
    CITYSCAPES_LABEL_MAPPING[_lid] = _tid

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(
        description='Decision-Centered Teamwork for UNet++ (Deep Supervision)')

    # Mode
    p.add_argument('--selftest', action='store_true',
                   help='Run self-test with synthetic data')
    p.add_argument('--train', action='store_true',
                   help='Train UNet++ with deep supervision')

    # Dataset
    p.add_argument('--dataset', type=str, default='kvasir',
                   choices=['kvasir', 'cityscapes'],
                   help='Dataset to use (default: kvasir, auto-downloads)')
    p.add_argument('--data_root', type=str, default=None,
                   help='Dataset root (default: data/kvasir-seg or data/cityscapes)')
    p.add_argument('--img_size', type=int, default=256,
                   help='Image size for training/eval (default: 256 for Kvasir)')

    # Model
    p.add_argument('--model_file', type=str, default=None,
                   help='Path to trained checkpoint (.pth)')
    p.add_argument('--encoder', type=str, default='resnet34',
                   choices=['resnet34', 'resnet50'])
    p.add_argument('--no_pretrained_encoder', action='store_true')

    # Training
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--ds_weights', type=float, nargs=3, default=[0.4, 0.3, 0.2],
                   help='Deep supervision weights for exits 1-3 (exit 4=1.0)')
    p.add_argument('--save_dir', type=str, default='checkpoints_unetpp')

    # Teamwork
    p.add_argument('--calib_ratio', type=float, default=0.3)
    p.add_argument('--bayes_matrix_file', type=str, default=None)
    p.add_argument('--output_csv', type=str, default=None)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--eval_batch_size', type=int, default=4)

    # General
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)

    return p.parse_args()


# ==============================================================================
# Auto-download Kvasir-SEG
# ==============================================================================

def download_kvasir_seg(data_root: str):
    """Downloads and extracts Kvasir-SEG dataset (~46MB) if not present."""
    images_dir = os.path.join(data_root, 'images')
    masks_dir = os.path.join(data_root, 'masks')

    if os.path.isdir(images_dir) and os.path.isdir(masks_dir):
        n_img = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        if n_img >= 1000:
            print(f"[Dataset] Kvasir-SEG already present: {n_img} images in {data_root}")
            return
        print(f"[Dataset] Kvasir-SEG incomplete ({n_img} images), re-downloading...")

    os.makedirs(data_root, exist_ok=True)
    zip_path = os.path.join(data_root, 'kvasir-seg.zip')

    if not os.path.exists(zip_path):
        print(f"[Dataset] Downloading Kvasir-SEG (~46MB) from {KVASIR_URL} ...")
        try:
            urllib.request.urlretrieve(KVASIR_URL, zip_path, _download_progress)
            print()  # newline after progress
        except Exception as e:
            print(f"\n[Error] Download failed: {e}")
            print("You can manually download from:")
            print(f"  {KVASIR_URL}")
            print(f"  and extract to: {data_root}")
            sys.exit(1)

    print(f"[Dataset] Extracting to {data_root} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_root)

    # Handle nested directory (zip may contain Kvasir-SEG/ folder)
    nested = os.path.join(data_root, 'Kvasir-SEG')
    if os.path.isdir(nested) and not os.path.isdir(images_dir):
        for sub in os.listdir(nested):
            src = os.path.join(nested, sub)
            dst = os.path.join(data_root, sub)
            if not os.path.exists(dst):
                os.rename(src, dst)
        if os.path.isdir(nested):
            try:
                os.rmdir(nested)
            except OSError:
                pass

    # Clean up zip
    try:
        os.remove(zip_path)
    except OSError:
        pass

    n_img = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    print(f"[Dataset] Kvasir-SEG ready: {n_img} images")


def _download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        bar = '#' * (pct // 2) + '-' * (50 - pct // 2)
        print(f"\r  [{bar}] {pct}% ({downloaded/1e6:.1f}/{total_size/1e6:.1f} MB)",
              end='', flush=True)


# ==============================================================================
# Datasets
# ==============================================================================

class KvasirSEGDataset(Dataset):
    """Kvasir-SEG polyp segmentation dataset (1000 images, binary).

    Directory layout:
        data_root/images/*.jpg
        data_root/masks/*.jpg
    """

    def __init__(self, data_root: str, indices: Optional[List[int]] = None,
                 img_size: int = 256, augment: bool = False):
        import PIL.Image
        self.img_size = img_size
        self.augment = augment
        self._PIL = PIL.Image

        img_dir = os.path.join(data_root, 'images')
        mask_dir = os.path.join(data_root, 'masks')

        all_imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.image_paths = [os.path.join(img_dir, f) for f in all_imgs]
        self.mask_paths = [os.path.join(mask_dir, f) for f in all_imgs]

        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        PIL = self._PIL
        img = PIL.Image.open(self.image_paths[idx]).convert('RGB')
        mask = PIL.Image.open(self.mask_paths[idx]).convert('L')

        # Resize
        sz = (self.img_size, self.img_size)
        img = img.resize(sz, PIL.Image.BILINEAR)
        mask = mask.resize(sz, PIL.Image.NEAREST)

        img = np.array(img, dtype=np.float32) / 255.0
        mask = (np.array(mask, dtype=np.float32) > 127.5).astype(np.int64)  # binary

        # Augmentations
        if self.augment:
            if np.random.random() > 0.5:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if np.random.random() > 0.5:
                img = img[::-1, :].copy()
                mask = mask[::-1, :].copy()

        # Normalize (ImageNet)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        mask = torch.from_numpy(mask.copy()).long()

        return img, mask


class CityscapesDataset(Dataset):
    """Cityscapes segmentation dataset (19 classes).

    Directory layout:
        data_root/leftImg8bit/{split}/{city}/*_leftImg8bit.png
        data_root/gtFine/{split}/{city}/*_gtFine_labelIds.png
    """

    def __init__(self, data_root: str, split: str = 'val',
                 img_size: Optional[int] = None, augment: bool = False):
        import glob as _glob
        import PIL.Image
        self._PIL = PIL.Image
        self.img_size = img_size
        self.augment = augment

        img_dir = os.path.join(data_root, 'leftImg8bit', split)
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Cityscapes not found at: {img_dir}")

        self.images = sorted(_glob.glob(os.path.join(img_dir, '*', '*_leftImg8bit.png')))
        self.labels = []
        for ip in self.images:
            ln = os.path.basename(ip).replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            city = os.path.basename(os.path.dirname(ip))
            self.labels.append(
                os.path.join(data_root, 'gtFine', split, city, ln))

        print(f"[Dataset] Cityscapes {split}: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        PIL = self._PIL
        img = PIL.Image.open(self.images[idx]).convert('RGB')
        lbl = PIL.Image.open(self.labels[idx])

        if self.img_size:
            img = img.resize((self.img_size, self.img_size), PIL.Image.BILINEAR)
            lbl = lbl.resize((self.img_size, self.img_size), PIL.Image.NEAREST)

        img = np.array(img, dtype=np.float32) / 255.0
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = CITYSCAPES_LABEL_MAPPING[lbl].astype(np.int64)

        if self.augment:
            if np.random.random() > 0.5:
                img = img[:, ::-1].copy()
                lbl = lbl[:, ::-1].copy()

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        lbl = torch.from_numpy(lbl.copy()).long()
        return img, lbl


# ==============================================================================
# UNet++ Model
# ==============================================================================

class ConvBlock(nn.Module):
    """Conv3x3-BN-ReLU x 2."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlus(nn.Module):
    """UNet++ with 4 deep supervision exits (ResNet34/50 encoder).

    All exits are post-fusion by design: even Exit 1 combines features from
    two encoder levels through skip connections. This gives more balanced
    exit quality compared to backbone-only exits.

    Returns: list of 4 tensors [exit1, exit2, exit3, exit4], each (B, C, H, W)
    at the original input resolution.
    """

    def __init__(self, num_classes: int = 2, encoder: str = 'resnet34',
                 pretrained_encoder: bool = True):
        super().__init__()
        self.num_classes = num_classes

        import torchvision.models as tv
        if encoder == 'resnet34':
            try:
                resnet = tv.resnet34(weights='IMAGENET1K_V1' if pretrained_encoder else None)
            except TypeError:
                resnet = tv.resnet34(pretrained=pretrained_encoder)
            ec = [64, 64, 128, 256, 512]
        elif encoder == 'resnet50':
            try:
                resnet = tv.resnet50(weights='IMAGENET1K_V1' if pretrained_encoder else None)
            except TypeError:
                resnet = tv.resnet50(pretrained=pretrained_encoder)
            ec = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")

        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        dc = [ec[0], ec[1], ec[2], ec[3]]  # decoder channels per level

        # Decoder nodes X^{i,j} - see Zhou et al. 2018
        # Level 3 (deepest decoder)
        self.x31 = ConvBlock(ec[3] + ec[4], dc[3])
        # Level 2
        self.x21 = ConvBlock(ec[2] + ec[3], dc[2])
        self.x22 = ConvBlock(ec[2] + dc[2] + dc[3], dc[2])
        # Level 1
        self.x11 = ConvBlock(ec[1] + ec[2], dc[1])
        self.x12 = ConvBlock(ec[1] + dc[1] + dc[2], dc[1])
        self.x13 = ConvBlock(ec[1] + 2*dc[1] + dc[2], dc[1])
        # Level 0 (produces exits)
        self.x01 = ConvBlock(ec[0] + ec[1], dc[0])
        self.x02 = ConvBlock(ec[0] + dc[0] + dc[1], dc[0])
        self.x03 = ConvBlock(ec[0] + 2*dc[0] + dc[1], dc[0])
        self.x04 = ConvBlock(ec[0] + 3*dc[0] + dc[1], dc[0])

        # Deep supervision heads
        self.seg1 = nn.Conv2d(dc[0], num_classes, 1)
        self.seg2 = nn.Conv2d(dc[0], num_classes, 1)
        self.seg3 = nn.Conv2d(dc[0], num_classes, 1)
        self.seg4 = nn.Conv2d(dc[0], num_classes, 1)

    @staticmethod
    def _up(t, ref):
        return F.interpolate(t, size=ref.shape[2:], mode='bilinear', align_corners=True)

    def forward(self, x):
        H, W = x.shape[2:]
        up = self._up

        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder
        d31 = self.x31(torch.cat([e3, up(e4, e3)], 1))

        d21 = self.x21(torch.cat([e2, up(e3, e2)], 1))
        d22 = self.x22(torch.cat([e2, d21, up(d31, e2)], 1))

        d11 = self.x11(torch.cat([e1, up(e2, e1)], 1))
        d12 = self.x12(torch.cat([e1, d11, up(d21, e1)], 1))
        d13 = self.x13(torch.cat([e1, d11, d12, up(d22, e1)], 1))

        d01 = self.x01(torch.cat([e0, up(e1, e0)], 1))
        d02 = self.x02(torch.cat([e0, d01, up(d11, e0)], 1))
        d03 = self.x03(torch.cat([e0, d01, d02, up(d12, e0)], 1))
        d04 = self.x04(torch.cat([e0, d01, d02, d03, up(d13, e0)], 1))

        # Deep supervision exits -> upsample to input resolution
        out1 = F.interpolate(self.seg1(d01), (H, W), mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.seg2(d02), (H, W), mode='bilinear', align_corners=True)
        out3 = F.interpolate(self.seg3(d03), (H, W), mode='bilinear', align_corners=True)
        out4 = F.interpolate(self.seg4(d04), (H, W), mode='bilinear', align_corners=True)

        return [out1, out2, out3, out4]


# ==============================================================================
# Metrics
# ==============================================================================

def compute_confusion_matrix(pred: np.ndarray, target: np.ndarray,
                             num_classes: int) -> np.ndarray:
    valid = (target >= 0) & (target < num_classes)
    p, t = pred[valid].astype(np.int64), target[valid].astype(np.int64)
    return np.bincount(num_classes * p + t,
                       minlength=num_classes**2).reshape(num_classes, num_classes)


def calculate_miou_from_matrix(matrix: np.ndarray) -> Tuple[float, float, np.ndarray]:
    tp = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0).astype(np.float64) + matrix.sum(1).astype(np.float64) - tp
    valid = union > 0
    ious = np.zeros(matrix.shape[0], dtype=np.float64)
    ious[valid] = tp[valid] / union[valid]
    miou = float(np.mean(ious[valid])) if valid.any() else 0.0
    total = matrix.sum()
    acc = float(tp.sum() / total) if total > 0 else 0.0
    return miou, acc, ious


def dice_from_matrix(matrix: np.ndarray) -> float:
    """Dice coefficient for the foreground class (class 1)."""
    if matrix.shape[0] < 2:
        return 0.0
    tp = float(matrix[1, 1])
    fp = float(matrix[1, 0])
    fn = float(matrix[0, 1])
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


# ==============================================================================
# Decision-Centered Teamwork Engine
# ==============================================================================

class TeamworkSegmentationEngine:
    """Decision-Centered Teamwork for multi-exit dense prediction."""

    def __init__(self, num_classes: int = 2, num_exits: int = 4,
                 conf_thresholds: Tuple[float, float] = (0.70, 0.90),
                 laplace_eps: float = 1.0, device: str = 'cpu'):
        self.num_classes = num_classes
        self.num_exits = num_exits
        self.conf_thresholds = conf_thresholds
        self.laplace_eps = laplace_eps
        self.device = device
        self.log_likelihood_matrix: Optional[torch.Tensor] = None
        self.log_binned_likelihood_matrix: Optional[torch.Tensor] = None
        self.exit_weights: List[float] = [1.0] * num_exits

    def fit_likelihood_matrices(self, calib_loader, model,
                                max_batches: Optional[int] = None):
        print(f"\n[Teamwork] Fitting Bayesian likelihoods ({self.num_exits} exits)...")
        model.eval()
        C = self.num_classes

        counts = np.zeros((self.num_exits, C, C), dtype=np.int64)
        binned = np.zeros((self.num_exits, 3, C, C), dtype=np.int64)
        t0 = time.time()
        n = 0

        with torch.no_grad():
            for bi, batch in enumerate(calib_loader):
                if max_batches and bi >= max_batches:
                    break
                imgs = batch[0].to(self.device)
                tgts = batch[1].long().to(self.device)
                tgt_np = tgts.cpu().numpy()

                outs = model(imgs)
                if not isinstance(outs, (list, tuple)):
                    outs = [outs]

                for j in range(min(len(outs), self.num_exits)):
                    o = outs[j]
                    if o.shape[-2:] != tgts.shape[-2:]:
                        o = F.interpolate(o, tgts.shape[-2:],
                                          mode='bilinear', align_corners=True)
                    prob = F.softmax(o, dim=1)
                    conf, pred = prob.max(dim=1)
                    p_np = pred.cpu().numpy()
                    c_np = conf.cpu().numpy()

                    v = (tgt_np >= 0) & (tgt_np < C)
                    pv, tv, cv = p_np[v], tgt_np[v], c_np[v]

                    counts[j] += np.bincount(C*pv+tv, minlength=C*C).reshape(C,C)

                    lo, hi = self.conf_thresholds
                    bi_idx = np.zeros_like(cv, dtype=np.int64)
                    bi_idx[cv >= hi] = 2
                    bi_idx[(cv >= lo) & (cv < hi)] = 1
                    for b in range(3):
                        m = bi_idx == b
                        if m.any():
                            binned[j,b] += np.bincount(
                                C*pv[m]+tv[m], minlength=C*C).reshape(C,C)
                n += 1
                if n % 10 == 0:
                    print(f"  Calibrated {n} batches ({time.time()-t0:.1f}s)...")

        print(f"[Teamwork] Calibration done: {n} batches in {time.time()-t0:.1f}s")

        eps = self.laplace_eps
        norm = np.zeros_like(counts, dtype=np.float64)
        for j in range(self.num_exits):
            d = counts[j].sum(0, keepdims=True) + C * eps
            norm[j] = (counts[j] + eps) / np.maximum(d, 1e-12)
        self.log_likelihood_matrix = torch.from_numpy(np.log(norm)).float().to(self.device)

        norm_b = np.zeros_like(binned, dtype=np.float64)
        for j in range(self.num_exits):
            for b in range(3):
                d = binned[j,b].sum(0, keepdims=True) + C * eps
                norm_b[j,b] = (binned[j,b] + eps) / np.maximum(d, 1e-12)
        self.log_binned_likelihood_matrix = torch.from_numpy(np.log(norm_b)).float().to(self.device)

        mious = [calculate_miou_from_matrix(counts[j])[0] for j in range(self.num_exits)]
        self.exit_weights = mious
        print("[Teamwork] Calibration mIoU: " +
              ", ".join(f"exit{j+1}={m*100:.2f}%" for j,m in enumerate(mious)))

    def save_likelihoods(self, path):
        np.save(path, {
            'log_l': self.log_likelihood_matrix.cpu().numpy(),
            'log_bl': self.log_binned_likelihood_matrix.cpu().numpy(),
            'nc': self.num_classes, 'ne': self.num_exits,
            'ct': self.conf_thresholds, 'ew': list(self.exit_weights)
        }, allow_pickle=True)
        print(f"[Teamwork] Saved matrices to {path}")

    def load_likelihoods(self, path):
        d = np.load(path, allow_pickle=True).item()
        self.log_likelihood_matrix = torch.from_numpy(d['log_l']).float().to(self.device)
        self.log_binned_likelihood_matrix = torch.from_numpy(d['log_bl']).float().to(self.device)
        if 'ew' in d:
            self.exit_weights = list(d['ew'])
        print(f"[Teamwork] Loaded matrices from {path}")

    def predict_baseline(self, logits, stage):
        return logits[stage].argmax(dim=1)

    def predict_logit_voting(self, logits, stage, weights=None):
        combined = torch.zeros_like(logits[0])
        for j in range(stage + 1):
            w = weights[j] if weights else 1.0
            combined = combined + w * logits[j]
        return combined.argmax(dim=1)

    def predict_bayesian(self, logits, stage):
        B, C, H, W = logits[0].shape
        post = torch.zeros((B, H, W, self.num_classes), device=self.device)
        for j in range(stage + 1):
            pred_j = logits[j].argmax(dim=1)
            post = post + self.log_likelihood_matrix[j][pred_j]
        return post.argmax(dim=-1)

    def predict_conformal_bayesian(self, logits, stage):
        B, C, H, W = logits[0].shape
        post = torch.zeros((B, H, W, self.num_classes), device=self.device)
        lo, hi = self.conf_thresholds
        for j in range(stage + 1):
            prob = F.softmax(logits[j], dim=1)
            conf, pred = prob.max(dim=1)
            bn = torch.zeros_like(pred, dtype=torch.long)
            bn[conf >= hi] = 2
            bn[(conf >= lo) & (conf < hi)] = 1
            post = post + self.log_binned_likelihood_matrix[j, bn, pred]
        return post.argmax(dim=-1)


# ==============================================================================
# Training
# ==============================================================================

def train_unetpp(args):
    print("\n" + "=" * 80)
    print("TRAINING UNet++ WITH DEEP SUPERVISION")
    print("=" * 80)

    device = args.device
    os.makedirs(args.save_dir, exist_ok=True)

    # Determine dataset
    if args.dataset == 'kvasir':
        data_root = args.data_root or 'data/kvasir-seg'
        download_kvasir_seg(data_root)
        num_classes = KVASIR_NUM_CLASSES
        ignore_label = KVASIR_IGNORE_LABEL

        # 80/20 train/val split
        n_total = len([f for f in os.listdir(os.path.join(data_root, 'images'))
                       if f.endswith('.jpg')])
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n_total)
        n_train = int(0.8 * n_total)
        train_idx = perm[:n_train].tolist()
        val_idx = perm[n_train:].tolist()

        train_ds = KvasirSEGDataset(data_root, indices=train_idx,
                                    img_size=args.img_size, augment=True)
        val_ds = KvasirSEGDataset(data_root, indices=val_idx,
                                  img_size=args.img_size, augment=False)
        print(f"[Split] {n_total} images -> {len(train_idx)} train / {len(val_idx)} val")
    else:
        data_root = args.data_root or 'data/cityscapes'
        num_classes = CITYSCAPES_NUM_CLASSES
        ignore_label = CITYSCAPES_IGNORE_LABEL
        train_ds = CityscapesDataset(data_root, 'train', args.img_size, augment=True)
        val_ds = CityscapesDataset(data_root, 'val', args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = UNetPlusPlus(num_classes=num_classes, encoder=args.encoder,
                         pretrained_encoder=not args.no_pretrained_encoder).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] UNet++ ({args.encoder}): {n_params/1e6:.1f}M params, {num_classes} classes")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    max_iter = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
    ds_w = list(args.ds_weights) + [1.0]  # [exit1, exit2, exit3, exit4]
    print(f"[Training] {args.epochs} epochs, bs={args.batch_size}, lr={args.lr}, "
          f"img_size={args.img_size}")
    print(f"[Training] DS weights: {ds_w}")

    best_miou = 0.0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = sum(w * criterion(o, labels) for w, o in zip(ds_w, outputs))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)

        # Validate periodically
        do_val = (epoch + 1) % 5 == 0 or epoch == args.epochs - 1
        if do_val:
            model.eval()
            # Evaluate ALL 4 exits on validation
            exit_matrices = [np.zeros((num_classes, num_classes), dtype=np.int64)
                             for _ in range(4)]
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs = imgs.to(device)
                    outs = model(imgs)
                    for j in range(4):
                        pred = outs[j].argmax(dim=1).cpu().numpy()
                        exit_matrices[j] += compute_confusion_matrix(
                            pred, lbls.numpy(), num_classes)

            mious = [calculate_miou_from_matrix(m)[0] for m in exit_matrices]
            dices = [dice_from_matrix(m) for m in exit_matrices] if num_classes == 2 else [0]*4
            main_miou = mious[-1]

            exit_str = " | ".join(f"E{j+1}={m*100:.1f}%" for j, m in enumerate(mious))
            dice_str = ""
            if num_classes == 2:
                dice_str = " | Dice: " + ", ".join(f"E{j+1}={d*100:.1f}%" for j, d in enumerate(dices))
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | loss={avg_loss:.4f} | "
                  f"mIoU: {exit_str}{dice_str} | {time.time()-t0:.1f}s")

            if main_miou > best_miou:
                best_miou = main_miou
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'miou': main_miou,
                    'all_mious': mious,
                    'encoder': args.encoder,
                    'num_classes': num_classes,
                    'dataset': args.dataset,
                    'img_size': args.img_size,
                }, os.path.join(args.save_dir, 'unetpp_best.pth'))
                print(f"    -> New best! Saved checkpoint (mIoU={main_miou*100:.2f}%)")
        else:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | loss={avg_loss:.4f} | "
                  f"{time.time()-t0:.1f}s")

    # Save final
    torch.save({
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'miou': best_miou,
        'encoder': args.encoder,
        'num_classes': num_classes,
        'dataset': args.dataset,
        'img_size': args.img_size,
    }, os.path.join(args.save_dir, 'unetpp_final.pth'))

    print(f"\n[OK] Training complete! Best mIoU: {best_miou*100:.2f}%")
    print(f"     Best:  {os.path.join(args.save_dir, 'unetpp_best.pth')}")
    print(f"     Final: {os.path.join(args.save_dir, 'unetpp_final.pth')}")


# ==============================================================================
# Teamwork Evaluation
# ==============================================================================

def run_evaluation(val_loader, model, engine: TeamworkSegmentationEngine,
                   output_csv: Optional[str] = None,
                   max_samples: Optional[int] = None,
                   class_names: Optional[List[str]] = None):
    print("\n" + "=" * 90)
    print("STARTING DECISION-CENTERED TEAMWORK EVALUATION (UNet++ 4 EXITS)")
    print("=" * 90)

    model.eval()
    ne = engine.num_exits
    nc = engine.num_classes

    methods = [
        'Baseline (Single Exit)',
        'Logit Voting (Uniform)',
        'Logit Voting (Weighted)',
        'Bayesian Updating',
        'Bayes + Conformal Bins',
    ]
    matrices = {m: [np.zeros((nc, nc), dtype=np.int64) for _ in range(ne)]
                for m in methods}

    t0 = time.time()
    evaluated = 0

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if max_samples and evaluated >= max_samples:
                break
            imgs = batch[0].to(engine.device)
            tgts = batch[1].long().to(engine.device)
            tgt_np = tgts.cpu().numpy()

            outs = model(imgs)
            if not isinstance(outs, (list, tuple)):
                outs = [outs]

            aligned = []
            for j in range(ne):
                o = outs[j]
                if o.shape[-2:] != tgts.shape[-2:]:
                    o = F.interpolate(o, tgts.shape[-2:], mode='bilinear', align_corners=True)
                aligned.append(o)

            for s in range(ne):
                p_b = engine.predict_baseline(aligned, s).cpu().numpy()
                matrices['Baseline (Single Exit)'][s] += compute_confusion_matrix(p_b, tgt_np, nc)

                p_u = engine.predict_logit_voting(aligned, s).cpu().numpy()
                matrices['Logit Voting (Uniform)'][s] += compute_confusion_matrix(p_u, tgt_np, nc)

                p_w = engine.predict_logit_voting(aligned, s, engine.exit_weights).cpu().numpy()
                matrices['Logit Voting (Weighted)'][s] += compute_confusion_matrix(p_w, tgt_np, nc)

                p_bay = engine.predict_bayesian(aligned, s).cpu().numpy()
                matrices['Bayesian Updating'][s] += compute_confusion_matrix(p_bay, tgt_np, nc)

                p_cp = engine.predict_conformal_bayesian(aligned, s).cpu().numpy()
                matrices['Bayes + Conformal Bins'][s] += compute_confusion_matrix(p_cp, tgt_np, nc)

            evaluated += imgs.size(0)
            if (bi + 1) % 10 == 0 or evaluated == max_samples:
                print(f"  Processed {evaluated} samples ({time.time()-t0:.1f}s)...")

    # Print results
    print("\n" + "=" * 90)
    print("DECISION-CENTERED TEAMWORK RESULTS ON UNet++")
    print("=" * 90)

    use_dice = (nc == 2)
    metric_name = "mIoU" if not use_dice else "mIoU / Dice"
    header = f"{'Method':<28}"
    for s in range(ne):
        header += f" | {'Exit '+str(s+1)+' ('+metric_name+')':>20}"
    print(header)
    print("-" * len(header))

    csv_rows = []
    for m in methods:
        row = f"{m:<28}"
        for s in range(ne):
            miou, acc, _ = calculate_miou_from_matrix(matrices[m][s])
            if use_dice:
                d = dice_from_matrix(matrices[m][s])
                row += f" | {miou*100:>5.2f}% / {d*100:>5.2f}%"
            else:
                row += f" |    {miou*100:>6.2f}% ({acc*100:>5.1f}%)"
            csv_rows.append({
                'method': m, 'exit_stage': s + 1,
                'miou': miou, 'pixel_acc': acc,
                'dice': dice_from_matrix(matrices[m][s]) if use_dice else None,
            })
        print(row)
    print("=" * 90)

    # Highlight key comparison
    base4 = calculate_miou_from_matrix(matrices['Baseline (Single Exit)'][-1])[0]
    best_method = None
    best_miou4 = base4
    for m in methods[1:]:
        m4 = calculate_miou_from_matrix(matrices[m][-1])[0]
        if m4 > best_miou4:
            best_miou4 = m4
            best_method = m

    if best_method:
        diff = (best_miou4 - base4) * 100
        print(f"\n  -> TEAMWORK IMPROVED Exit 4: {best_method} "
              f"({best_miou4*100:.2f}% vs {base4*100:.2f}% baseline, +{diff:.2f}pp)")
    else:
        print(f"\n  -> No teamwork method improved over Exit 4 baseline ({base4*100:.2f}%)")

    if output_csv:
        d = os.path.dirname(os.path.abspath(output_csv))
        if d:
            os.makedirs(d, exist_ok=True)
        fields = ['method', 'exit_stage', 'miou', 'pixel_acc']
        if use_dice:
            fields.append('dice')
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in csv_rows:
                if not use_dice:
                    r.pop('dice', None)
                writer.writerows([r])
        print(f"[Teamwork] Results saved to: {output_csv}")


# ==============================================================================
# Self-Test
# ==============================================================================

class SyntheticUNetPPModel(nn.Module):
    """Mock 4-exit model simulating UNet++ balanced-quality exits."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        B, _, H, W = x.shape
        true_labels = x[:, 0].clamp(0, self.num_classes - 1).long()
        base = torch.zeros(B, self.num_classes, H, W, device=x.device)
        base.scatter_(1, true_labels.unsqueeze(1), 5.0)
        shared = torch.randn_like(base) * 2.0
        # UNet++ exits: similar quality (small noise spread)
        return [base + shared + torch.randn_like(base) * std
                for std in [3.5, 3.0, 2.5, 2.0]]


def run_selftest():
    print("\n" + "=" * 80)
    print("RUNNING SELF-TEST: Decision-Centered Teamwork (UNet++ 4-Exit)")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    nc = KVASIR_NUM_CLASSES  # binary
    model = SyntheticUNetPPModel(num_classes=nc).to(device)
    H, W = 128, 128

    def make_data(n, seed):
        torch.manual_seed(seed)
        out = []
        for _ in range(n):
            lbl = torch.randint(0, nc, (4, H, W))
            img = torch.randn(4, 3, H, W)
            img[:, 0] = lbl.float()
            out.append((img, lbl))
        return out

    calib = make_data(5, 100)
    evalu = make_data(10, 200)

    engine = TeamworkSegmentationEngine(num_classes=nc, num_exits=4, device=device)
    engine.fit_likelihood_matrices(calib, model)
    assert engine.log_likelihood_matrix is not None
    assert engine.log_likelihood_matrix.shape == (4, nc, nc)
    print("[OK] Likelihood matrices verified.")

    run_evaluation(evalu, model, engine, max_samples=40,
                   class_names=KVASIR_CLASS_NAMES)
    print("\n[OK] Self-test PASSED! All modules verified.")


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    if args.selftest:
        run_selftest()
        return

    if args.train:
        train_unetpp(args)
        return

    # ---- Evaluation mode ----
    if not args.model_file:
        print("[Error] --model_file required for evaluation.")
        print("  Train first: python test_teamwork_unetpp.py --train")
        print("  Self-test:   python test_teamwork_unetpp.py --selftest")
        sys.exit(1)

    device = args.device
    print(f"[Model] Loading: {args.model_file}")
    ckpt = torch.load(args.model_file, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        encoder = ckpt.get('encoder', args.encoder)
        nc = ckpt.get('num_classes', args.num_classes)
        dataset = ckpt.get('dataset', args.dataset)
        img_size = ckpt.get('img_size', args.img_size)
        sd = ckpt['state_dict']
        print(f"  Checkpoint info: encoder={encoder}, classes={nc}, "
              f"dataset={dataset}, img_size={img_size}")
    else:
        encoder, nc, dataset, img_size = args.encoder, args.num_classes, args.dataset, args.img_size
        sd = ckpt

    model = UNetPlusPlus(num_classes=nc, encoder=encoder, pretrained_encoder=False)
    cleaned = {}
    mk = model.state_dict()
    for k, v in sd.items():
        ck = k
        for pfx in ('module.', 'model.'):
            if ck.startswith(pfx):
                ck = ck[len(pfx):]
        if ck in mk and mk[ck].shape == v.shape:
            cleaned[ck] = v
    r = len(cleaned) / max(len(mk), 1)
    print(f"  Matched {len(cleaned)}/{len(mk)} tensors ({r*100:.1f}%)")
    if r < 0.9:
        print("[Error] Checkpoint mismatch!")
        sys.exit(1)
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()

    # Dataset
    if dataset == 'kvasir':
        data_root = args.data_root or 'data/kvasir-seg'
        download_kvasir_seg(data_root)
        n_all = len([f for f in os.listdir(os.path.join(data_root, 'images'))
                     if f.endswith('.jpg')])
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n_all)
        val_idx = perm[int(0.8*n_all):].tolist()
        val_ds = KvasirSEGDataset(data_root, indices=val_idx,
                                  img_size=img_size, augment=False)
        class_names = KVASIR_CLASS_NAMES
        ignore_label = KVASIR_IGNORE_LABEL
    else:
        data_root = args.data_root or 'data/cityscapes'
        val_ds = CityscapesDataset(data_root, 'val', img_size, augment=False)
        class_names = CITYSCAPES_CLASS_NAMES
        ignore_label = CITYSCAPES_IGNORE_LABEL

    # Calib/eval split
    n = len(val_ds)
    perm2 = np.random.RandomState(args.seed + 1).permutation(n)
    n_cal = max(1, int(n * args.calib_ratio))
    cal_idx = perm2[:n_cal].tolist()
    eval_idx = perm2[n_cal:].tolist()
    print(f"[Split] {n} val images -> {len(cal_idx)} calib / {len(eval_idx)} eval")

    lkw = dict(batch_size=args.eval_batch_size, shuffle=False,
               num_workers=args.num_workers, pin_memory=True)
    cal_loader = DataLoader(Subset(val_ds, cal_idx), **lkw)
    eval_loader = DataLoader(Subset(val_ds, eval_idx), **lkw)

    engine = TeamworkSegmentationEngine(num_classes=nc, num_exits=4, device=device)

    if args.bayes_matrix_file and os.path.exists(args.bayes_matrix_file):
        engine.load_likelihoods(args.bayes_matrix_file)
    else:
        engine.fit_likelihood_matrices(cal_loader, model)
        if args.bayes_matrix_file:
            engine.save_likelihoods(args.bayes_matrix_file)

    csv_out = args.output_csv or f'results_teamwork_unetpp_{dataset}.csv'
    run_evaluation(eval_loader, model, engine,
                   output_csv=csv_out, max_samples=args.max_samples,
                   class_names=class_names)


if __name__ == '__main__':
    main()

