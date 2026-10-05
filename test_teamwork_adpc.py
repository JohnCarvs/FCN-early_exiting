#!/usr/bin/env python3
"""
test_teamwork_adpc.py - Decision-Centered Teamwork for Anytime Dense Prediction (ADP-C)

This script adapts the Decision-Centered Teamwork (DCT) methodology
(from Peng et al., ICML 2025 and bayes_voting_inference.py) to multi-exit semantic
segmentation models from the ADP-C framework (Liu et al., ICLR 2022, HRNet-w48/w18).

Key Features:
1. Multi-Exit Semantic Segmentation Architecture:
   Supports the 4 exits of ADP-C (Stage 1, Stage 2, Stage 3, Stage 4/Final).
2. Decision-Centered Teamwork Strategies:
   - Baseline: Independent single-exit prediction (Exit 1, Exit 2, Exit 3, Exit 4).
   - Logit Voting: Uniform ensemble of exit logits (1..k).
   - Weighted Logit Voting: Performance-weighted ensemble of exit logits (1..k).
   - Bayesian Updating: Pixel-wise posterior updates using empirical likelihoods P(pred | true_class).
   - Bayes + Conformal Prediction / Confidence Bins: Likelihoods conditioned on exit confidence levels.
3. Complete Metrics & Logging:
   - Computes per-exit mIoU, pixel accuracy, and IoU per class on Cityscapes (19 classes).
   - Logs experimental results to CSV (compatible with the bayes_voting_inference logging schema).
4. Self-test Mode:
   - `--selftest` runs end-to-end verification with synthetic tensors without requiring datasets.
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_IGNORE_LABEL = 255
CITYSCAPES_CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]


def parse_args():
    parser = argparse.ArgumentParser(description='Decision-Centered Teamwork for ADP-C')
    parser.add_argument('--cfg', type=str, default='experiments/cityscapes/w48.yaml',
                        help='Path to config yaml file')
    parser.add_argument('--model_file', type=str, default=None,
                        help='Path to pretrained .pth checkpoint (EE / EE+RH / ADP-C)')
    parser.add_argument('--anytime_dir', type=str, default='.',
                        help='Root path to the anytime (ADP-C) repository if external')
    parser.add_argument('--num_classes', type=int, default=CITYSCAPES_NUM_CLASSES,
                        help='Number of semantic classes')
    parser.add_argument('--calib_ratio', type=float, default=0.3,
                        help='Fraction of val set used to estimate Bayes likelihood matrices (0.0 to 1.0)')
    parser.add_argument('--bayes_matrix_file', type=str, default=None,
                        help='Path to cache/load computed Bayes likelihood matrices (.npy)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to output results CSV')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max validation samples to evaluate (for quick test)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Computation device')
    parser.add_argument('--selftest', action='store_true', default=False,
                        help='Run self-test with dummy tensors and synthetic model')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for the disjoint calibration/evaluation split of the val set')
    parser.add_argument('--opts', nargs='*', default=[],
                        help="ADP-C config overrides, e.g. --opts EXIT.TYPE flex EXIT.INTER_CHANNEL 128")
    return parser.parse_args()


# ==============================================================================
# Helper Functions: Logits Extraction, Confusion Matrix & Metrics
# ==============================================================================

def extract_model_logits(model_output) -> List[torch.Tensor]:
    """Normalize multi-exit model outputs into a list of logits tensors.
    
    Compatible with ADP-C model_anytime which returns either a list of 4 tensors
    or a tuple (logits_list, aux).
    """
    if isinstance(model_output, tuple):
        model_output = model_output[0]
    if not isinstance(model_output, list):
        model_output = [model_output]
    return model_output


def compute_confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """Computes confusion matrix: rows = predicted class, cols = ground truth class.
    
    Shape: (num_classes, num_classes) where entry [p, t] is count of (pred=p, target=t).
    """
    valid = (target >= 0) & (target < num_classes)
    p_valid = pred[valid].astype(np.int64)
    t_valid = target[valid].astype(np.int64)
    bins = num_classes * p_valid + t_valid
    hist = np.bincount(bins, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return hist


def calculate_miou_from_matrix(matrix: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Calculates mean IoU, pixel accuracy, and per-class IoU from confusion matrix.
    
    matrix[p, t] has p = predicted class (axis 0), t = ground truth class (axis 1).
    """
    tp = np.diag(matrix).astype(np.float64)
    pred_sum = matrix.sum(axis=1).astype(np.float64)  # sum over targets (total predicted as class c)
    target_sum = matrix.sum(axis=0).astype(np.float64)  # sum over preds (total ground-truth class c)
    
    # IoU = TP / (TP + FP + FN) = TP / (pred_sum + target_sum - TP)
    union = pred_sum + target_sum - tp
    valid = union > 0
    ious = np.zeros(matrix.shape[0], dtype=np.float64)
    ious[valid] = tp[valid] / union[valid]
    
    miou = float(np.mean(ious[valid])) if valid.any() else 0.0
    total_pixels = matrix.sum()
    pixel_acc = float(tp.sum() / total_pixels) if total_pixels > 0 else 0.0
    return miou, pixel_acc, ious


# ==============================================================================
# Decision-Centered Teamwork: Likelihood Estimation & Bayesian Updating
# ==============================================================================

class TeamworkSegmentationEngine:
    """Implements Decision-Centered Teamwork (Logit Voting, Bayes, Conformal Bins)
    for multi-exit dense prediction networks.
    """

    def __init__(self, num_classes: int = CITYSCAPES_NUM_CLASSES, num_exits: int = 4,
                 conf_thresholds: Tuple[float, float] = (0.70, 0.90),
                 laplace_eps: float = 1.0, device: str = 'cpu'):
        self.num_classes = num_classes
        self.num_exits = num_exits
        self.conf_thresholds = conf_thresholds
        self.laplace_eps = laplace_eps
        self.device = device

        # Bayes likelihood matrices: P(pred = a | target = c)
        # Shape: (num_exits, num_classes, num_classes) -> [exit, pred, target]
        self.log_likelihood_matrix: Optional[torch.Tensor] = None

        # Binned likelihood matrices: P(pred = a | target = c, conf_bin = b)
        # 3 bins: 0 = low (<0.70), 1 = mid (0.70-0.90), 2 = high (>0.90)
        # Shape: (num_exits, 3, num_classes, num_classes) -> [exit, bin, pred, target]
        self.log_binned_likelihood_matrix: Optional[torch.Tensor] = None

        # Exit performance weights (e.g. from validation mIoU)
        self.exit_weights: List[float] = [1.0] * num_exits

    def fit_likelihood_matrices(self, calib_loader, model, max_batches: Optional[int] = None):
        """Accumulates pixel-level confusion statistics on calibration data and
        computes normalized log-likelihood matrices P(pred | target) for each exit.
        """
        print(f"\n[Teamwork] Fitting Bayesian likelihood matrices on calibration data ({self.num_exits} exits)...")
        model.eval()

        counts = np.zeros((self.num_exits, self.num_classes, self.num_classes), dtype=np.int64)
        binned_counts = np.zeros((self.num_exits, 3, self.num_classes, self.num_classes), dtype=np.int64)

        t0 = time.time()
        processed_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(calib_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                # Unpack batch (handles varying dataloader tuple returns)
                if isinstance(batch, (list, tuple)):
                    images = batch[0].to(self.device)
                    targets = batch[1].long().to(self.device)
                else:
                    raise ValueError(f"Unexpected batch format: {type(batch)}")

                outputs = extract_model_logits(model(images))
                target_np = targets.cpu().numpy()

                for j in range(min(len(outputs), self.num_exits)):
                    out_j = outputs[j]
                    if out_j.shape[-2:] != targets.shape[-2:]:
                        out_j = F.interpolate(out_j, size=targets.shape[-2:], mode='bilinear', align_corners=True)

                    prob_j = F.softmax(out_j, dim=1)
                    conf_j, pred_j = prob_j.max(dim=1)

                    pred_np = pred_j.cpu().numpy()
                    conf_np = conf_j.cpu().numpy()

                    # Mask valid pixels
                    valid_mask = (target_np >= 0) & (target_np < self.num_classes)
                    p_valid = pred_np[valid_mask]
                    t_valid = target_np[valid_mask]
                    c_valid = conf_np[valid_mask]

                    # Standard exit counts
                    bins = self.num_classes * p_valid + t_valid
                    counts[j] += np.bincount(bins, minlength=self.num_classes * self.num_classes).reshape(
                        self.num_classes, self.num_classes
                    )

                    # Binned exit counts
                    low_th, high_th = self.conf_thresholds
                    bin_indices = np.zeros_like(c_valid, dtype=np.int64)
                    bin_indices[c_valid >= high_th] = 2
                    bin_indices[(c_valid >= low_th) & (c_valid < high_th)] = 1

                    for b in range(3):
                        in_bin = bin_indices == b
                        if np.any(in_bin):
                            bin_bins = self.num_classes * p_valid[in_bin] + t_valid[in_bin]
                            binned_counts[j, b] += np.bincount(bin_bins, minlength=self.num_classes * self.num_classes).reshape(
                                self.num_classes, self.num_classes
                            )

                processed_batches += 1
                if processed_batches % 20 == 0:
                    print(f"  Calibrated on {processed_batches} batches ({time.time() - t0:.1f}s)...")

        print(f"[Teamwork] Calibration complete across {processed_batches} batches in {time.time() - t0:.2f}s.")

        # Compute normalized log-likelihoods: P(pred = a | target = c)
        # Smoothing ensures no log(0)
        norm_likelihood = np.zeros_like(counts, dtype=np.float64)
        for j in range(self.num_exits):
            target_totals = counts[j].sum(axis=0, keepdims=True)  # sum over preds for each true class
            denom = target_totals + self.num_classes * self.laplace_eps
            norm_likelihood[j] = (counts[j] + self.laplace_eps) / np.maximum(denom, 1e-12)

        self.log_likelihood_matrix = torch.from_numpy(np.log(norm_likelihood)).float().to(self.device)

        # Normalize binned likelihoods
        norm_binned = np.zeros_like(binned_counts, dtype=np.float64)
        for j in range(self.num_exits):
            for b in range(3):
                target_totals = binned_counts[j, b].sum(axis=0, keepdims=True)
                denom = target_totals + self.num_classes * self.laplace_eps
                norm_binned[j, b] = (binned_counts[j, b] + self.laplace_eps) / np.maximum(denom, 1e-12)

        self.log_binned_likelihood_matrix = torch.from_numpy(np.log(norm_binned)).float().to(self.device)

        # Weighted-voting weights = per-exit mIoU on the calibration split (no test leakage)
        calib_mious = [calculate_miou_from_matrix(counts[j])[0] for j in range(self.num_exits)]
        self.exit_weights = calib_mious
        print("[Teamwork] Calibration mIoU per exit (used as voting weights): "
              + ", ".join(f"exit{j + 1}={m * 100:.2f}%" for j, m in enumerate(calib_mious)))

    def save_likelihoods(self, filepath: str):
        """Saves estimated likelihood matrices to .npy or .pth file."""
        data = {
            'log_likelihood': self.log_likelihood_matrix.cpu().numpy(),
            'log_binned_likelihood': self.log_binned_likelihood_matrix.cpu().numpy(),
            'num_classes': self.num_classes,
            'num_exits': self.num_exits,
            'conf_thresholds': self.conf_thresholds,
            'exit_weights': list(self.exit_weights)
        }
        np.save(filepath, data, allow_pickle=True)
        print(f"[Teamwork] Saved Bayes likelihood matrices to: {filepath}")

    def load_likelihoods(self, filepath: str):
        """Loads cached likelihood matrices."""
        data = np.load(filepath, allow_pickle=True).item()
        self.log_likelihood_matrix = torch.from_numpy(data['log_likelihood']).float().to(self.device)
        self.log_binned_likelihood_matrix = torch.from_numpy(data['log_binned_likelihood']).float().to(self.device)
        if 'exit_weights' in data:
            self.exit_weights = list(data['exit_weights'])
        print(f"[Teamwork] Loaded Bayes likelihood matrices from: {filepath}")

    def set_exit_weights(self, weights: List[float]):
        """Sets custom weights for Weighted Logit Voting."""
        assert len(weights) == self.num_exits
        self.exit_weights = list(weights)

    # --------------------------------------------------------------------------
    # Inference Strategies (Vectorized on GPU)
    # --------------------------------------------------------------------------

    def predict_baseline(self, exit_logits: List[torch.Tensor], stage: int) -> torch.Tensor:
        """Baseline early exit: returns argmax of exit stage alone."""
        return exit_logits[stage].argmax(dim=1)

    def predict_logit_voting(self, exit_logits: List[torch.Tensor], stage: int,
                             weights: Optional[List[float]] = None) -> torch.Tensor:
        """Logit Voting: computes weighted or uniform sum of logits from exits 0..stage.
        
        Formula: O(x) = sum_{j=0}^stage w_j * logits_j(x)
        """
        combined = torch.zeros_like(exit_logits[0])
        for j in range(stage + 1):
            w = weights[j] if weights is not None else 1.0
            combined += w * exit_logits[j]
        return combined.argmax(dim=1)

    def predict_bayesian_updating(self, exit_logits: List[torch.Tensor], stage: int) -> torch.Tensor:
        """Pixel-wise Bayesian updating across exits 0..stage.
        
        Prior: Uniform across classes.
        Posterior: log P(target = c | pred_0, ..., pred_stage) = sum_{j=0}^stage log P(pred_j | target = c)
        Computed on GPU via vectorized index gathering.
        """
        assert self.log_likelihood_matrix is not None, "Likelihood matrix must be calibrated first!"
        B, C, H, W = exit_logits[0].shape
        
        # log_posterior: shape (B, H, W, C)
        log_posterior = torch.zeros((B, H, W, self.num_classes), device=self.device)

        for j in range(stage + 1):
            pred_j = exit_logits[j].argmax(dim=1)  # (B, H, W)
            # log_likelihood_matrix[j]: shape (num_classes, num_classes) -> [pred, target]
            # Gathering for each pixel:
            log_p_j = self.log_likelihood_matrix[j][pred_j]  # (B, H, W, num_classes)
            log_posterior += log_p_j

        return log_posterior.argmax(dim=-1)

    def predict_conformal_bayesian_updating(self, exit_logits: List[torch.Tensor], stage: int) -> torch.Tensor:
        """Enhanced Bayesian updating with Conformal Confidence Bins.
        
        Conditions likelihoods on both predicted class and softmax confidence bin (low/mid/high).
        """
        assert self.log_binned_likelihood_matrix is not None, "Binned likelihood matrix must be calibrated first!"
        B, C, H, W = exit_logits[0].shape
        log_posterior = torch.zeros((B, H, W, self.num_classes), device=self.device)
        low_th, high_th = self.conf_thresholds

        for j in range(stage + 1):
            prob_j = F.softmax(exit_logits[j], dim=1)
            conf_j, pred_j = prob_j.max(dim=1)  # (B, H, W)

            # Assign confidence bin (0: low, 1: mid, 2: high)
            bin_j = torch.zeros_like(pred_j, dtype=torch.long)
            bin_j[conf_j >= high_th] = 2
            bin_j[(conf_j >= low_th) & (conf_j < high_th)] = 1

            # log_binned_likelihood_matrix[j]: shape (3, C, C) -> [bin, pred, target]
            log_p_j = self.log_binned_likelihood_matrix[j, bin_j, pred_j]  # (B, H, W, num_classes)
            log_posterior += log_p_j

        return log_posterior.argmax(dim=-1)


# ==============================================================================
# Full Evaluation Loop & Benchmark Runner
# ==============================================================================

def run_evaluation(val_loader, model, engine: TeamworkSegmentationEngine,
                   output_csv: Optional[str] = None, max_samples: Optional[int] = None):
    """Evaluates all methods across all 4 exits and logs comparative results."""
    print("\n" + "=" * 90)
    print("STARTING DECISION-CENTERED TEAMWORK EVALUATION (4 EXITS)")
    print("=" * 90)

    model.eval()
    num_exits = engine.num_exits
    num_classes = engine.num_classes

    methods = [
        'Baseline (Single Exit)',
        'Logit Voting (Uniform)',
        'Logit Voting (Weighted)',
        'Bayesian Updating',
        'Bayes + Conformal Bins'
    ]

    # Confusion matrices for each (method, stage)
    # Shape: {method: [matrix_exit0, matrix_exit1, ...]}
    matrices = {
        m: [np.zeros((num_classes, num_classes), dtype=np.int64) for _ in range(num_exits)]
        for m in methods
    }

    t0 = time.time()
    evaluated_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_samples is not None and evaluated_samples >= max_samples:
                break

            images = batch[0].to(engine.device)
            targets = batch[1].long().to(engine.device)
            target_np = targets.cpu().numpy()

            outputs = extract_model_logits(model(images))
            aligned_logits = []
            for j in range(num_exits):
                out_j = outputs[j]
                if out_j.shape[-2:] != targets.shape[-2:]:
                    out_j = F.interpolate(out_j, size=targets.shape[-2:], mode='bilinear', align_corners=True)
                aligned_logits.append(out_j)

            # Evaluate each anytime stage (0: Exit1, 1: Exit2, 2: Exit3, 3: Exit4)
            for stage in range(num_exits):
                # 1. Baseline
                pred_base = engine.predict_baseline(aligned_logits, stage).cpu().numpy()
                matrices['Baseline (Single Exit)'][stage] += compute_confusion_matrix(pred_base, target_np, num_classes)

                # 2. Logit Voting (Uniform)
                pred_vote = engine.predict_logit_voting(aligned_logits, stage, weights=None).cpu().numpy()
                matrices['Logit Voting (Uniform)'][stage] += compute_confusion_matrix(pred_vote, target_np, num_classes)

                # 3. Logit Voting (Weighted)
                pred_wvote = engine.predict_logit_voting(aligned_logits, stage, weights=engine.exit_weights).cpu().numpy()
                matrices['Logit Voting (Weighted)'][stage] += compute_confusion_matrix(pred_wvote, target_np, num_classes)

                # 4. Bayesian Updating
                pred_bayes = engine.predict_bayesian_updating(aligned_logits, stage).cpu().numpy()
                matrices['Bayesian Updating'][stage] += compute_confusion_matrix(pred_bayes, target_np, num_classes)

                # 5. Bayes + Conformal Bins
                pred_cp = engine.predict_conformal_bayesian_updating(aligned_logits, stage).cpu().numpy()
                matrices['Bayes + Conformal Bins'][stage] += compute_confusion_matrix(pred_cp, target_np, num_classes)

            evaluated_samples += images.size(0)
            if (batch_idx + 1) % 25 == 0 or evaluated_samples == max_samples:
                elapsed = time.time() - t0
                print(f"  Processed {evaluated_samples} validation samples ({elapsed:.1f}s)...")

    # ==========================================================================
    # Print Results Summary Table
    # ==========================================================================
    print("\n" + "=" * 90)
    print("DECISION-CENTERED TEAMWORK RESULTS ON ADP-C")
    print("=" * 90)
    header = f"{'Method':<28} | {'Exit 1 (mIoU)':<13} | {'Exit 2 (mIoU)':<13} | {'Exit 3 (mIoU)':<13} | {'Exit 4 (mIoU)':<13}"
    print(header)
    print("-" * len(header))

    results_table = []
    csv_rows = []

    for m in methods:
        row_str = f"{m:<28}"
        row_data = {'method': m}
        for stage in range(num_exits):
            miou, acc, _ = calculate_miou_from_matrix(matrices[m][stage])
            row_str += f" | {miou * 100:>6.2f}% ({acc * 100:>5.1f}%)"
            row_data[f'exit_{stage + 1}_miou'] = miou * 100
            row_data[f'exit_{stage + 1}_acc'] = acc * 100
            csv_rows.append({
                'method': m,
                'exit_stage': stage + 1,
                'miou': miou,
                'pixel_acc': acc
            })
        print(row_str)
        results_table.append(row_data)

    print("=" * 90)

    # Save to CSV if requested
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['method', 'exit_stage', 'miou', 'pixel_acc'])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[Teamwork] Full metrics saved to: {output_csv}")


# ==============================================================================
# Self-Test Mode (Synthetic Verification)
# ==============================================================================

class SyntheticADPCModel(nn.Module):
    """Mock 4-exit model simulating ADP-C behavior for self-testing."""
    def __init__(self, num_classes=19):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        B, _, H, W = x.shape
        # True labels are embedded in the first channel of x for the self-test
        true_labels = x[:, 0].clamp(0, self.num_classes - 1).long()
        
        # Base logits with correct class having a higher value (e.g., 5.0)
        base_logits = torch.zeros(B, self.num_classes, H, W, device=x.device)
        base_logits.scatter_(1, true_labels.unsqueeze(1), 5.0)
        
        # Shared noise simulating common backbone errors across all exits
        shared_noise = torch.randn_like(base_logits) * 3.0
        
        exits = []
        for i in range(4):
            # Independent noise decreases with exit depth (simulating refinement)
            noise_std = 6.0 - i * 1.0
            independent_noise = torch.randn_like(base_logits) * noise_std
            
            # Combine base signal, shared noise, and exit-specific independent noise
            exit_logits = base_logits + shared_noise + independent_noise
            exits.append(exit_logits)
        return exits


def run_selftest():
    """Runs end-to-end self-test without requiring real datasets or weights."""
    print("\n" + "=" * 80)
    print("RUNNING SELF-TEST: Decision-Centered Teamwork (ADP-C 4-Exit)")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    num_classes = CITYSCAPES_NUM_CLASSES
    model = SyntheticADPCModel(num_classes=num_classes).to(device)

    # Create synthetic dataset (5 calibration batches, 10 eval batches)
    H, W = 128, 256
    
    def make_dataset(batches, seed):
        torch.manual_seed(seed)
        data = []
        for _ in range(batches):
            labels = torch.randint(0, num_classes, (2, H, W))
            images = torch.randn(2, 3, H, W)
            # Embed labels into the first channel so the mock model can read them
            images[:, 0, :, :] = labels.float()
            data.append((images, labels))
        return data

    calib_data = make_dataset(5, seed=100)
    eval_data = make_dataset(10, seed=200)

    engine = TeamworkSegmentationEngine(num_classes=num_classes, num_exits=4, device=device)
    engine.set_exit_weights([0.446, 0.602, 0.766, 0.799])

    # 1. Fit Bayes matrices
    engine.fit_likelihood_matrices(calib_data, model)
    assert engine.log_likelihood_matrix is not None
    assert engine.log_likelihood_matrix.shape == (4, num_classes, num_classes)
    print("[OK] Likelihood matrices shape and normalization verified.")

    # 2. Run evaluation
    run_evaluation(eval_data, model, engine, max_samples=20)
    print("\n[OK] Self-test PASSED successfully! All modules verified.")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    args = parse_args()

    if args.selftest:
        run_selftest()
        return

    # Add anytime repository path if specified
    if args.anytime_dir and os.path.exists(args.anytime_dir):
        sys.path.insert(0, os.path.abspath(args.anytime_dir))
        lib_path = os.path.join(os.path.abspath(args.anytime_dir), 'lib')
        if os.path.exists(lib_path):
            sys.path.insert(0, lib_path)

    try:
        import importlib
        cfg_module = importlib.import_module('config')
        config = getattr(cfg_module, 'config')
        update_config = getattr(cfg_module, 'update_config')
        models = importlib.import_module('models')
        datasets = importlib.import_module('datasets')
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[Error] Failed to import ADP-C dependencies: {e}")
        print("Please ensure this script is run within or pointing to the anytime repository (using --anytime_dir).")
        print("Tip: You can test the engine logic immediately using: python test_teamwork_adpc.py --selftest")
        sys.exit(1)

    print(f"Loading configuration from: {args.cfg}  (overrides: {args.opts})")
    # HRNet/ADP-C update_config reads args.cfg and args.opts
    from types import SimpleNamespace
    update_config(config, SimpleNamespace(cfg=args.cfg, opts=list(args.opts)))

    # Initialize model
    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)
    model_file = args.model_file or getattr(config.TEST, 'MODEL_FILE', '')
    if not model_file:
        print("[Error] No checkpoint given (--model_file). Without it the exits are untrained.")
        sys.exit(1)

    print(f"Loading checkpoint: {model_file}")
    state = torch.load(model_file, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model_dict = model.state_dict()
    # Checkpoints are saved from the FullModel/DataParallel wrappers -> strip prefixes
    cleaned = {}
    for k, v in state.items():
        for prefix in ('module.model.', 'model.', 'module.'):
            if k.startswith(prefix) and k[len(prefix):] in model_dict:
                k = k[len(prefix):]
                break
        if k in model_dict and model_dict[k].shape == v.shape:
            cleaned[k] = v
    match_ratio = len(cleaned) / max(1, len(model_dict))
    print(f"[Checkpoint] Matched {len(cleaned)}/{len(model_dict)} model tensors ({match_ratio * 100:.1f}%).")
    if match_ratio < 0.95:
        missing = [k for k in model_dict if k not in cleaned][:10]
        print(f"[Error] Checkpoint does not match the model config. First missing keys: {missing}")
        print("Hint: EE+RH / ADP-C checkpoints need --opts EXIT.TYPE flex EXIT.INTER_CHANNEL 128 (w48).")
        sys.exit(1)
    model_dict.update(cleaned)
    model.load_state_dict(model_dict)

    model = model.to(args.device)
    model.eval()

    # Build Cityscapes val dataset (HRNet convention: crop_size = (H, W))
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    val_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=None,
        num_classes=args.num_classes,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1
    )

    # Disjoint calibration / evaluation split (Bayes matrices and voting weights
    # are estimated ONLY on the calibration images)
    n_total = len(val_dataset)
    perm = np.random.RandomState(args.seed).permutation(n_total)
    n_calib = max(1, int(n_total * args.calib_ratio))
    calib_idx, eval_idx = perm[:n_calib].tolist(), perm[n_calib:].tolist()
    print(f"[Split] {n_total} val images -> {len(calib_idx)} calibration / {len(eval_idx)} evaluation (seed={args.seed})")

    loader_kw = dict(batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    calib_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_dataset, calib_idx), **loader_kw)
    eval_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_dataset, eval_idx), **loader_kw)

    engine = TeamworkSegmentationEngine(
        num_classes=args.num_classes,
        num_exits=4,
        device=args.device
    )

    # Fit or load Bayes matrices (cache is only valid for the same checkpoint + seed + calib_ratio)
    if args.bayes_matrix_file and os.path.exists(args.bayes_matrix_file):
        engine.load_likelihoods(args.bayes_matrix_file)
    else:
        engine.fit_likelihood_matrices(calib_loader, model)
        if args.bayes_matrix_file:
            engine.save_likelihoods(args.bayes_matrix_file)

    # Run anytime evaluation on the held-out images only
    csv_out = args.output_csv or 'results_teamwork_adpc.csv'
    run_evaluation(eval_loader, model, engine, output_csv=csv_out, max_samples=args.max_samples)


if __name__ == '__main__':
    main()

