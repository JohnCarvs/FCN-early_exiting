import os
import numpy as np
import torch

def compute_class_weights(dataset, n_classes=21, cache_path=None):
    if cache_path is not None and os.path.exists(cache_path):
        weights = np.load(cache_path)
        print(f"Loaded class weights from {cache_path}")
        return torch.tensor(weights, dtype=torch.float32)
    
    counts = np.zeros(n_classes, dtype=np.float64)
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        target_np = target.numpy() if hasattr(target, 'numpy') else np.asarray(target)
        valid = target_np[(target_np >= 0) & (target_np < n_classes)]

        if valid.size == 0:
            continue    
        counts += np.bincount(valid.reshape(-1), minlength=n_classes)[:n_classes]

    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.5, 5.0)

    if cache_path is not None:
        np.save(cache_path, weights)
        print(f"Saved class weights to {cache_path}")

    return torch.tensor(weights, dtype=torch.float32)