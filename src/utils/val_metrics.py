import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConfusionMatrix:

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def reset(self):
        self.matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        

    def update(self, preds, targets):
        preds = preds.detach().cpu().numpy().astype(np.int64)
        targets = targets.detach().cpu().numpy().astype(np.int64)

        # for each coordinate in (targets, preds), sum '1'
        np.add.at(
            self.matrix,
            (targets, preds),
            1
        )

        

    def mean_pixel_acc(self):
        true_positives = np.diag(self.matrix)
        total_per_class = np.sum(self.matrix, axis=1)
        
        acc_per_class = (
            true_positives /
            np.maximum(total_per_class, 1)  # avoid division by 0
        )
        return(np.mean(acc_per_class))

