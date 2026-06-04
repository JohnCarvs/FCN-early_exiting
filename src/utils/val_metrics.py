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
