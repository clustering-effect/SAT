import torch.nn as nn
import torch
import numpy as np


class feature_loss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0, p: int or np.inf = 2) -> None:
        super().__init__(weight, size_average, ignore_index,
                         reduce, reduction, label_smoothing)
        self.p = p
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor, mixed_features: torch.Tensor):
        output = output.unsqueeze(dim=1)
        mixed_features = mixed_features.unsqueeze(dim=0)
        path = output - mixed_features
        dis = -torch.norm(path, dim=-1).pow(2)/9
        loss = self.loss_func(dis, target)

        return loss
