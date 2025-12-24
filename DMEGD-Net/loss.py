import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class CombinedEdgeAndContentLoss(nn.Module):
    def __init__(self, edge_weight, content_weight, reduction='mean'):
        super(CombinedEdgeAndContentLoss, self).__init__()
        self.edge_weight = edge_weight
        self.content_weight = content_weight
        self.reduction = reduction

        self.edge_detector = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, prediction, target):
        content_loss = F.mse_loss(prediction, target, reduction=self.reduction)
        target_edges = self.edge_detector(target)
        pred_edges = self.edge_detector(prediction)
        edge_loss = F.mse_loss(pred_edges, target_edges, reduction=self.reduction)
        total_loss = self.edge_weight * edge_loss + self.content_weight * content_loss
        return total_loss

