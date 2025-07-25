"""
    @file:              losses.py
    @Author:            Ihssene Brahimi, Moustafa Amine Bezzahi

    @Creation Date:     06/2024
    @Last modification: 07/2024

    @Description:       This file is used to define the losses.
"""

import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        if inputs.dim() > 1:
            targets = targets.view(-1, 1)
            targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets, 1)
        else:
            targets_one_hot = targets

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()