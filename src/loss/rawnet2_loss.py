import torch
import torch.nn as nn


class RawNet2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.weighted_ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor([1, 9]))
        self.weighted_ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target, **kwargs):
        return {"loss": self.weighted_ce_loss(pred, target)}
