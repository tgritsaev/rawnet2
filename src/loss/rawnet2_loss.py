import torch
import torch.nn as nn


class RawNet2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weighted_ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor([1, 9]))

    def forward(self, pred, target, **kwargs):
        assert pred.shape == target.shape
        return {"loss": self.weighted_ce_loss(pred, target)}
