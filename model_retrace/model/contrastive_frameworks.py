import torch
import torch.nn as nn

from lightly.loss import NTXentLoss
from lightly.models.modules import NNCLRProjectionHead




class SimCLR(nn.Module):
    def __init__(self, encoder, nemb, nout, temperature = 0.07):
        super().__init__()
        self.temperature = temperature
        self.backbone = encoder
        self.projection_head = NNCLRProjectionHead(nemb, 2048, nemb)
        self.criterion = NTXentLoss(temperature=temperature)

    def forward_enc(self, x):
        x = self.backbone(**x)
        z = self.projection_head(x)
        return z

    def forward(self, x1, x2):
        z1 = self.forward_enc(x1)
        z2 = self.forward_enc(x2)
        loss = self.criterion(z1, z2)
        return loss

    def encode(self, x):
        return self.backbone(**x)
 