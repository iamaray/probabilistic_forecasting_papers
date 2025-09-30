import torch
import torch.nn as nn
import torch.nn.functional as F


class RealNVP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        if self.train():
            # forward pass
            pass
        elif self.eval():
            # reverse pass
            pass

    def negative_log_likelihood(self):
        pass
