import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        in_dim: int,
        hidden_size: int,
        kernel_size: int,
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        pad_left = (kernel_size - 1) // 2
        pad_right = (kernel_size - 1) - pad_left

        self.input_layer = nn.Sequential(
            nn.ConstantPad1d((pad_left, pad_right), 0.0),
            nn.Conv1d(in_dim, hidden_size, kernel_size, padding=0),
            nn.GroupNorm(1, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.hidden = nn.ModuleList([
            nn.Sequential(
                nn.ConstantPad1d((pad_left, pad_right), 0.0),
                nn.Conv1d(hidden_size, hidden_size,
                          kernel_size, padding=0),
                nn.GroupNorm(1, hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_hidden_layers)
        ])

        self.post_skip = nn.SiLU()
        self.output_head = nn.Conv1d(hidden_size, out_dim, kernel_size=1)

    def forward(self, y: torch.Tensor, c: torch.Tensor):
        # y: [B, pred]
        # c: [B, L, H]
        B = y.shape[0]
        o = torch.cat([y.unsqueeze(1), c.view(B, -1)])  # [B, in_dim]
        out = self.input_layer(o)
        for block in self.hidden:
            out = self.post_skip(block(out) + out)
        out = self.output_head(out)  # [B, out_dim]
        return out


class RandomPermute1D(nn.Module):
    def __init__(self, N: int, seed: int | None = None):
        super().__init__()
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
            perm = torch.randperm(N, generator=g)
        else:
            perm = torch.randperm(N)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(N)

        self.register_buffer("perm", perm)     # [N]
        self.register_buffer("inv",  inv)      # [N]

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: [B, N] (or [..., N]); permutes last dim
        return y.index_select(dim=-1, index=self.perm)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y.index_select(dim=-1, index=self.inv)


class CouplingLayer(nn.Module):
    def __init__(self, s_net: torch.Tensor, t_net: torch.Tensor, mask: torch.Tensor):
        super().__init__()

        self.s_net = s_net
        self.t_net = t_net

        self.mask = mask

    def forward(self, y: torch.Tensor, c: torch.Tensor):
        y_keep = self.mask * y
        s = self.s_net(y, c)
        t = self.t_net(y, c)

        y = y_keep + (1 - self.mask) * (y * torch.exp(s) + t)
        logdet = ((1 - self.mask) * s).sum(dim=-1)

        return y, logdet

    def reverse(self, y: torch.Tensor, c: torch.Tensor):
        y_keep = y * self.mask
        s = self.s_net(y, c)
        t = self.t_net(y, c)

        y = y_keep + (1 - self.mask) * ((y - t) * torch.exp(-s))

        return y