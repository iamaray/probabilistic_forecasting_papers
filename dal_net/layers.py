import math
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmb(nn.Module):
    def __init__(self, device=None, dtype=torch.float32):
        super().__init__()
        self.device = device or "cpu"
        self.dtype = dtype

        exponents = torch.linspace(
            0, 4, steps=32, device=device, dtype=dtype)         # (32,)
        base = torch.tensor(10.0, device=device, dtype=dtype)
        # 10^{i*4/31}
        self.freqs = torch.pow(base, exponents)

    def forward(self, t: int):
        t_val = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        angles = t_val * self.freqs

        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=0)


def t_emb(t: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    t_emb(t) = [sin(10^{(0*4)/31} t), ..., sin(10^{(31*4)/31} t),
                cos(10^{(0*4)/31} t), ..., cos(10^{(31*4)/31} t)]
    Returns a tensor of shape (64,).
    """
    device = device or "cpu"
    exponents = torch.linspace(
        0, 4, steps=32, device=device, dtype=dtype)         # (32,)
    base = torch.tensor(10.0, device=device, dtype=dtype)
    # 10^{i*4/31}
    freqs = torch.pow(base, exponents)
    t_val = torch.as_tensor(t, device=device, dtype=dtype)
    # (32,)
    angles = t_val * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=0)


ScaleType = Literal["global", "window", "dilated"]


@dataclass
class HeadSpec:
    kind: ScaleType
    window: Optional[int] = None
    dilation: Optional[int] = None


def _build_head_mask(seq_len: int, spec: HeadSpec, device: torch.device) -> torch.Tensor:
    if spec.kind == "global":
        return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

    i = torch.arange(seq_len, device=device)
    j = torch.arange(seq_len, device=device)

    D = (i[:, None] - j[None, :]).abs()  # [N, N]

    if spec.kind == "window":
        assert spec.window is not None and spec.window >= 0
        return (D <= spec.window)

    if spec.kind == "dilated":
        assert spec.window is not None and spec.window >= 0
        assert spec.dilation is not None and spec.dilation >= 1

        return (D % spec.dilation == 0) & (D // spec.dilation <= spec.window)

    raise ValueError("Unknown kind")


def _stack_masks(seq_len: int, specs: List[HeadSpec], device: torch.device):
    return torch.stack([_build_head_mask(seq_len, s, device) for s in specs])


class TMSAB(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            head_dim: int,
            head_specs: List[HeadSpec],
            t_gate_hidden: Optional[int] = None):
        super().__init__()

        assert len(head_specs) == n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.head_specs = head_specs
        self.scale = 1.0 / math.sqrt(head_dim)

        self.Wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.Wk = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.Wv = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.Wo = nn.Linear(n_heads * head_dim, d_model, bias=False)

        hidden_t = t_gate_hidden or d_model
        self.T_lin1 = nn.Linear(d_model, hidden_t, bias=True)
        self.T_lin2 = nn.Linear(hidden_t, 1, bias=True)

        self.ln_in = nn.LayerNorm(d_model)
        self.ln_out = nn.LayerNorm(d_model)

        self._mask_cache_N = None
        self._mask_cache = None  # [H, N, N] bool

    def forward(self):
        pass


if __name__ == "__main__":
    emb = TimeEmb()

    t = 1

    temb = emb(t)
    print(temb.shape)
