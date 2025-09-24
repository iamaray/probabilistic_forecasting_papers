import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .layers import TimeEmb, TMSAB, HeadSpec


# def pad_to_batch(x: torch.Tensor, B: int):
#     if x.size(0) < B:
#         pad_amt = B - x.size(0)
#         x = torch.nn.functional.pad(x, (0, 0) * (x.dim() - 1) + (0, pad_amt))
#     return x


class DALNet(nn.Module):
    def __init__(
            self,
            lstm_hidden_dim,
            num_feats,
            dt,
            cond_dim,
            head_types: List[HeadSpec],
            num_lstm_layers,
            num_conv_layers,
            pred_seq_len,
            device):

        super().__init__()

        self.device = device or "cpu"

        self.lstm = nn.LSTM(input_size=1,
                            num_layers=num_lstm_layers,
                            hidden_size=lstm_hidden_dim,
                            bidirectional=False,
                            batch_first=True)

        d_model = lstm_hidden_dim + cond_dim + dt
        n_heads = len(head_types)
        assert d_model % n_heads == 0

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=d_model,
                      out_channels=d_model, kernel_size=1),
            nn.SiLU(),
            *[nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.SiLU()
            ) for _ in range(num_conv_layers - 1)],
            nn.Conv1d(in_channels=d_model, out_channels=1, kernel_size=1),
        )

        self.cond_proj = nn.Conv1d(in_channels=num_feats,
                                   out_channels=cond_dim, kernel_size=1)

        self.diffus_step_emb = nn.Sequential(
            TimeEmb(self.device),          # (64,)
            nn.Linear(64, 128), nn.SiLU(),
            nn.Linear(128, dt), nn.SiLU()    # -> (dt,)
        )

        self.atten = TMSAB(d_model, n_heads, head_types)

    def forward(self, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        """
            y: [B, N]       Noised input
            x: [B, L, C]    Historical & feature input
            t: [1]
        """
        y = y.to(self.device)
        x = x.to(self.device)

        if isinstance(t, torch.Tensor):
            t = t.to(self.device).item() if t.numel(
            ) == 1 else int(t.squeeze().item())

        B, N = y.shape

        seq, _ = self.lstm(y.unsqueeze(-1))  # [B, N, H]

        cond = self.cond_proj(x.transpose(1, 2))        # [B, cond_dim, L]
        if x.size(1) != N:
            cond = F.interpolate(
                cond, size=N, mode="linear", align_corners=False)
        proj_feats = cond.transpose(1, 2)               # [B, N, cond_dim]
        t_vec = self.diffus_step_emb(t)                      # [d_t]
        t_vec = t_vec.view(1, 1, -1).expand(B, N, -1)       # [B, N, d_t]

        # [B, N, (H + C + dt)]
        # seq = pad_to_batch(seq, B)
        # proj_feats = pad_to_batch(proj_feats, B)
        # t_vec = pad_to_batch(t_vec, B)

        atten_in = torch.cat([seq, proj_feats, t_vec], dim=-1)
        # [B, N, (H + C + 1)]
        atten_out = self.atten(atten_in)

        y_hat = self.conv_net(atten_out.transpose(
            1, 2)).transpose(1, 2).squeeze(-1)  # [B, N]

        return y_hat
