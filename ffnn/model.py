import torch
import torch.nn as nn


class QMLP(nn.Module):
    def __init__(
        self,
        in_dim=168,
        frc_dim=24,
        depth=3,
        hidden_size=128,
        dropout=0.1,
        num_quantiles=3,
        device=None,
        strict_eps: float = 0.0
    ):
        super().__init__()
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.frc_dim = frc_dim
        self.num_quantiles = num_quantiles
        self.depth = depth
        self.activation = nn.ReLU()
        self.softplus = nn.Softplus()
        self.strict_eps = float(strict_eps)

        self.input_transform = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_size),
            self.activation
        )

        self.hidden_net = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_features=hidden_size,
                              out_features=hidden_size),
                    self.activation,
                    nn.Dropout(dropout)
                )
                for _ in range(depth - 1)
            ]
        )

        self.output_transform = nn.Linear(
            in_features=hidden_size, out_features=(frc_dim * num_quantiles)
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor):
        """
        x: [B, *, *] -> flattened to [B, in_dim]
        returns: [B, frc_dim, num_quantiles]
        """
        x = x.to(self.device)
        B = x.shape[0]
        z = x.view(B, -1)

        z = self.input_transform(z)
        z = self.hidden_net(z)
        out = self.output_transform(z)  # [B, frc_dim * Q]
        out = out.view(-1, self.frc_dim, self.num_quantiles)  # [B, F, Q]

        if self.num_quantiles == 1:
            return out

        base = out[..., :1]               # [B, F, 1]
        raw_deltas = out[..., 1:]         # [B, F, Q-1]
        deltas_pos = self.softplus(raw_deltas)
        if self.strict_eps > 0.0:
            deltas_pos = deltas_pos + self.strict_eps

        cum_increments = torch.cumsum(deltas_pos, dim=-1)  # [B, F, Q-1]
        q = torch.cat([base, base + cum_increments],
                      dim=-1)  # [B, F, Q]

        return q

    def get_device(self):
        return next(self.parameters()).device
