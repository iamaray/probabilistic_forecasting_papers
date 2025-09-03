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
            device=None):

        super().__init__()
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.frc_dim = frc_dim
        self.num_quantiles = num_quantiles
        self.depth = depth
        self.activation = nn.ReLU()

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
                    nn.Dropout(dropout))

                for _ in range(depth - 1)
            ]
        )

        self.output_transform = nn.Linear(
            in_features=hidden_size, out_features=(frc_dim*num_quantiles))

        self.to(self.device)

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        B, _, _ = x.shape
        z = x.view(B, -1)
        
        z = self.input_transform(z)
        z = self.hidden_net(z)
        out = self.output_transform(z)

        out = out.view(-1, self.frc_dim, self.num_quantiles)
        return out

    def get_device(self):
        return next(self.parameters()).device
