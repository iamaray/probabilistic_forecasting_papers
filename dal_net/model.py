import torch
import torch.nn as nn

from .layers import TimeEmb


class DALNet(nn.Module):
    def __init__(
            self,
            lstm_hidden_dim,
            num_features,
            num_lstm_layers,
            num_conv_layers,
            feature_seq_len,
            pred_seq_len,
            device):

        super().__init__()

        self.device = device or "cpu"

        self.lstm = nn.LSTM(input_size=num_features,
                            num_layers=num_lstm_layers,
                            hidden_size=lstm_hidden_dim,
                            bidirectional=False,
                            batch_first=True)

        self.conv_net = nn.Sequential(
            *[nn.Conv1d(in_channels=lstm_hidden_dim, out_channels=lstm_hidden_dim)
              for _ in range(num_conv_layers)],
            nn.Conv1d(in_channels=lstm_hidden_dim, out_channels=1)
        )

        self.cond_proj = nn.Conv1d(
            in_channels=feature_seq_len, out_channels=pred_seq_len)

        self.diffus_step_emb = nn.Sequential(
            TimeEmb(device=self.device),
            nn.Linear(in_features=64, out_features=128),
            nn.SiLU(),
            nn.Linear(in_features=128, out_features=pred_seq_len),
            nn.SiLU()
        )

    def forward(self, y: torch.Tensor, x: torch.Tensor):
        """
            y: [B, L]       Noised input
            x: [B, L, C]    Historical & feature input
        """

        pass
