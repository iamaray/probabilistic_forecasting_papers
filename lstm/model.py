import torch
import torch.nn as nn
import copy


class QLSTM(nn.LSTM):
    """
    LSTM wrapper that outputs non-crossing quantile forecasts using
    the nonnegative-increments construction:

        y_{q1} = base
        y_{qk} = y_{q(k-1)} + softplus(delta_k) + min_gap,  k=2..Q

    This guarantees y_{q1} <= y_{q2} <= ... <= y_{qQ} at every horizon.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.0,
                 forecast_length=24,
                 num_quantiles=3,
                 batch_first=True,
                 device=None,
                 min_gap: float = 0.0,
                 softplus_beta: float = 1.0):
        """
        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden state.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout probability (0 to 1).
            forecast_length (int): Number of steps to forecast.
            num_quantiles (int): Number of quantiles, ordered low->high.
            bidirectional (bool): If True, uses a bidirectional LSTM.
            batch_first (bool): If True, input/output are (batch, seq, feature).
            device: torch.device or None (auto-detects CUDA).
            min_gap (float): Optional strictly-positive floor added to each
                             increment to enforce *strict* monotonicity.
            softplus_beta (float): Beta parameter for Softplus.
        """
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=batch_first
        )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forecast_length = forecast_length
        self.num_quantiles = num_quantiles
        self.min_gap = float(min_gap)

        self.output_size = hidden_size
        self.forecast_head = nn.Linear(
            self.output_size, forecast_length * num_quantiles)
        self.softplus = nn.Softplus(beta=softplus_beta)

        self.to(self.device)

    def init_hidden(self, batch_size: int):
        """
        Initialize hidden state.

        Returns:
            (h0, c0)
        """
        num_directions = 2 if self.bidirectional else 1

        h0 = torch.zeros(self.num_layers * num_directions,
                         batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers * num_directions,
                         batch_size, self.hidden_size, device=self.device)
        return (h0, c0)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Args:
            x: (batch, seq_len, input_size) if batch_first=True
            hidden: optional (h0, c0)

        Returns:
            forecast: (batch, forecast_length, num_quantiles)
                      guaranteed non-decreasing across the quantile dimension
        """
        if not x.is_cuda and self.device.type == 'cuda':
            x = x.to(self.device)

        if hidden is not None:
            hidden = tuple(h.to(self.device) for h in hidden)

        output, hidden = super().forward(x, hidden)

        last_output = output[:, -1, :]  # [B, H]

        raw = self.forecast_head(last_output)  # [B, F*Q]
        raw = raw.view(-1, self.forecast_length,
                       self.num_quantiles)  # [B, F, Q]

        base = raw[:, :, 0]                                # [B, F]
        incr_raw = raw[:, :, 1:]                           # [B, F, Q-1]
        deltas = self.softplus(incr_raw)
        if self.min_gap > 0.0:
            deltas = deltas + self.min_gap

        csum = torch.cumsum(deltas, dim=-1)                # [B, F, Q-1]
        higher = base.unsqueeze(-1) + csum                 # [B, F, Q-1]
        forecast = torch.cat([base.unsqueeze(-1), higher], dim=-1)  # [B, F, Q]

        return forecast