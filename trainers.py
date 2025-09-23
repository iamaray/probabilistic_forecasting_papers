import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, Tuple


import torch


# def pad_to_batch(x: torch.Tensor, B: int):
#     if x.size(0) < B:
#         pad_amt = B - x.size(0)
#         x = torch.nn.functional.pad(x, (0, 0) * (x.dim() - 1) + (0, pad_amt))
#     return x


def quantileLoss(y_pred: torch.Tensor,   # [..., Q]
                 y_true: torch.Tensor,   # [...]
                 quantiles: torch.Tensor  # [Q] in (0,1), ascending
                 ) -> torch.Tensor:
    """
    Pinball (quantile) loss over all elements and all quantiles.
    y_pred and y_true must broadcast to the same leading dims; y_pred ends with Q.
    """
    if y_true.dim() == y_pred.dim() - 1:
        y_true = y_true.unsqueeze(-1)

    e = y_true - y_pred                                 # [..., Q]

    q = quantiles.to(device=y_pred.device, dtype=y_pred.dtype)
    q = q.view(*([1] * (e.ndim - 1)), -1)

    loss = torch.maximum(q * e, (q - 1.0) * e)          # [..., Q]
    return loss.mean()                                  # scalar


def ACR(y: torch.Tensor, frc_top: torch.Tensor, frc_bottom: torch.Tensor):
    """Average coverage rate: fraction of targets that fall within [lower, upper]."""
    lower = torch.minimum(frc_bottom, frc_top)
    upper = torch.maximum(frc_bottom, frc_top)

    valid = torch.isfinite(y) & torch.isfinite(lower) & torch.isfinite(upper)
    if valid.sum() == 0:
        return torch.tensor(float('nan'), device=y.device)
    # print('here')
    covered = (y >= lower) & (y <= upper)
    return (covered & valid).float().sum() / valid.float().sum()


def AIL(y: torch.Tensor, frc_top: torch.Tensor, frc_bottom: torch.Tensor):
    """Average interval length: mean of (upper - lower) over valid bounds."""
    lower = torch.minimum(frc_bottom, frc_top)
    upper = torch.maximum(frc_bottom, frc_top)

    valid = torch.isfinite(lower) & torch.isfinite(upper)
    if valid.sum() == 0:
        return torch.tensor(float('nan'), device=lower.device)

    lengths = (upper - lower).clamp_min(0)
    return lengths[valid].mean()


def compute_metrics(
        y_true: torch.Tensor,       # [B, pred]
        # [B, pred, D]  (D=Q for quantile models; D=num_samples for sampled)
        y_pred: torch.Tensor,
        quantiles: torch.Tensor,    # [Q], ascending in (0,1)
        sampled: bool = False):

    if sampled:
        q = quantiles.to(device=y_pred.device, dtype=y_pred.dtype)
        pred_quantiles = torch.quantile(
            y_pred, q, dim=-1).permute(1, 2, 0)  # [B, pred, Q]
        print("pred quantiles", pred_quantiles.shape)
    else:
        # [B, pred, Q]
        pred_quantiles = y_pred

    pinball = quantileLoss(pred_quantiles, y_true, quantiles)
    lower = pred_quantiles[..., 0]       # [B, pred]
    upper = pred_quantiles[..., -1]      # [B, pred]
    acr = ACR(y_true, upper, lower)
    ail = AIL(y_true, upper, lower)

    return {
        "pinball": pinball,
        "acr": acr,
        "ail": ail
    }


class QRTrainer(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            device: Optional[torch.device] = None,
            quantiles: torch.Tensor = torch.Tensor([0.1, 0.5, 0.9]),
            lr: float = 0.001,
            max_grad_norm: float = 1.0,
            T_max: int = 20):

        super().__init__()
        """
            Trainer for standard pinball-loss guided quantile regression.
        """

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(device)
        self.quantiles = quantiles.to(device)
        self.max_grad_norm = max_grad_norm

        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=T_max)

        self.running_train_loss = []
        self.running_val_loss = []
        self.running_test_loss = []

    def _train_epoch(self, train_loader: DataLoader):
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for (xt, yt) in train_loader:
            # xt: [B, H, C]
            # yt: [B, pred]

            self.optim.zero_grad()

            xt = xt.to(self.device)
            yt = yt.to(self.device)

            out = self.model(xt)
            loss = quantileLoss(out, yt, quantiles=self.quantiles)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            self.optim.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        return epoch_loss

    def _eval_epoch(self, val_loader: DataLoader):
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for xv, yv in val_loader:

                xv = xv.to(self.device)
                yv = yv.to(self.device)

                preds = self.model(xv)
                loss = quantileLoss(preds, yv, quantiles=self.quantiles)
                val_loss += loss.item()
                num_batches += 1

        return val_loss / num_batches

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        for i in range(num_epochs):
            train_epoch_loss = self._train_epoch(train_loader)
            eval_epoch_loss = self._eval_epoch(val_loader)

            self.scheduler.step()

            print(
                f"Epoch {i+1}/{num_epochs}: train loss = {train_epoch_loss:.6f}, eval loss = {eval_epoch_loss:.6f}, lr = {self.scheduler.get_last_lr()[0]:.2e}")

            self.running_train_loss.append(train_epoch_loss)
            self.running_val_loss.append(eval_epoch_loss)

    def test(self, test_loader, transform):
        self.model.eval()
        test_loss = 0.0
        num_batches = 0

        history = []

        transform.set_device(self.device)

        with torch.no_grad():
            metrics = []
            for xt, yt in test_loader:
                yt = yt.to(self.device)
                xt = xt.to(self.device)

                xt_transformed = transform.transform(xt)
                xt_transformed = xt_transformed.to(self.device)

                yt_transformed = transform.transform(
                    yt.unsqueeze(-1), transform_col=0).squeeze()
                yt_transformed = yt_transformed.to(self.device)

                out = self.model(xt_transformed)
                # out, _ = torch.sort(out, dim=-1)
                # loss = quantileLoss(out, yt_transformed,
                #                     quantiles=self.quantiles)
                # test_loss += loss.item()

                metrics.append(compute_metrics(
                    yt_transformed.detach().cpu(), out.detach().cpu(), self.quantiles, sampled=False))

                out_reversed = transform.reverse(
                    transformed=out.unsqueeze(-1), reverse_col=0).squeeze()

                num_batches += 1
                history.append(
                    [yt.detach().cpu(), out_reversed.detach().cpu()])

        test_res = torch.cat([
            torch.cat([yt.unsqueeze(-1), out_reversed], dim=-1)
            for yt, out_reversed in history
        ], dim=0)
        print("res shape", test_res.shape)

        # avg_test_loss = test_loss / num_batches if num_batches > 0 else 0.0

        metrics_dict = {
            "pinball": np.mean([m['pinball'] for m in metrics]),
            "acr": np.mean([m['acr'] for m in metrics]),
            "ail": np.mean([m['ail'] for m in metrics])
        }
        return metrics_dict, test_res


@torch.no_grad()
def sample_ddpm_model(
        model: nn.Module,
        variance_sched: torch.Tensor,
        # [B, L, C]
        cond: torch.Tensor,
        out_shape: Tuple[int, ...],
        num_samples: int,
        device: Optional[torch.device] = None):

    device = device or next(model.parameters()).device
    beta = variance_sched.to(device).clamp_(1e-8, 0.999)  # [T]
    T = beta.shape[0]
    alpha = 1.0 - beta                                   # [T]
    alphabar = torch.cumprod(alpha, dim=0)               # [T]

    def tilde_beta(t: int) -> torch.Tensor:
        ab_t = alphabar[t]
        ab_tm1 = alphabar[t-1] if t > 0 else torch.tensor(
            1.0, device=device, dtype=ab_t.dtype)
        return ((1.0 - ab_tm1) / (1.0 - ab_t)) * beta[t]

    samples = []

    cond = cond.to(device)

    for _ in range(num_samples):
        y_t = torch.randn(out_shape, device=device)

        for t in reversed(range(T)):
            sqrt_inv_alpha_t = (1.0 / alpha[t].sqrt())
            sqrt_one_minus_ab_t = (1.0 - alphabar[t]).sqrt()

            eps_pred = model(y_t, cond, t)
            y_mean = sqrt_inv_alpha_t * \
                (y_t - beta[t] / sqrt_one_minus_ab_t * eps_pred)

            if t > 0:
                z = torch.randn_like(y_t)
                sigma_t = tilde_beta(t).sqrt()
                y_t = y_mean + sigma_t * z
            else:
                y_t = y_mean

        samples.append(y_t.unsqueeze(-1))

    return torch.cat(samples, dim=-1)   # [*out_shape, num_samples]


class DDPMTrainer(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            variance_sched: torch.Tensor,
            device: Optional[torch.device] = None,
            lr: float = 1e-3,
            max_grad_norm: float = 1.0,
            T_max: int = 20):

        super().__init__()
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.max_grad_norm = max_grad_norm

        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=T_max)

        self.running_train_loss = []
        self.running_val_loss = []

        self.beta = variance_sched.to(self.device).clamp_(1e-8, 0.999)  # [T]
        self.T = self.beta.shape[0]
        self.alpha = 1.0 - self.beta                     # [T]
        self.alphabar = torch.cumprod(self.alpha, dim=0)  # [T]

        self.mse = nn.MSELoss()

    def _train_epoch(self, train_loader: DataLoader):
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for (x_hist, y_future) in train_loader:
            # x_hist: [B, L, C]  (condition)
            # y_future: [B, N]   (x_0)
            x_hist = x_hist.to(self.device)
            y0 = y_future.to(self.device)

            B, N = y0.shape

            t = torch.randint(low=0, high=self.T, size=(),
                              device=self.device).item()  # scalar

            eps = torch.randn_like(y0)               # [B, N]
            sqrt_ab_t = self.alphabar[t].sqrt()    # scalar
            sqrt_1mab_t = (1.0 - self.alphabar[t]).sqrt()  # scalar
            x_t = sqrt_ab_t * y0 + sqrt_1mab_t * eps       # [B, N]

            self.optim.zero_grad(set_to_none=True)

            eps_pred = self.model(x_t, x_hist, t)    # [B, N]
            loss = self.mse(eps_pred, eps)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            self.optim.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        return epoch_loss / max(1, num_batches)

    def _eval_epoch(self, val_loader: DataLoader):
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for (x_hist, y_future) in val_loader:
                x_hist = x_hist.to(self.device)
                y0 = y_future.to(self.device)
                B, N = y0.shape

                t = torch.randint(low=0, high=self.T, size=(),
                                  device=self.device).item()
                eps = torch.randn_like(y0)
                sqrt_ab_t = self.alphabar[t].sqrt()
                sqrt_1mab_t = (1.0 - self.alphabar[t]).sqrt()
                x_t = sqrt_ab_t * y0 + sqrt_1mab_t * eps

                eps_pred = self.model(x_t, x_hist, t)
                loss = self.mse(eps_pred, eps)

                val_loss += float(loss.item())
                num_batches += 1
        return val_loss / max(1, num_batches)

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader], num_epochs: int):
        for epoch in range(num_epochs):
            train_epoch_loss = self._train_epoch(train_loader)
            self.running_train_loss.append(train_epoch_loss)

            eval_epoch_loss = None
            if val_loader is not None:
                eval_epoch_loss = self._eval_epoch(val_loader)
                self.running_val_loss.append(eval_epoch_loss)

            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"train loss = {train_epoch_loss:.6f}"
                + (f", val loss = {eval_epoch_loss:.6f}" if eval_epoch_loss is not None else "")
                + f", lr={lr:.2e}"
            )

    def test(self, test_loader: DataLoader, transform, variance_sched: torch.Tensor, num_samples: int, quantiles=torch.Tensor([0.1, 0.5, 0.9])):
        self.model.eval()
        test_loss = 0.0
        num_batches = 0

        history = []

        transform.set_device(self.device)

        with torch.no_grad():
            metrics = []
            for xt, yt in test_loader:
                xt = xt.to(self.device)
                yt = yt.to(self.device)

                xt_transformed = transform.transform(xt)
                xt_transformed = xt_transformed.to(self.device)

                yt_transformed = transform.transform(
                    yt.unsqueeze(-1), transform_col=0).squeeze()
                yt_transformed = yt_transformed.to(self.device)

                out_shape = yt.shape
                out = sample_ddpm_model(self.model, variance_sched, cond=xt,
                                        out_shape=out_shape, num_samples=num_samples, device=self.device)
                # out, _ = torch.sort(out, dim=-1)
                # loss = quantileLoss(out, yt_transformed,
                #                     quantiles=self.quantiles)
                # test_loss += loss.item()

                metrics.append(compute_metrics(
                    yt_transformed.detach().cpu(), out, quantiles.detach().cpu(), sampled=True))

                out_reversed = transform.reverse(
                    transformed=out.unsqueeze(-1), reverse_col=0).squeeze()

                num_batches += 1
                history.append(
                    [yt.detach().cpu(), out_reversed.detach().cpu()])

        test_res = torch.cat([
            torch.cat([yt.unsqueeze(-1), out_reversed], dim=-1)
            for yt, out_reversed in history
        ], dim=0)
        print("res shape", test_res.shape)

        # avg_test_loss = test_loss / num_batches if num_batches > 0 else 0.0

        metrics_dict = {
            "pinball": np.mean([m['pinball'] for m in metrics]),
            "acr": np.mean([m['acr'] for m in metrics]),
            "ail": np.mean([m['ail'] for m in metrics])
        }
        return metrics_dict, test_res
