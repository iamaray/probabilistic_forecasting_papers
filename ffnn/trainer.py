import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def quantileLoss(
        y_pred: torch.Tensor,  # [B, H, Q]
        y_true: torch.Tensor,  # [B, H]
        quantiles: torch.Tensor):

    diff = y_true.unsqueeze(-1) - y_pred
    q = quantiles.view(1, 1, -1).to(y_pred.device)

    return torch.mean(torch.max(q*diff, (q-1)*diff))


class Trainer(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device = torch.device('cpu'),
            quantiles: torch.Tensor = torch.Tensor([0.1, 0.5, 0.9]),
            lr: float = 0.001,
            max_grad_norm: float = 1.0,
            T_max: int = 20):

        super().__init__()
        self.device = device
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
            for xt, yt in test_loader:
                xt_transformed = transform.transform(xt)
                xt_transformed = xt_transformed.to(self.device)

                yt_transformed = transform.transform(
                    yt.unsqueeze(-1), transform_col=0).squeeze()
                yt_transformed = yt_transformed.to(self.device)

                out = self.model(xt_transformed)
                out, _ = torch.sort(out, dim=-1)
                loss = quantileLoss(out, yt_transformed,
                                    quantiles=self.quantiles)
                test_loss += loss.item()

                out_reversed = transform.reverse(
                    transformed=out.unsqueeze(-1), reverse_col=0).squeeze()

                num_batches += 1
                history.append([yt.cpu(), out_reversed.cpu()])

        test_res = torch.cat([
            torch.stack([yt, out_reversed[:, :, 0],
                        out_reversed[:, :, 1], out_reversed[:, :, 2]], dim=1)
            for yt, out_reversed in history
        ], dim=0)

        avg_test_loss = test_loss / num_batches if num_batches > 0 else 0.0


        return avg_test_loss, test_res
