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
            quantiles: torch.Tensor = torch.Tensor([0.1, 0.5, 0.9])):

        super().__init__()
        self.device = device 
        self.model = model.to(device) 
        self.quantiles = quantiles.to(device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=10)

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

            xt = xt.to(self.device)
            yt = yt.to(self.device)

            B, H, C = xt.shape
            out = self.model(xt.view(B, -1))

            loss = quantileLoss(out, yt, quantiles=self.quantiles)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

        self.scheduler.step()
        epoch_loss /= num_batches

        return epoch_loss

    def _eval_epoch(self, val_loader: DataLoader):
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for xv, yv in val_loader:
                B, H, C = xv.shape

                xv = xv.to(self.device)
                yv = yv.to(self.device)

                preds = self.model(xv.view(B, -1))
                loss = quantileLoss(preds, yv, quantiles=self.quantiles)
                val_loss += loss.item()
                num_batches += 1

        return val_loss / num_batches

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        for i in range(num_epochs):
            train_epoch_loss = self._train_epoch(train_loader)
            eval_epoch_loss = self._eval_epoch(val_loader)

            self.running_train_loss.append(train_epoch_loss)
            self.running_val_loss.append(eval_epoch_loss)

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for xt, yt in test_loader:
                xt = xt.to(self.device)
                yt = yt.to(self.device)
                B, H, C = xt.shape
                
                out = self.model(xt.view(B, -1))
                loss = quantileLoss(out, yt, quantiles=self.quantiles)
                test_loss += loss.item()
                num_batches += 1

        avg_test_loss = test_loss / num_batches if num_batches > 0 else 0.0
        return avg_test_loss
