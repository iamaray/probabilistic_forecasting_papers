import torch
from torch.utils.data import DataLoader, TensorDataset
from ffnn.model import QMLP
# from ffnn.trainer import Trainer, quantileLoss
from lstm.model import QLSTM
import matplotlib.pyplot as plt
import numpy as np
from processing.transforms import *

from trainers import QRTrainer, DDPMTrainer, sample_ddpm_model, compute_metrics
from dal_net.model import DALNet
from dal_net.layers import HeadSpec


def main():
    data_path = 'data/spain_data'

    train_loader = torch.load(
        f'{data_path}/train_loader_non_spatial.pt', weights_only=False)
    val_loader = torch.load(
        f'{data_path}/val_loader_non_spatial.pt', weights_only=False)
    test_loader = torch.load(
        f'{data_path}/test_loader_non_spatial.pt', weights_only=False)
    transform = torch.load(
        f'{data_path}/transform_non_spatial.pt', weights_only=False)

    x_sample, y_sample = next(iter(train_loader))
    print(f"Training data shapes: X={x_sample.shape}, Y={y_sample.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QMLP(
        in_dim=x_sample.shape[1]*x_sample.shape[2],
        depth=5,
        strict_eps=1e-4,
        device=device
    )

    trainer = QRTrainer(model, device)

    trainer.train(train_loader=train_loader,
                  val_loader=val_loader, num_epochs=1)

    metrics, _ = trainer.test(test_loader, transform)  # [B, L, Q]
    print(
        f"Pinball:{metrics['pinball']}, ACR:{metrics['acr']}, AIL:{metrics['ail']}")


if __name__ == "__main__":
    main()
