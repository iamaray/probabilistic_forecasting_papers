import torch
from torch.utils.data import DataLoader, TensorDataset
from ffnn.model import QMLP
# from ffnn.trainer import Trainer, quantileLoss
from lstm.model import QLSTM
import matplotlib.pyplot as plt
import numpy as np
from processing.transforms import *

from trainers import QRTrainer, DDPMTrainer, sample_ddpm_model
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

    head_types: List[HeadSpec] = [
        HeadSpec("global"),
        # HeadSpec("global"),
        # HeadSpec("window", window=4),
        # HeadSpec("window", window=4),
        # HeadSpec("dilated", window=4, dilation=2),
        # HeadSpec("dilated", window=4, dilation=2),
        # HeadSpec("window", window=2),
        # HeadSpec("window", window=2)
    ]

    model = DALNet(
        lstm_hidden_dim=10,
        num_feats=10,       # number of exogenous features
        dt=16,              # diffusion step embedding dim
        cond_dim=16,        # condition projection dim
        head_types=head_types,
        num_lstm_layers=2,
        num_conv_layers=3,
        pred_seq_len=24,
        device=device,
    ).to(device)

    def quadratic_variance_schedule(T: int,
                                    beta_start: float = 1e-4,
                                    beta_end: float = 0.5,
                                    device=None,
                                    dtype=torch.float32) -> torch.Tensor:
        steps = torch.arange(1, T+1, device=device, dtype=dtype)  # [1,...,T]
        sqrt_beta_start = beta_start**0.5
        sqrt_beta_end = beta_end**0.5

        betas = (((T - steps) / (T - 1)) * sqrt_beta_start +
                 ((steps - 1) / (T - 1)) * sqrt_beta_end) ** 2
        return betas

    variance_sched = quadratic_variance_schedule(500, 1e-4, 0.5, device=device)

    trainer = DDPMTrainer(
        model=model,
        variance_sched=variance_sched,
        device=device,
        lr=1e-3,
    )

    trainer.train(train_loader=train_loader, val_loader=None, num_epochs=300)

    metrics, _ = trainer.test(test_loader=test_loader, transform=transform,
                              variance_sched=variance_sched, num_samples=50)
    print(
        f"Pinball:{metrics['pinball']}, ACR:{metrics['acr']}, AIL:{metrics['ail']}")


if __name__ == "__main__":
    main()
