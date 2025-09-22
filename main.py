import torch
from torch.utils.data import DataLoader, TensorDataset
from ffnn.model import QMLP
from lstm.model import QLSTM
import matplotlib.pyplot as plt
import numpy as np
import argparse

from processing.transforms import *
from trainers import QRTrainer, DDPMTrainer, sample_ddpm_model
from dal_net.model import DALNet
from dal_net.layers import HeadSpec


def dalnet_test(train_loader, val_loader, test_loader, transform, in_shape, label_shape, device):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head_types: List[HeadSpec] = [
        HeadSpec("global"),
        HeadSpec("global"),
        HeadSpec("window", window=4),
        HeadSpec("window", window=4),
        HeadSpec("dilated", window=4, dilation=2),
        HeadSpec("dilated", window=4, dilation=2),
        HeadSpec("window", window=2),
        HeadSpec("window", window=2)
    ]

    model = DALNet(
        lstm_hidden_dim=256,
        num_feats=in_shape[-1],       # number of exogenous features
        dt=16,              # diffusion step embedding dim
        cond_dim=16,        # condition projection dim
        head_types=head_types,
        num_lstm_layers=2,
        num_conv_layers=3,
        pred_seq_len=label_shape[1],
        device=device,
    ).to(device)

    def quadratic_variance_schedule(
            T: int,
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


def qlstm_test(train_loader, val_loader, test_loader, transform, in_shape, label_shape, device):
    model = QLSTM(
        input_size=in_shape[-1],
        hidden_size=512,
        num_layers=4,
        min_gap=1e-4,
        forecast_length=label_shape[1],
        dropout=0.1
    )

    trainer = QRTrainer(model, device)

    trainer.train(train_loader=train_loader,
                  val_loader=val_loader, num_epochs=300)

    metrics, _ = trainer.test(test_loader, transform)  # [B, L, Q]
    print(
        f"Pinball:{metrics['pinball']}, ACR:{metrics['acr']}, AIL:{metrics['ail']}")


def qmlp_test(train_loader, val_loader, test_loader, transform, in_shape, label_shape, device):
    model = QMLP(
        in_dim=in_shape[-2] * in_shape[-1],
        frc_dim=label_shape[1],
        depth=3,
        hidden_size=256,
        device=device,
        strict_eps=1e-4
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        description='Run model training or grid search based on configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Which model')
    parser.add_argument('--data_path', type=str,
                        required=True, help='Path to data folder.')

    args = parser.parse_args()

    data_path = args.data_path

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

    in_shape = x_sample.shape
    label_shape = y_sample.shape

    if args.model == 'qmlp':
        qmlp_test(train_loader, val_loader, test_loader,
                  transform, in_shape, label_shape, device)

    if args.model == 'qlstm':
        qlstm_test(train_loader, val_loader, test_loader,
                   transform, in_shape, label_shape, device)

    if args.model == "dalnet":
        dalnet_test(train_loader, val_loader, test_loader,
                    transform, in_shape, label_shape, device)
