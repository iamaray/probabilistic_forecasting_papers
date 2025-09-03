import torch
from torch.utils.data import DataLoader, TensorDataset
from ffnn.model import QMLP
from ffnn.trainer import Trainer, quantileLoss
import matplotlib.pyplot as plt
import numpy as np
from processing.transforms import *


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
    print(f"Using device: {device}")

    B, H, C = x_sample.shape
    input_dim = H * C
    frc_dim = y_sample.shape[1]

    quantiles = torch.tensor([0.1, 0.5, 0.9])

    model = QMLP(
        in_dim=input_dim,
        frc_dim=frc_dim,
        depth=3,
        hidden_size=128,
        dropout=0.1,
        num_quantiles=3,
        device=device
    )

    num_epochs = 100

    trainer = Trainer(
        model=model,
        device=device,
        quantiles=quantiles,
        lr=0.001,
        max_grad_norm=1.0,
        T_max=num_epochs
    )

    print("Starting training...")
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)

    print("\nTraining completed. Now testing...")

    avg_test_loss, test_results = trainer.test(
        test_loader, transform=transform)
    print(f"Average test loss: {avg_test_loss:.6f}")
    print(f"Test results shape: {test_results.shape}")

    true_values = test_results[:, 0, :]
    pred_q10 = test_results[:, 1, :]
    pred_q50 = test_results[:, 2, :]
    pred_q90 = test_results[:, 3, :]

    print("Creating plots for random test samples...")

    n_samples = test_results.shape[0]
    random_indices = torch.randperm(n_samples)[:3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, idx in enumerate(random_indices):
        ax = axes[i]

        true_vals = true_values[idx].cpu().numpy()
        pred_10_vals = pred_q10[idx].cpu().numpy()
        pred_50_vals = pred_q50[idx].cpu().numpy()
        pred_90_vals = pred_q90[idx].cpu().numpy()

        time_steps = np.arange(len(true_vals))

        ax.plot(time_steps, true_vals, 'k-', linewidth=2.5,
                label='True Values', alpha=0.9)
        ax.plot(time_steps, pred_50_vals, 'b-', linewidth=2,
                label='Predicted (Median)', alpha=0.8)
        ax.fill_between(time_steps, pred_10_vals, pred_90_vals,
                        alpha=0.3, color='lightblue', label='80% CI (10th-90th percentile)')
        ax.plot(time_steps, pred_10_vals, '--', color='blue',
                alpha=0.6, linewidth=1, label='10th percentile')
        ax.plot(time_steps, pred_90_vals, '--', color='blue',
                alpha=0.6, linewidth=1, label='90th percentile')

        ax.set_title(f'Test Sample {idx.item() + 1}',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Forecast Horizon (hours)', fontsize=12)
        ax.set_ylabel('Energy Price', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        mae = np.mean(np.abs(pred_50_vals - true_vals))
        ax.text(0.02, 0.98, f'MAE: {mae:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('test_predictions_with_ci.png', dpi=300, bbox_inches='tight')
    # plt.show()

    print(f"\nSummary Statistics:")
    print(
        f"True values range: [{true_values.min():.3f}, {true_values.max():.3f}]")
    print(
        f"Predicted median range: [{pred_q50.min():.3f}, {pred_q50.max():.3f}]")
    print(
        f"Mean Absolute Error (median): {torch.mean(torch.abs(pred_q50 - true_values)).item():.6f}")

    coverage = torch.mean(((true_values >= pred_q10) & (
        true_values <= pred_q90)).float()).item()
    print(f"80% CI Coverage: {coverage:.3f} ({coverage*100:.1f}%)")

    print("\nPlots saved as 'test_predictions_with_ci.png'")

    print("\nCreating comprehensive test period plot...")

    all_true = true_values.cpu().numpy()
    all_pred_median = pred_q50.cpu().numpy()
    all_pred_q10 = pred_q10.cpu().numpy()
    all_pred_q90 = pred_q90.cpu().numpy()

    true_series = all_true[:, 0]
    pred_series = all_pred_median[:, 0]
    q10_series = all_pred_q10[:, 0]
    q90_series = all_pred_q90[:, 0]

    time_axis = np.arange(len(true_series))

    plt.figure(figsize=(20, 8))

    plt.plot(time_axis, true_series, 'k-', linewidth=1,
             label='True Values', alpha=0.8)
    plt.plot(time_axis, pred_series, 'b-', linewidth=1,
             label='Predicted (Median)', alpha=0.7)
    plt.fill_between(time_axis, q10_series, q90_series,
                     alpha=0.2, color='lightblue', label='80% CI')

    plt.title('QMLP Probabilistic Forecasting - Entire Test Period',
              fontsize=16, fontweight='bold')
    plt.xlabel('Time (hours)', fontsize=14)
    plt.ylabel('Energy Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    overall_mae = np.mean(np.abs(pred_series - true_series))
    overall_coverage = np.mean(
        (true_series >= q10_series) & (true_series <= q90_series))

    stats_text = f'Overall MAE: {overall_mae:.3f}\nOverall Coverage: {overall_coverage:.1%}\nSamples: {len(true_series):,}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11)

    plt.tight_layout()
    plt.savefig('full_test_period.png', dpi=300, bbox_inches='tight')
    # plt.show()

    print("Creating zoomed-in view of representative section...")

    start_idx = len(true_series) // 3
    end_idx = start_idx + 1000

    if end_idx > len(true_series):
        end_idx = len(true_series)
        start_idx = max(0, end_idx - 1000)

    plt.figure(figsize=(16, 6))

    zoom_time = time_axis[start_idx:end_idx]
    zoom_true = true_series[start_idx:end_idx]
    zoom_pred = pred_series[start_idx:end_idx]
    zoom_q10 = q10_series[start_idx:end_idx]
    zoom_q90 = q90_series[start_idx:end_idx]

    plt.plot(zoom_time, zoom_true, 'k-', linewidth=1.5,
             label='True Values', alpha=0.9)
    plt.plot(zoom_time, zoom_pred, 'b-', linewidth=1.5,
             label='Predicted (Median)', alpha=0.8)
    plt.fill_between(zoom_time, zoom_q10, zoom_q90,
                     alpha=0.3, color='lightblue', label='80% CI')

    plt.title(f'QMLP Forecasting - Detailed View (Samples {start_idx:,} to {end_idx:,})',
              fontsize=14, fontweight='bold')
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Energy Price', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    zoom_mae = np.mean(np.abs(zoom_pred - zoom_true))
    zoom_coverage = np.mean((zoom_true >= zoom_q10) & (zoom_true <= zoom_q90))

    zoom_stats = f'Section MAE: {zoom_mae:.3f}\nSection Coverage: {zoom_coverage:.1%}'
    plt.text(0.02, 0.98, zoom_stats, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=10)

    plt.tight_layout()
    plt.savefig('test_period_detailed.png', dpi=300, bbox_inches='tight')
    # plt.show()

    print(f"\nFull test period plots saved:")
    print(
        f"- 'full_test_period.png': Complete overview ({len(true_series):,} samples)")
    print(
        f"- 'test_period_detailed.png': Detailed view (samples {start_idx:,}-{end_idx:,})")


if __name__ == "__main__":
    main()
