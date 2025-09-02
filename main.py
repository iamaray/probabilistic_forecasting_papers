import torch
from torch.utils.data import DataLoader, TensorDataset
from ffnn.model import QMLP
from ffnn.trainer import Trainer
import matplotlib.pyplot as plt

def main():
    # Dummy data parameters
    batch_size = 16
    num_samples = 64
    history = 168
    num_features = 5
    forecast_horizon = 24
    num_quantiles = 3

    # Create dummy input and target tensors
    X = torch.randn((num_samples, history, num_features))
    y = torch.randn((num_samples, forecast_horizon))

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Split into train/val/test loaders (for demonstration, use same data)
    train_loader = loader
    val_loader = loader
    test_loader = loader

    # Instantiate model and trainer
    model = QMLP(
        in_dim=history * num_features,
        frc_dim=forecast_horizon,
        num_quantiles=num_quantiles
    )
    trainer = Trainer(model=model)

    # Test a forward pass
    for xb, yb in train_loader:
        out = model(xb.view(xb.size(0), -1))
        print("Model output shape:", out.shape)  # Should be [B, frc_dim, num_quantiles]
        break

    # Test training for a few epochs
    trainer.train(train_loader, val_loader, num_epochs=20)
    print("Train loss history:", trainer.running_train_loss)
    print("Val loss history:", trainer.running_val_loss)

    # Plot training and validation loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(trainer.running_train_loss, label='Train Loss')
    plt.plot(trainer.running_val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Test evaluation on test set
    test_loss = trainer.test(test_loader)
    print("Test loss:", test_loss)

if __name__ == "__main__":
    main()