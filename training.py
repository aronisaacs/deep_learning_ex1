import os
from typing import Any
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def _binary_predictions(raw_outputs: torch.Tensor) -> torch.Tensor:
    """Convert model outputs to binary predictions for accuracy."""
    if raw_outputs.min() < 0 or raw_outputs.max() > 1:
        return (raw_outputs >= 0).long()
    return (raw_outputs >= 0.5).long()

def train_model_epoch_eval(
    model,
    optimizer,
    train_dataset,
    test_dataset,
    loss_module,
    num_epochs=100,
    batch_size=256,
    device="cpu",
    model_save_path="trained_model.pt",
):
    """
    Trains 'model' for 'num_epochs' and evaluates after each epoch on the entire test set in one pass.
    Uses a single pass for test data (assuming test_dataset is small).

    Returns:
        dict: history with per-epoch metrics and model save metadata.
    """
    # 1) Set up DataLoaders
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # If the test set is tiny, we can put the entire dataset into a single batch
    test_loader = data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    history: dict[str, Any] = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "saved_model_path": None,
    }

    model.to(device)
    for epoch in tqdm(range(num_epochs)):
        ####################################################
        # Training phase
        ####################################################
        model.train()
        epoch_train_loss = []
        train_correct = 0
        train_total = 0

        for data_inputs, data_labels in train_loader:
            # 1) Move input data and labels to device
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            # 2) Forward pass
            preds = model(data_inputs)
            # If model output shape is [B,1] but your labels are [B],
            # you can squeeze the prediction:
            preds = preds.squeeze(dim=1)

            # 3) Compute loss
            loss = loss_module(preds, data_labels.float())
            epoch_train_loss.append(loss.item())

            batch_preds = _binary_predictions(preds)
            batch_labels = data_labels.long()
            train_correct += (batch_preds == batch_labels).sum().item()
            train_total += batch_labels.numel()

            # 4) Backprop + weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Average loss for this epoch
        history["train_loss"].append(np.mean(epoch_train_loss))
        history["train_accuracy"].append(train_correct / train_total if train_total > 0 else 0.0)

        ####################################################
        # Evaluation phase (single pass over the entire test set)
        ####################################################
        with torch.no_grad():
            model.eval()

            # Get the single batch from test_loader
            (X_test, Y_test) = next(iter(test_loader))
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)

            # Forward pass
            test_preds = model(X_test).squeeze(dim=1)

            # Compute test loss
            test_loss = loss_module(test_preds, Y_test.float())
            history["test_loss"].append(test_loss.item())

            test_labels = Y_test.long()
            test_correct = (_binary_predictions(test_preds) == test_labels).sum().item()
            test_total = test_labels.numel()
            history["test_accuracy"].append(test_correct / test_total if test_total > 0 else 0.0)

    if model_save_path:
        save_dir = os.path.dirname(model_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        history["saved_model_path"] = model_save_path
    else:
        history["saved_model_path"] = None

    return history


def plot_training_history(history):
    """Plot loss and accuracy curves from the history returned by train_model_epoch_eval."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["test_loss"], label="Test loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over epochs")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train accuracy")
    axes[1].plot(epochs, history["test_accuracy"], label="Test accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy over epochs")
    axes[1].legend()

    fig.tight_layout()
    plt.savefig("training_history.png")


