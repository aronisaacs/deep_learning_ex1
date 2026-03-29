from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def epochs_for(*series) -> np.ndarray:
    """Build an epoch index that fits the longest provided series."""
    return np.arange(max((len(s) for s in series), default=0))


def style_accuracy_axis(
    ax,
    *,
    title: str,
    xlabel: str = "Epoch",
    ylabel: str = "Accuracy",
    y_min: float = 0.0,
    y_max: float = 1.0,
    grid_alpha: float = 0.3,
    legend: bool = True,
    legend_kwargs: dict | None = None,
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.grid(True, alpha=grid_alpha)
    if legend and ax.lines:
        ax.legend(**(legend_kwargs or {}))


def save_figure(fig, output_dir: str, filename: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_dir / filename))


def plot_train_test_curves(
    train_values,
    test_values,
    *,
    train_label: str,
    test_label: str,
    title: str,
    output_dir: str,
    filename: str,
) -> None:
    epochs = epochs_for(train_values, test_values)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    if len(train_values) > 0:
        ax.plot(epochs[: len(train_values)], train_values, label=train_label)
    if len(test_values) > 0:
        ax.plot(epochs[: len(test_values)], test_values, label=test_label)
    style_accuracy_axis(ax, title=title)
    save_figure(fig, output_dir, filename)


def plot_train_test_loss_curves(
    train_values,
    test_values,
    *,
    train_label: str = "Train loss",
    test_label: str = "Test loss",
    title: str = "Loss over epochs (includes epoch 0 baseline)",
    output_dir: str,
    filename: str,
) -> None:
    epochs = epochs_for(train_values, test_values)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    if len(train_values) > 0:
        ax.plot(epochs[: len(train_values)], train_values, label=train_label)
    if len(test_values) > 0:
        ax.plot(epochs[: len(test_values)], test_values, label=test_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()
    save_figure(fig, output_dir, filename)


def plot_per_label_panels(
    train_label_acc: np.ndarray,
    test_label_acc: np.ndarray,
    *,
    output_dir: str,
    filename: str,
) -> None:
    epochs = np.arange(train_label_acc.shape[0])
    num_labels = train_label_acc.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for label_idx in range(num_labels):
        axes[0].plot(epochs, train_label_acc[:, label_idx], label=f"Label {label_idx}")
        axes[1].plot(epochs, test_label_acc[:, label_idx], label=f"Label {label_idx}")

    style_accuracy_axis(
        axes[0],
        title="Train Per-Label Accuracy",
        legend_kwargs={"loc": "best", "ncol": 2, "fontsize": 8},
    )
    style_accuracy_axis(
        axes[1],
        title="Test Per-Label Accuracy",
        legend_kwargs={"loc": "best", "ncol": 2, "fontsize": 8},
    )
    save_figure(fig, output_dir, filename)

