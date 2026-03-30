from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from plotters import plot_per_label_panels, plot_train_test_curves, plot_train_test_loss_curves


def _multi_class_predictions(raw_outputs: torch.Tensor) -> torch.Tensor:
    return torch.argmax(raw_outputs, dim=1)


def _multilabel_predictions(raw_outputs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (raw_outputs >= threshold).to(raw_outputs.dtype)


def _with_prefix(filename: str, filename_prefix: str = "") -> str:
    if not filename_prefix:
        return filename
    return f"{filename_prefix}_{filename}"


def _prepare_targets_for_loss(loss_module, labels: torch.Tensor) -> torch.Tensor:
    """Normalize target format for common loss modules."""
    if isinstance(loss_module, torch.nn.CrossEntropyLoss):
        # CrossEntropyLoss expects class indices, not one-hot vectors.
        return torch.argmax(labels, dim=1).long() if labels.ndim > 1 else labels.long()
    if isinstance(loss_module, (torch.nn.BCEWithLogitsLoss, torch.nn.BCELoss)):
        return labels.float()
    return labels.float()


class AbstractEvaluator(ABC):
    key = "metric"

    def __init__(self):
        self.train_history: list[Any] = []
        self.test_history: list[Any] = []

    @abstractmethod
    def update(
        self,
        raw_outputs: torch.Tensor,
        labels: torch.Tensor,
        split: str,
        value: Any | None = None,
    ) -> None:
        """Update evaluator state for one split. split is either 'train' or 'test'."""

    @abstractmethod
    def plot(self, output_dir: str = ".", filename_prefix: str = "") -> None:
        """Plot evaluator history and save figures under output_dir."""

    @abstractmethod
    def print_final_result(self) -> None:
        """Print the final train/test result tracked by this evaluator."""


class MultiClassAccuracyEvaluator(AbstractEvaluator):
    key = "accuracy"

    def update(
        self,
        raw_outputs: torch.Tensor,
        labels: torch.Tensor,
        split: str,
        value: Any | None = None,
    ) -> None:
        batch_labels = labels.float()
        batch_preds = _multilabel_predictions(raw_outputs)
        # Exact-match accuracy for multi-label vectors.
        value = (batch_preds == batch_labels).all(dim=1).float().mean().item()
        if split == "train":
            self.train_history.append(value)
        elif split == "test":
            self.test_history.append(value)
        else:
            raise ValueError(f"Unsupported split: {split}")

    def plot(self, output_dir: str = ".", filename_prefix: str = "") -> None:
        if not self.train_history and not self.test_history:
            return
        plot_train_test_curves(
            self.train_history,
            self.test_history,
            train_label="Train accuracy",
            test_label="Test accuracy",
            title="Multi-class Accuracy",
            output_dir=output_dir,
            filename=_with_prefix("training_history_accuracy.png", filename_prefix),
        )

    def print_final_result(self) -> None:
        train_final = f"{self.train_history[-1]:.4f}" if self.train_history else "N/A"
        test_final = f"{self.test_history[-1]:.4f}" if self.test_history else "N/A"
        print(f"  {self.key}: train={train_final}, test={test_final}")


class PosNegAccuracyEvaluator(AbstractEvaluator):
    key = "posneg_accuracy"

    def update(
        self,
        raw_outputs: torch.Tensor,
        labels: torch.Tensor,
        split: str,
        value: Any | None = None,
    ) -> None:
        batch_labels = labels.float()
        batch_preds = _multilabel_predictions(raw_outputs)
        # Positive means at least one positive label bit is set.
        binary_preds = (batch_preds.sum(dim=1) > 0)
        binary_labels = (batch_labels.sum(dim=1) > 0)
        value = (binary_preds == binary_labels).float().mean().item()
        if split == "train":
            self.train_history.append(value)
        elif split == "test":
            self.test_history.append(value)
        else:
            raise ValueError(f"Unsupported split: {split}")

    def plot(self, output_dir: str = ".", filename_prefix: str = "") -> None:
        if not self.train_history and not self.test_history:
            return
        if self.train_history:
            plot_train_test_curves(
                self.train_history,
                [],
                train_label="Train pos/neg accuracy",
                test_label="",
                title="Train Positive-vs-Negative Accuracy",
                output_dir=output_dir,
                filename=_with_prefix("training_history_posneg_accuracy_train.png", filename_prefix),
            )

        if self.test_history:
            plot_train_test_curves(
                self.test_history,
                [],
                train_label="Test pos/neg accuracy",
                test_label="",
                title="Test Positive-vs-Negative Accuracy",
                output_dir=output_dir,
                filename=_with_prefix("training_history_posneg_accuracy_test.png", filename_prefix),
            )

    def print_final_result(self) -> None:
        train_final = f"{self.train_history[-1]:.4f}" if self.train_history else "N/A"
        test_final = f"{self.test_history[-1]:.4f}" if self.test_history else "N/A"
        print(f"  {self.key}: train={train_final}, test={test_final}")


class PerLabelAccuracyEvaluator(AbstractEvaluator):
    key = "label_accuracy"

    def update(
        self,
        raw_outputs: torch.Tensor,
        labels: torch.Tensor,
        split: str,
        value: Any | None = None,
    ) -> None:
        batch_labels = labels.float()
        batch_preds = _multilabel_predictions(raw_outputs)
        per_label_correct = (batch_preds == batch_labels).float().mean(dim=0)
        value = per_label_correct.detach().cpu().tolist()

        if split == "train":
            self.train_history.append(value)
        elif split == "test":
            self.test_history.append(value)
        else:
            raise ValueError(f"Unsupported split: {split}")

    def plot(self, output_dir: str = ".", filename_prefix: str = "") -> None:
        if not self.train_history or not self.test_history:
            return

        train_label_acc = np.array(self.train_history, dtype=float)
        test_label_acc = np.array(self.test_history, dtype=float)
        plot_per_label_panels(
            train_label_acc,
            test_label_acc,
            output_dir=output_dir,
            filename=_with_prefix("training_history_per_label_accuracy.png", filename_prefix),
        )

    def print_final_result(self) -> None:
        if self.train_history:
            train_final = "[" + ", ".join(f"{v:.4f}" for v in self.train_history[-1]) + "]"
        else:
            train_final = "N/A"
        if self.test_history:
            test_final = "[" + ", ".join(f"{v:.4f}" for v in self.test_history[-1]) + "]"
        else:
            test_final = "N/A"
        print(f"  {self.key}: train={train_final}, test={test_final}")


class LossEvaluator(AbstractEvaluator):
    key = "loss"

    def __init__(self, loss_module=None):
        super().__init__()
        self.loss_module = loss_module

    def update(
        self,
        raw_outputs: torch.Tensor,
        labels: torch.Tensor,
        split: str,
        value: Any | None = None,
    ) -> None:
        if value is not None:
            numeric_value = float(value)
        else:
            if self.loss_module is None:
                raise ValueError("LossEvaluator requires loss_module to compute loss")
            if raw_outputs.numel() == 0:
                numeric_value = 0.0
            else:
                targets = _prepare_targets_for_loss(self.loss_module, labels)
                numeric_value = float(self.loss_module(raw_outputs, targets).item())
        if split == "train":
            self.train_history.append(numeric_value)
        elif split == "test":
            self.test_history.append(numeric_value)
        else:
            raise ValueError(f"Unsupported split: {split}")

    def plot(self, output_dir: str = ".", filename_prefix: str = "") -> None:
        if not self.train_history and not self.test_history:
            return
        plot_train_test_loss_curves(
            self.train_history,
            self.test_history,
            output_dir=output_dir,
            filename=_with_prefix("training_history_loss.png", filename_prefix),
        )

    def print_final_result(self) -> None:
        train_final = f"{self.train_history[-1]:.4f}" if self.train_history else "N/A"
        test_final = f"{self.test_history[-1]:.4f}" if self.test_history else "N/A"
        print(f"  {self.key}: train={train_final}, test={test_final}")


class EvaluatorHolder:
    """Collects losses and delegates metric updates to configured evaluators."""

    def __init__(
        self,
        evaluators: list[AbstractEvaluator] | None = None,
        device: str = "cpu",
        loss_module=None,
    ):
        self.loss_evaluator = LossEvaluator(loss_module=loss_module)
        self.evaluators = [self.loss_evaluator, *(evaluators or [])]
        self.device = device
        self.history: dict[str, Any] = {"saved_model_path": None}
        for evaluator in self.evaluators:
            self.history[f"train_{evaluator.key}"] = []
            self.history[f"test_{evaluator.key}"] = []

    def _collect_outputs(self, model, data_loader):
        model.eval()
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                preds = model(inputs).squeeze(dim=1)
                all_outputs.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())

        outputs = torch.cat(all_outputs, dim=0) if all_outputs else torch.empty((0, 0))
        targets = torch.cat(all_labels, dim=0) if all_labels else torch.empty((0, 0))
        return outputs, targets

    def set_loss_module(self, loss_module) -> None:
        self.loss_evaluator.loss_module = loss_module

    def update(self, model, train_loader, test_loader):
        train_outputs, train_labels = self._collect_outputs(model, train_loader)
        test_outputs, test_labels = self._collect_outputs(model, test_loader)

        for evaluator in self.evaluators:
            evaluator.update(train_outputs, train_labels, split="train")
            evaluator.update(test_outputs, test_labels, split="test")
            self.history[f"train_{evaluator.key}"].append(evaluator.train_history[-1])
            self.history[f"test_{evaluator.key}"].append(evaluator.test_history[-1])

    def plot_evaluators(self, output_dir: str = ".", filename_prefix: str = "") -> None:
        """Ask each evaluator to render its own plots."""
        for evaluator in self.evaluators:
            evaluator.plot(output_dir=output_dir, filename_prefix=filename_prefix)

    def print_evaluator_results(self) -> None:
        """Ask each evaluator to print its final result."""
        for evaluator in self.evaluators:
            evaluator.print_final_result()


