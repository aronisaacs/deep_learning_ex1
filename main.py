import json
import secrets
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset

from data_loaders import create_data_loaders
from training import train_model_epoch_eval
from check import write_deduplicated_positive_files
from evaluators import (
    EvaluatorHolder,
    MultiClassAccuracyEvaluator,
    PerLabelAccuracyEvaluator,
    PosNegAccuracyEvaluator,
)
from model import AminoAcidNet
from better_model import BetterAminoAcidNet

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_VOCAB)}

NUM_CLASSES = 7  # 6 positive files + 1 negative class


def read_sequences(file_path: Path) -> list[str]:
    lines = [line.strip().upper() for line in file_path.read_text().splitlines()]
    return [seq for seq in lines if len(seq) == 9 and all(ch in AA_TO_INDEX for ch in seq)]


def split_source_sequences(
    sequences: list[str],
    class_label: int,
    source_name: str,
    test_ratio: float,
    rng: np.random.Generator,
):
    if not sequences:
        return [], [], [], []

    indices = rng.permutation(len(sequences))
    test_count = int(round(len(sequences) * test_ratio))
    test_count = min(max(test_count, 1), len(sequences) - 1)

    test_idx = set(indices[:test_count].tolist())
    train_rows, test_rows = [], []
    train_labels, test_labels = [], []

    for i, seq in enumerate(sequences):
        row = (seq, source_name)
        if i in test_idx:
            test_rows.append(row)
            test_labels.append(class_label)
        else:
            train_rows.append(row)
            train_labels.append(class_label)

    return train_rows, train_labels, test_rows, test_labels


def encode_sequences(rows: list[tuple[str, str]]) -> torch.Tensor:
    encoded = [[AA_TO_INDEX[ch] for ch in seq] for seq, _ in rows]
    return torch.tensor(encoded, dtype=torch.long)


def to_one_hot(class_indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert class indices to one-hot encoded vectors."""
    return torch.nn.functional.one_hot(class_indices, num_classes=num_classes).float()


def make_datasets(data_dir: Path, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)

    pos_paths = sorted(data_dir.glob("*_pos_dedup.txt"))
    if len(pos_paths) != NUM_CLASSES - 1:
        raise ValueError(
            f"Expected {NUM_CLASSES - 1} deduplicated positive files, found {len(pos_paths)}"
        )
    neg_path = data_dir / "negs_filtered.txt"

    train_rows, train_labels, test_rows, test_labels = [], [], [], []

    # Assign unique class labels 0-5 to each positive file
    for class_idx, pos_path in enumerate(pos_paths):
        sequences = read_sequences(pos_path)
        tr_r, tr_l, te_r, te_l = split_source_sequences(
            sequences=sequences,
            class_label=class_idx,  # 0 to 5
            source_name=pos_path.name,
            test_ratio=test_ratio,
            rng=rng,
        )
        train_rows.extend(tr_r)
        train_labels.extend(tr_l)
        test_rows.extend(te_r)
        test_labels.extend(te_l)

    # Assign label 6 to negatives
    neg_sequences = read_sequences(neg_path)
    tr_r, tr_l, te_r, te_l = split_source_sequences(
        sequences=neg_sequences,
        class_label=6,  # Negative class
        source_name=neg_path.name,
        test_ratio=test_ratio,
        rng=rng,
    )
    train_rows.extend(tr_r)
    train_labels.extend(tr_l)
    test_rows.extend(te_r)
    test_labels.extend(te_l)

    train_x = encode_sequences(train_rows)
    test_x = encode_sequences(test_rows)
    train_y_indices = torch.tensor(train_labels, dtype=torch.long)
    test_y_indices = torch.tensor(test_labels, dtype=torch.long)
    
    # Convert to one-hot encoding
    train_y = to_one_hot(train_y_indices, NUM_CLASSES)
    test_y = to_one_hot(test_y_indices, NUM_CLASSES)

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    split_stats = {
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "train_label_counts": {i: int((train_y_indices == i).sum().item()) for i in range(NUM_CLASSES)},
        "test_label_counts": {i: int((test_y_indices == i).sum().item()) for i in range(NUM_CLASSES)},
    }

    return train_dataset, test_dataset, train_y_indices, split_stats


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_and_evaluate(
    model,
    train_loader,
    train_eval_loader,
    test_loader,
    num_epochs,
    lr,
    device,
    seed,
):
    """Train a model, evaluate it, save results and plots."""
    model_name = model.__class__.__name__
    run_id = f"{model_name}_seed{seed}"
    model_save_path = f"artifacts/model_{run_id}.pt"
    history_path = f"artifacts/history_{run_id}.json"

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_module = nn.CrossEntropyLoss()
    evaluator_holder = EvaluatorHolder(
        evaluators=[
            MultiClassAccuracyEvaluator(),
            PosNegAccuracyEvaluator(),
            PerLabelAccuracyEvaluator(),
        ],
        device=device,
        loss_module=loss_module,
    )

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    trained_evaluator_holder = train_model_epoch_eval(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        test_loader=test_loader,
        loss_module=loss_module,
        num_epochs=num_epochs,
        device=device,
        evaluator_holder=evaluator_holder,
        model_save_path=model_save_path,
    )
    history = trained_evaluator_holder.history

    print(f"\nFinal {model_name} evaluator metrics:")
    trained_evaluator_holder.print_evaluator_results()

    history_path_obj = Path(history_path)
    history_path_obj.parent.mkdir(parents=True, exist_ok=True)
    history["seed"] = seed
    history_path_obj.write_text(json.dumps(history, indent=2))
    print(f"Saved history to {history_path}")

    trained_evaluator_holder.plot_evaluators(
        output_dir="artifacts",
        filename_prefix=run_id,
    )
    print(f"Saved plots with prefix '{run_id}' to artifacts/")


def main():
    data_dir = Path("ex1 data")
    test_ratio = 0.1
    # Generate a new seed every run so data split/training vary by default.
    seed = secrets.randbelow(2**32)
    num_epochs = 50
    batch_size = 256
    lr = 1e-3
    device = choose_device()

    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Run seed: {seed}")

    dedup_stats = write_deduplicated_positive_files(data_dir)
    print("Wrote deduplicated positive files (*_pos_dedup.txt)")
    for file_name in sorted(dedup_stats):
        s = dedup_stats[file_name]
        print(
            f"  {file_name}: original={s['original_valid']}, "
            f"removed_shared={s['removed_shared']}, remaining={s['remaining']}"
        )

    train_dataset, test_dataset, train_y_indices, split_stats = make_datasets(
        data_dir=data_dir,
        test_ratio=test_ratio,
        seed=seed,
    )

    print("Split stats:", split_stats)

    train_loader, train_eval_loader, test_loader = create_data_loaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_labels=train_y_indices,
        batch_size=batch_size,
        num_classes=NUM_CLASSES,
    )
    print("Created WeightedRandomSampler for balanced training")

    # Train both models
    train_and_evaluate(
        model=AminoAcidNet(),
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        seed=seed,
    )

    train_and_evaluate(
        model=BetterAminoAcidNet(),
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        seed=seed,
    )


if __name__ == "__main__":
    main()

