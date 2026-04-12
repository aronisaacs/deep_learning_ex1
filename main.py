import csv
import json
import secrets
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset

from data_loaders import create_data_loaders, create_basic_data_loaders
from training import train_model_epoch_eval
from evaluators import (
    EvaluatorHolder,
    MultiClassAccuracyEvaluator,
    PerLabelAccuracyEvaluator,
    PosNegAccuracyEvaluator,
    PrecisionEvaluator,
    RecallEvaluator,
    PositiveSamplesAverageEvaluator,
    ClassDistributionEvaluator,
)
from generate_multihot_labels import build_records, write_csv
from model import AminoAcidNet
from better_model import BetterAminoAcidNet
from peptide_cnn import PeptideCNN

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_VOCAB)}
NUM_CLASSES_ORIGINAL = 7
NUM_CLASSES_MULTIHOT = 6


def encode_sequence(seq: str) -> list[int]:
    return [AA_TO_INDEX[ch] for ch in seq]


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


def make_original_datasets(data_dir: Path, test_ratio: float, seed: int):
    """Create 7-class dataset from original pos/neg files (no dedup)."""
    rng = np.random.default_rng(seed)

    pos_paths = sorted(p for p in data_dir.glob("*_pos.txt") if not p.name.endswith("_dedup.txt"))
    if len(pos_paths) != NUM_CLASSES_ORIGINAL - 1:
        raise ValueError(
            f"Expected {NUM_CLASSES_ORIGINAL - 1} positive files, found {len(pos_paths)}"
        )
    neg_path = data_dir / "negs.txt"

    train_rows, train_labels, test_rows, test_labels = [], [], [], []

    for class_idx, pos_path in enumerate(pos_paths):
        sequences = read_sequences(pos_path)
        tr_r, tr_l, te_r, te_l = split_source_sequences(
            sequences=sequences,
            class_label=class_idx,
            source_name=pos_path.name,
            test_ratio=test_ratio,
            rng=rng,
        )
        train_rows.extend(tr_r)
        train_labels.extend(tr_l)
        test_rows.extend(te_r)
        test_labels.extend(te_l)

    neg_sequences = read_sequences(neg_path)
    tr_r, tr_l, te_r, te_l = split_source_sequences(
        sequences=neg_sequences,
        class_label=NUM_CLASSES_ORIGINAL - 1,
        source_name=neg_path.name,
        test_ratio=test_ratio,
        rng=rng,
    )
    train_rows.extend(tr_r)
    train_labels.extend(tr_l)
    test_rows.extend(te_r)
    test_labels.extend(te_l)

    train_x = torch.tensor([encode_sequence(seq) for seq, _ in train_rows], dtype=torch.long)
    test_x = torch.tensor([encode_sequence(seq) for seq, _ in test_rows], dtype=torch.long)
    train_y_indices = torch.tensor(train_labels, dtype=torch.long)
    test_y_indices = torch.tensor(test_labels, dtype=torch.long)

    train_y = torch.nn.functional.one_hot(train_y_indices, num_classes=NUM_CLASSES_ORIGINAL).float()
    test_y = torch.nn.functional.one_hot(test_y_indices, num_classes=NUM_CLASSES_ORIGINAL).float()

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    split_stats = {
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "train_label_counts": {i: int((train_y_indices == i).sum().item()) for i in range(NUM_CLASSES_ORIGINAL)},
        "test_label_counts": {i: int((test_y_indices == i).sum().item()) for i in range(NUM_CLASSES_ORIGINAL)},
    }

    return train_dataset, test_dataset, train_y_indices, split_stats


def load_multihot_rows(csv_path: Path) -> list[tuple[str, list[float], int]]:
    rows: list[tuple[str, list[float], int]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row["sequence"].strip().upper()
            if len(seq) != 9 or any(ch not in AA_TO_INDEX for ch in seq):
                continue
            labels = [float(row[f"label_{i}"]) for i in range(NUM_CLASSES_MULTIHOT)]
            in_negative = int(row.get("in_negative", "0"))
            rows.append((seq, labels, in_negative))
    return rows


def split_multihot_rows(
    rows: list[tuple[str, list[float], int]],
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, list[float], int]], list[tuple[str, list[float], int]]]:
    """Split while preserving positive-only vs negative-only distribution."""
    rng = np.random.default_rng(seed)
    positive_rows = [r for r in rows if sum(r[1]) > 0]
    negative_rows = [r for r in rows if sum(r[1]) == 0]

    def _split(group):
        if len(group) <= 1:
            return group, []
        indices = rng.permutation(len(group))
        test_count = int(round(len(group) * test_ratio))
        test_count = min(max(test_count, 1), len(group) - 1)
        test_idx = set(indices[:test_count].tolist())
        train_group = [item for i, item in enumerate(group) if i not in test_idx]
        test_group = [item for i, item in enumerate(group) if i in test_idx]
        return train_group, test_group

    train_pos, test_pos = _split(positive_rows)
    train_neg, test_neg = _split(negative_rows)

    train_rows = train_pos + train_neg
    test_rows = test_pos + test_neg
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def make_multihot_datasets(data_dir: Path, test_ratio: float, seed: int):
    """Create 6-dim multi-hot dataset from negs_filtered.txt."""
    csv_path = Path("artifacts/multihot_labels.csv")
    records, pos_names = build_records(data_dir)
    write_csv(records, pos_names, csv_path)

    rows = load_multihot_rows(csv_path)
    train_rows, test_rows = split_multihot_rows(rows, test_ratio=test_ratio, seed=seed)

    train_x = torch.tensor([encode_sequence(seq) for seq, _, _ in train_rows], dtype=torch.long)
    test_x = torch.tensor([encode_sequence(seq) for seq, _, _ in test_rows], dtype=torch.long)
    train_y = torch.tensor([labels for _, labels, _ in train_rows], dtype=torch.float32)
    test_y = torch.tensor([labels for _, labels, _ in test_rows], dtype=torch.float32)

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    split_stats = {
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "train_positive_rows": int((train_y.sum(dim=1) > 0).sum().item()),
        "train_negative_rows": int((train_y.sum(dim=1) == 0).sum().item()),
        "test_positive_rows": int((test_y.sum(dim=1) > 0).sum().item()),
        "test_negative_rows": int((test_y.sum(dim=1) == 0).sum().item()),
        "train_label_positive_counts": [int(train_y[:, i].sum().item()) for i in range(NUM_CLASSES_MULTIHOT)],
        "test_label_positive_counts": [int(test_y[:, i].sum().item()) for i in range(NUM_CLASSES_MULTIHOT)],
    }
    return train_dataset, test_dataset, split_stats


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_and_evaluate(
    model,
    optimizer,
    loss_module,
    evaluator_holder,
    train_loader,
    train_eval_loader,
    test_loader,
    num_epochs,
    device,
    seed,
):
    model_name = model.__class__.__name__
    run_id = model_name
    model_save_path = f"artifacts/model_{model_name}_seed{seed}.pt"
    history_path = f"artifacts/history_{model_name}_seed{seed}.json"

    print(f"\n{'=' * 60}")
    print(f"Training {model_name}")
    print(f"{'=' * 60}")
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
    seed = secrets.randbelow(2**32)
    num_epochs = 50
    batch_size = 256
    lr = 1e-3
    device = choose_device()

    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Run seed: {seed}")

    # Train AminoAcidNet with original 7-class dataset
    print("\n" + "=" * 60)
    print("Loading original 7-class dataset for AminoAcidNet")
    print("=" * 60)
    train_dataset_orig, test_dataset_orig, train_y_indices, split_stats_orig = make_original_datasets(
        data_dir=data_dir,
        test_ratio=test_ratio,
        seed=seed,
    )
    print("Split stats (original):", split_stats_orig)

    train_loader_orig, train_eval_loader_orig, test_loader_orig = create_data_loaders(
        train_dataset=train_dataset_orig,
        test_dataset=test_dataset_orig,
        train_labels=train_y_indices,
        batch_size=batch_size,
        num_classes=NUM_CLASSES_ORIGINAL,
    )

    model = AminoAcidNet(output_dim=NUM_CLASSES_ORIGINAL)
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
    # train_and_evaluate(
    #     model=model,
    #     optimizer=optim.Adam(model.parameters(), lr=lr),
    #     loss_module=loss_module,
    #     evaluator_holder=evaluator_holder,
    #     train_loader=train_loader_orig,
    #     train_eval_loader=train_eval_loader_orig,
    #     test_loader=test_loader_orig,
    #     num_epochs=num_epochs,
    #     device=device,
    #     seed=seed,
    # )

    # Train BetterAminoAcidNet with 6-dim multi-hot dataset
    print("\n" + "=" * 60)
    print("Loading 6-dim multi-hot dataset for BetterAminoAcidNet")
    print("=" * 60)
    train_dataset_multi, test_dataset_multi, split_stats_multi = make_multihot_datasets(
        data_dir=data_dir,
        test_ratio=test_ratio,
        seed=seed,
    )
    print("Split stats (multi-hot):", split_stats_multi)

    train_loader_multi, train_eval_loader_multi, test_loader_multi = create_basic_data_loaders(
        train_dataset=train_dataset_multi,
        test_dataset=test_dataset_multi,
        batch_size=batch_size,
    )

    model = BetterAminoAcidNet(output_dim=NUM_CLASSES_MULTIHOT)
    loss_module = nn.BCELoss()
    evaluator_holder = EvaluatorHolder(
        evaluators=[
            MultiClassAccuracyEvaluator(),
            PosNegAccuracyEvaluator(),
            PerLabelAccuracyEvaluator(),
            PrecisionEvaluator(),
            RecallEvaluator(),
            PositiveSamplesAverageEvaluator(),
            ClassDistributionEvaluator(),
        ],
        device=device,
        loss_module=loss_module,
    )
    # train_and_evaluate(
    #     model=model,
    #     optimizer=optim.Adam(model.parameters(), lr=lr),
    #     loss_module=loss_module,
    #     evaluator_holder=evaluator_holder,
    #     train_loader=train_loader_multi,
    #     train_eval_loader=train_eval_loader_multi,
    #     test_loader=test_loader_multi,
    #     num_epochs=num_epochs,
    #     device=device,
    #     seed=seed,
    # )

    # Train PeptideCNN with 6-dim multi-hot dataset
    print("\n" + "=" * 60)
    print("Training PeptideCNN with 6-dim multi-hot dataset")
    print("=" * 60)

    model_cnn = PeptideCNN(vocab_size=len(AA_VOCAB) + 1, output_dim=NUM_CLASSES_MULTIHOT)  # +1 for padding/unknown
    pos_weight=torch.tensor([1.5, 1.5, 1.5, 1.5, 10, 1.5])  # Adjust as needed for class imbalance
    loss_module_cnn = nn.BCEWithLogitsLoss()
    evaluator_holder_cnn = EvaluatorHolder(
        evaluators=[
            MultiClassAccuracyEvaluator(),
            PosNegAccuracyEvaluator(),
            PerLabelAccuracyEvaluator(),
            PrecisionEvaluator(),
            RecallEvaluator(),
            PositiveSamplesAverageEvaluator(),
            ClassDistributionEvaluator(),
        ],
        device=device,
        loss_module=loss_module_cnn,
    )
    train_and_evaluate(
        model=model_cnn,
        optimizer=optim.Adam(model_cnn.parameters(), lr=lr),
        loss_module=loss_module_cnn,
        evaluator_holder=evaluator_holder_cnn,
        train_loader=train_loader_multi,
        train_eval_loader=train_eval_loader_multi,
        test_loader=test_loader_multi,
        num_epochs=num_epochs,
        device=device,
        seed=seed,
    )

main()