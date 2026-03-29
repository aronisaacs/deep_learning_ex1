from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


AA_VOCAB = set("ACDEFGHIKLMNPQRSTVWY")


def read_valid_sequences(file_path: Path) -> list[str]:
    """Read 9-mer amino-acid sequences and drop invalid rows."""
    sequences: list[str] = []
    for line in file_path.read_text().splitlines():
        seq = line.strip().upper()
        if len(seq) == 9 and all(ch in AA_VOCAB for ch in seq):
            sequences.append(seq)
    return sequences


def find_positive_files(data_dir: Path) -> list[Path]:
    pos_files = sorted(
        p for p in data_dir.glob("*_pos.txt") if not p.name.endswith("_dedup.txt")
    )
    if len(pos_files) != 6:
        raise ValueError(f"Expected 6 positive files (*_pos.txt), found {len(pos_files)}")
    return pos_files


def build_records(data_dir: Path) -> tuple[list[dict], list[str]]:
    pos_files = find_positive_files(data_dir)
    neg_file = data_dir / "negs_filtered.txt"
    if not neg_file.exists():
        raise FileNotFoundError(f"Missing negative file: {neg_file}")

    seq_to_labels: dict[str, list[int]] = {}
    seq_to_pos_names: dict[str, set[str]] = {}
    seq_in_negative: dict[str, int] = {}

    for label_idx, path in enumerate(pos_files):
        for seq in read_valid_sequences(path):
            if seq not in seq_to_labels:
                seq_to_labels[seq] = [0] * len(pos_files)
                seq_to_pos_names[seq] = set()
                seq_in_negative[seq] = 0
            seq_to_labels[seq][label_idx] = 1
            seq_to_pos_names[seq].add(path.name)

    for seq in read_valid_sequences(neg_file):
        if seq not in seq_to_labels:
            seq_to_labels[seq] = [0] * len(pos_files)
            seq_to_pos_names[seq] = set()
        seq_in_negative[seq] = 1

    records: list[dict] = []
    for seq in sorted(seq_to_labels):
        records.append(
            {
                "sequence": seq,
                "labels": seq_to_labels[seq],
                "positive_files": sorted(seq_to_pos_names[seq]),
                "in_negative": int(seq_in_negative.get(seq, 0)),
            }
        )

    return records, [p.name for p in pos_files]


def write_csv(records: list[dict], pos_file_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["sequence", *[f"label_{i}" for i in range(6)], "in_negative", "positive_files"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            labels = row["labels"]
            writer.writerow(
                {
                    "sequence": row["sequence"],
                    "label_0": labels[0],
                    "label_1": labels[1],
                    "label_2": labels[2],
                    "label_3": labels[3],
                    "label_4": labels[4],
                    "label_5": labels[5],
                    "in_negative": row["in_negative"],
                    "positive_files": ";".join(row["positive_files"]),
                }
            )

    print(f"Wrote CSV: {output_path}")
    print("Label index mapping:")
    for i, name in enumerate(pos_file_names):
        print(f"  label_{i}: {name}")


def write_json(records: list[dict], pos_file_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label_mapping": {f"label_{i}": name for i, name in enumerate(pos_file_names)},
        "records": records,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote JSON: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 6-dim multi-hot labels from six positive files and negs_filtered.txt."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("ex1 data"),
        help="Directory containing *_pos.txt and negs_filtered.txt",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("artifacts/multihot_labels.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/multihot_labels.json"),
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, pos_file_names = build_records(args.data_dir)
    write_csv(records, pos_file_names, args.csv_out)
    write_json(records, pos_file_names, args.json_out)


if __name__ == "__main__":
    main()

