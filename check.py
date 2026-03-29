from pathlib import Path


AA_VOCAB = set("ACDEFGHIKLMNPQRSTVWY")


def read_sequences(file_path: Path) -> set[str]:
	"""Read valid 9-mer amino-acid sequences from a file."""
	sequences = set()
	for line in file_path.read_text().splitlines():
		seq = line.strip().upper()
		if len(seq) == 9 and all(ch in AA_VOCAB for ch in seq):
			sequences.add(seq)
	return sequences


def find_shared_sequences(data_dir: Path) -> dict[str, set[str]]:
	"""Return mapping from sequence to positive files where it appears (2+ files only)."""
	sequence_to_files: dict[str, set[str]] = {}

	for file_path in sorted(data_dir.glob("*_pos.txt")):
		file_sequences = read_sequences(file_path)
		for seq in file_sequences:
			if seq not in sequence_to_files:
				sequence_to_files[seq] = set()
			sequence_to_files[seq].add(file_path.name)

	return {
		seq: files
		for seq, files in sequence_to_files.items()
		if len(files) > 1
	}


def _read_valid_lines(file_path: Path) -> list[str]:
	"""Read valid 9-mer sequences preserving original file order."""
	sequences: list[str] = []
	for line in file_path.read_text().splitlines():
		seq = line.strip().upper()
		if len(seq) == 9 and all(ch in AA_VOCAB for ch in seq):
			sequences.append(seq)
	return sequences


def write_deduplicated_positive_files(data_dir: Path) -> dict[str, dict[str, int]]:
	"""Write *_pos_dedup.txt files after removing sequences shared across positive files."""
	shared = find_shared_sequences(data_dir)
	shared_sequences = set(shared.keys())
	stats: dict[str, dict[str, int]] = {}

	for pos_path in sorted(data_dir.glob("*_pos.txt")):
		valid_sequences = _read_valid_lines(pos_path)
		filtered = [seq for seq in valid_sequences if seq not in shared_sequences]
		out_path = pos_path.with_name(pos_path.name.replace("_pos.txt", "_pos_dedup.txt"))
		out_path.write_text("\n".join(filtered) + ("\n" if filtered else ""))

		stats[pos_path.name] = {
			"original_valid": len(valid_sequences),
			"removed_shared": len(valid_sequences) - len(filtered),
			"remaining": len(filtered),
		}

	return stats


def main() -> None:
	data_dir = Path("ex1 data")
	if not data_dir.exists():
		raise FileNotFoundError(f"Data directory not found: {data_dir}")

	stats = write_deduplicated_positive_files(data_dir)
	shared = find_shared_sequences(data_dir)

	print("Wrote deduplicated positive files (*_pos_dedup.txt).")
	for file_name in sorted(stats):
		file_stats = stats[file_name]
		print(
			f"{file_name}: original={file_stats['original_valid']}, "
			f"removed_shared={file_stats['removed_shared']}, remaining={file_stats['remaining']}"
		)
	print()

	if not shared:
		print("No shared sequences found across positive files.")
		return

	print(f"Found {len(shared)} sequences that appear in more than one positive file.\n")
	for seq in sorted(shared):
		files = ", ".join(sorted(shared[seq]))
		print(f"{seq}: {files}")


if __name__ == "__main__":
	main()

