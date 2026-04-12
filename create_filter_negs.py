import os

# Define the data directory
data_dir = 'ex1 data'

# List of positive files
pos_files = [
    'A0101_pos.txt',
    'A0201_pos.txt',
    'A0203_pos.txt',
    'A0207_pos.txt',
    'A0301_pos.txt',
    'A2402_pos.txt'
]

# Read all positive sequences into a set for fast lookup
pos_sequences = set()
for pos_file in pos_files:
    pos_path = os.path.join(data_dir, pos_file)
    with open(pos_path, 'r') as f:
        for line in f:
            seq = line.strip()
            if seq:  # Skip empty lines
                pos_sequences.add(seq)

# Read negative sequences and filter
negs_path = os.path.join(data_dir, 'negs.txt')
filtered_negs = []
with open(negs_path, 'r') as f:
    for line in f:
        seq = line.strip()
        if seq and seq not in pos_sequences:
            filtered_negs.append(seq)

# Write filtered negatives to new file
filtered_path = os.path.join(data_dir, 'negs_filtered.txt')
with open(filtered_path, 'w') as f:
    for seq in filtered_negs:
        f.write(seq + '\n')

print(f"Filtered {len(filtered_negs)} negative sequences out of {len(filtered_negs) + len(pos_sequences)} total negatives.")