import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from better_model import BetterAminoAcidNet

# Constants from main.py
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_VOCAB)}
ALLELES = ['A0101', 'A0201', 'A0203', 'A0207', 'A0301', 'A2402']

def encode_sequence(seq: str) -> list[int]:
    return [AA_TO_INDEX[ch] for ch in seq]

def load_model(model_path: str) -> BetterAminoAcidNet:
    model = BetterAminoAcidNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def read_fasta_sequence(fasta_path: str) -> str:
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
    # Skip header, join sequence lines
    sequence = ''.join(line.strip() for line in lines[1:])
    return sequence

def generate_9mers(sequence: str) -> list[str]:
    return [sequence[i:i+9] for i in range(len(sequence) - 8)]

def run_inference(model: BetterAminoAcidNet, sequences: list[str]) -> np.ndarray:
    encoded = [encode_sequence(seq) for seq in sequences]
    tensor = torch.tensor(encoded, dtype=torch.long)
    with torch.no_grad():
        outputs = model(tensor)
    return outputs.numpy()

def plot_predictions(predictions: np.ndarray, alleles: list[str], normalized: bool = False):
    positions = np.arange(1, len(predictions) + 1)  # 1-based positions
    
    plt.figure(figsize=(15, 8))
    for i, allele in enumerate(alleles):
        plt.plot(positions, predictions[:, i], label=allele, alpha=0.7)
    
    plt.xlabel('Position in Spike Protein (9-mer start position)')
    ylabel = 'Z-Score' if normalized else 'Prediction Probability'
    title = 'HLA Allele Binding Z-Scores Across SARS-CoV-2 Spike Protein' if normalized else 'HLA Allele Binding Predictions Across SARS-CoV-2 Spike Protein'
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = 'sars_predictions_zscore_plot.png' if normalized else 'sars_predictions_plot.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{filename}'")
    # plt.show()  # Commented out for headless environment

def plot_zscore_distributions(z_scores: np.ndarray, alleles: list[str]):
    """Create histograms showing z-score distributions for each HLA allele."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()  # Flatten to 1D array for easier iteration
    
    for i, allele in enumerate(alleles):
        ax = axes[i]
        # Create histogram of z-scores for this allele
        ax.hist(z_scores[:, i], bins=50, alpha=0.7, color=f'C{i}', edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Z-Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{allele} Z-Score Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at z=0 for reference
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    plt.savefig('sars_zscore_distributions.png', dpi=300, bbox_inches='tight')
    print("Z-score distribution plots saved as 'sars_zscore_distributions.png'")
    # plt.show()  # Commented out for headless environment

def main():
    # Load model
    model_path = 'artifacts/model_BetterAminoAcidNet_seed3510100532.pt'
    model = load_model(model_path)

    # Load sequence
    fasta_path = 'ex1 data/P0DTC2.fasta.txt'
    sequence = read_fasta_sequence(fasta_path)
    print(f"Loaded sequence of length {len(sequence)}")

    # Generate 9-mers
    peptides = generate_9mers(sequence)
    print(f"Generated {len(peptides)} 9-mer peptides")

    # Run inference
    predictions = run_inference(model, peptides)
    print(f"Predictions shape: {predictions.shape}")

    # Compute z-scores for each allele
    z_scores = np.zeros_like(predictions)
    for i in range(predictions.shape[1]):
        mean_val = np.mean(predictions[:, i])
        std_val = np.std(predictions[:, i])
        z_scores[:, i] = (predictions[:, i] - mean_val) / std_val

    # Plot z-score distributions for each allele
    plot_zscore_distributions(z_scores, ALLELES)

    # Plot original predictions
    plot_predictions(predictions, ALLELES, normalized=False)

    # Plot z-score normalized predictions
    plot_predictions(z_scores, ALLELES, normalized=True)

    # Find top 3 highest z-scores across all alleles
    max_z_scores = np.max(z_scores, axis=1)
    top_z_indices = np.argsort(max_z_scores)[-3:][::-1]  # Top 3 in descending order

    print("\nTop 3 highest z-scores:")
    for i, idx in enumerate(top_z_indices):
        peptide = peptides[idx]
        z_score = max_z_scores[idx]
        allele_idx = np.argmax(z_scores[idx])
        allele = ALLELES[allele_idx]
        position = idx + 1  # 1-based position
        print(f"{i+1}. Position {position}: {peptide} - Z-score {z_score:.4f} for allele {allele}")

    # Also show original top 3 for comparison
    max_probs = np.max(predictions, axis=1)
    top_indices = np.argsort(max_probs)[-3:][::-1]  # Top 3 in descending order

    print("\nTop 3 most detectable peptides (original probabilities):")
    for i, idx in enumerate(top_indices):
        peptide = peptides[idx]
        prob = max_probs[idx]
        allele_idx = np.argmax(predictions[idx])
        allele = ALLELES[allele_idx]
        position = idx + 1  # 1-based position
        print(f"{i+1}. Position {position}: {peptide} - Probability {prob:.4f} for allele {allele}")


if __name__ == "__main__":
    main()