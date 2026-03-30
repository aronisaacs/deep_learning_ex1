import torch
import torch.nn.functional as F
from torch import nn


class BetterAminoAcidNet(nn.Module):
    """Two-hidden-layer one-hot classifier with dropout."""

    def __init__(self, vocab_size=20, seq_len=9, output_dim=6, dropout_p=0.3):
        super().__init__()

        self.vocab_size = vocab_size
        self.flatten = nn.Flatten()

        input_dim = seq_len * vocab_size
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x):
        x = F.one_hot(x.to(torch.long), num_classes=self.vocab_size).float()
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        #x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        #x = self.dropout2(x)

        # Return raw logits; use BCEWithLogitsLoss for training.
        x = self.output(x)
        return self.softmax(x)
