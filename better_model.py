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
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.dropout3 = nn.Dropout(dropout_p)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.one_hot(x.to(torch.long), num_classes=self.vocab_size).float()
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)

        x = self.output(x)
        return self.sigmoid(x)
