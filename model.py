import torch
import torch.nn.functional as F
from torch import nn


class AminoAcidNet(nn.Module):
    def __init__(self, vocab_size=20, seq_len=9, output_dim=7):
        super().__init__()

        self.vocab_size = vocab_size
        self.flatten = nn.Flatten()

        input_dim = seq_len * vocab_size
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = F.one_hot(x.to(torch.long), num_classes=self.vocab_size).float()
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x
