from torch import nn


class AminoAcidNet(nn.Module):
    def __init__(self, vocab_size=20, embed_dim=8, seq_len=9):
        super(AminoAcidNet, self).__init__()

        # The Lookup Table: 20 rows, 8 columns
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        # After embedding, a 9-mer becomes a 9x8 matrix.
        # We flatten it to a vector of size 72 for the next layer.
        self.flatten = nn.Flatten()

        # Standard hidden layers
        self.fc1 = nn.Linear(seq_len * embed_dim, seq_len * embed_dim)  # 72 to 72)
        self.fc2 = nn.Linear(seq_len * embed_dim, seq_len * embed_dim)  # 72 to 72)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.output = nn.Linear(seq_len * embed_dim, 1)  # Assuming a regression or binary task

    def forward(self, x):
        # x input shape: (batch_size, 9)
        x = self.embedding(x)  # output shape: (batch_size, 9, 8)
        x = self.flatten(x)  # output shape: (batch_size, 72)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        x = self.output(x)
        x = self.sigmoid(x)  # For binary classification; remove if regression
        return x


# Initialize model
model = AminoAcidNet(vocab_size=20, embed_dim=8, seq_len=9)