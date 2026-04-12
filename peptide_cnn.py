import torch
from torch import nn

class PeptideCNN(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=32, seq_len=9, output_dim=6):
        super().__init__()

        # 1. שכבת שיכון - לומדת את הדמיון בין חומצות אמינו
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 2. קונבולוציה - לזהות מוטיבים (למשל שלשות של חומצות אמינו)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1) # לוקח את הסיגנל הכי חזק מכל פילטר

        # 3. שכבות Fully Connected רחבות יותר
        self.fc1 = nn.Linear(64, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch, seq_len]
        x = self.embedding(x.to(torch.long)) # -> [batch, seq_len, embed_dim]

        # התאמה ל-Conv1d שמצפה ל-[batch, channels, length]
        x = x.transpose(1, 2) # -> [batch, embed_dim, seq_len]

        x = self.relu(self.conv(x))
        x = self.pool(x).squeeze(-1) # -> [batch, 64]

        x = self.bn1(self.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.output(x)
        return x