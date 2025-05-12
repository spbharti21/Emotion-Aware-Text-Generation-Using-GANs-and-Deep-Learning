# models/generator.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, emotion_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm      = nn.LSTM(embedding_dim + emotion_dim, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, emo):
        emb = self.embedding(x)                                # (B, T, E)
        e   = emo.unsqueeze(1).repeat(1, emb.size(1), 1)      # (B, T, emo_dim)
        inp = torch.cat([emb, e], dim=2)                      # (B, T, E+emo_dim)
        out, _ = self.lstm(inp)                               # (B, T, H)
        return self.fc(out)                                   # (B, T, vocab_size)
