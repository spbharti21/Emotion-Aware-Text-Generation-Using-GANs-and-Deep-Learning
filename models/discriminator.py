# models/discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, emotion_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm      = nn.LSTM(embedding_dim + emotion_dim, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, 1)
        self.act       = nn.Sigmoid()

    def forward(self, x, emo):
        emb = self.embedding(x)
        e   = emo.unsqueeze(1).repeat(1, emb.size(1), 1)
        inp = torch.cat([emb, e], dim=2)
        out, _ = self.lstm(inp)
        return self.act(self.fc(out[:, -1, :]))
