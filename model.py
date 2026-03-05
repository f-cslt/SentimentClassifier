import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5,
        pad_idx=0,
    ):
        super().__init__()

        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, text):
        # Mask to ignore padding tokens
        mask = (text != self.pad_idx).unsqueeze(-1).float()

        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)

        # Masked mean pooling: average only over real tokens
        lstm_out = lstm_out * mask
        pooled = lstm_out.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        dropped = self.dropout(pooled)
        return self.fc(dropped).squeeze(1)
