import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, config, vocab_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'])
        nhead = 8
        if config['embedding_dim'] % nhead != 0:
            raise ValueError(f"embedding_dim ({config['embedding_dim']}) must be divisible by nhead ({nhead})")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['embedding_dim'],
            nhead=nhead,
            dropout=config.get('dropout', 0.1),
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(config['embedding_dim'], config['num_classes'])

    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)  # [L, B, D]
        x = self.transformer(x)
        out = x[0]  # 取首 token 表示句子
        return self.fc(out)
