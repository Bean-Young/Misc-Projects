import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, config, vocab_size):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'])
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (k, config['embedding_dim'])) for k in [3, 4, 5]
        ])
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(100 * 3, config.get("num_classes", 2))  # 自动适配类别数，默认二分类

    def forward(self, x):
        x = self.embedding(x)  # [B, L, D]
        x = x.unsqueeze(1)     # [B, 1, L, D]
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # [B, F, L-k+1]
        x = [torch.max(pool, dim=2)[0] for pool in x]  # [B, F]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
