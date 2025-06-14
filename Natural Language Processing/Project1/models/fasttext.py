import torch
import torch.nn as nn

class FastText(nn.Module):
    def __init__(self, config, vocab_size):
        super(FastText, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'])
        self.fc = nn.Linear(config['embedding_dim'], config['num_classes'])

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)