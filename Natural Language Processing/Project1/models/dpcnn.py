import torch
import torch.nn as nn

class DPCNN(nn.Module):
    def __init__(self, config, vocab_size):
        super(DPCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'])
        self.region = nn.Conv2d(1, 250, (3, config['embedding_dim']), padding=(1, 0))
        self.conv = nn.Conv2d(250, 250, (3, 1), padding=(1, 0))
        self.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.fc = nn.Linear(250, config['num_classes'])

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # [B, 1, L, D]
        x = self.region(x)
        x = torch.relu(x)
        x = self._block(x)
        x = torch.max(x.squeeze(3), dim=2)[0]
        return self.fc(x)

    def _block(self, x):
        while x.size(2) > 2:
            px = self.pool(x)
            out = torch.relu(self.conv(px))
            out = self.conv(out)
            x = px + out
        return x