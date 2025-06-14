import torch
import torch.nn as nn

class TextRNN(nn.Module):
    def __init__(self, config, vocab_size):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'])
        self.lstm = nn.LSTM(config['embedding_dim'], config['hidden_dim'], batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(2 * config['hidden_dim'], config['num_classes'])

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        out = self.dropout(output[:, -1, :])
        return self.fc(out)