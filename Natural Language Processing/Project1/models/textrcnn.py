import torch
import torch.nn as nn

class TextRCNN(nn.Module):
    def __init__(self, config, vocab_size):
        super(TextRCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'])
        self.lstm = nn.LSTM(config['embedding_dim'], config['hidden_dim'], bidirectional=True, batch_first=True)
        self.conv = nn.Conv1d(in_channels=2 * config['hidden_dim'] + config['embedding_dim'],
                              out_channels=config['hidden_dim'], kernel_size=3, padding=1)
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(config['hidden_dim'], config['num_classes'])

    def forward(self, x):
        embed = self.embedding(x)
        lstm_out, _ = self.lstm(embed)
        out = torch.cat((embed, lstm_out), 2).permute(0, 2, 1)
        out = torch.relu(self.conv(out)).permute(0, 2, 1)
        out = torch.max(out, dim=1)[0]
        out = self.dropout(out)
        return self.fc(out)