import torch
import torch.nn as nn

class TextRNNAtt(nn.Module):
    def __init__(self, config, vocab_size):
        super(TextRNNAtt, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'])
        self.lstm = nn.LSTM(config['embedding_dim'], config['hidden_dim'], batch_first=True, bidirectional=True)
        self.att_weight = nn.Parameter(torch.Tensor(2 * config['hidden_dim'], 1))
        nn.init.xavier_uniform_(self.att_weight)
        self.fc = nn.Linear(2 * config['hidden_dim'], config['num_classes'])

    def attention(self, lstm_output):
        scores = torch.matmul(lstm_output, self.att_weight).squeeze(-1)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(lstm_output * alpha, dim=1)
        return context

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.attention(lstm_out)
        return self.fc(out)
