import torch
import torch.nn as nn

class HAN(nn.Module):
    def __init__(self, config, vocab_size):
        super(HAN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'])
        self.word_rnn = nn.GRU(config['embedding_dim'], config['hidden_dim'], bidirectional=True, batch_first=True)
        self.word_att = nn.Linear(2*config['hidden_dim'], 2*config['hidden_dim'])
        self.word_context = nn.Parameter(torch.Tensor(2*config['hidden_dim'], 1))
        self.sent_rnn = nn.GRU(2*config['hidden_dim'], config['hidden_dim'], bidirectional=True, batch_first=True)
        self.sent_att = nn.Linear(2*config['hidden_dim'], 2*config['hidden_dim'])
        self.sent_context = nn.Parameter(torch.Tensor(2*config['hidden_dim'], 1))
        self.fc = nn.Linear(2*config['hidden_dim'], config['num_classes'])
        nn.init.xavier_uniform_(self.word_context)
        nn.init.xavier_uniform_(self.sent_context)

    def attention(self, rnn_output, att_linear, context_vector):
        u = torch.tanh(att_linear(rnn_output))
        scores = torch.matmul(u, context_vector).squeeze(-1)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(rnn_output * alpha, dim=1)

    def forward(self, x):
        # 假设 x 是 [B, L] 文本，无句子维度，先视作一个句子
        emb = self.embedding(x)
        word_enc, _ = self.word_rnn(emb)
        sent_vec = self.attention(word_enc, self.word_att, self.word_context).unsqueeze(1)
        sent_enc, _ = self.sent_rnn(sent_vec)
        doc_vec = self.attention(sent_enc, self.sent_att, self.sent_context)
        return self.fc(doc_vec)
