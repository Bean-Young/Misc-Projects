import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CNNTransRNN(nn.Module):
    """
    CNNTransRNN: 结合 TextCNN, TransformerEncoder, 双向 LSTM 的混合模型
    架构：Embedding -> CNN 提取局部特征 -> Transformer 编码上下文 -> Bi-LSTM 聚合序列 -> 分类层
    """
    def __init__(self, config, vocab_size):
        super(CNNTransRNN, self).__init__()
        emb_dim = config['embedding_dim']
        hid_dim = config['hidden_dim']
        nhead = config.get('nhead', 8)
        num_layers = config.get('trans_layers', 2)

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # CNN 层，多尺度卷积
        self.convs = nn.ModuleList([
            nn.Conv2d(1, hid_dim, (k, emb_dim), padding=(k//2, 0))
            for k in [3, 5, 7]
        ])
        # Transformer 编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=hid_dim * len([3, 5, 7]),
            nhead=nhead,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 双向 LSTM
        self.lstm = nn.LSTM(hid_dim * len([3, 5, 7]), hid_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config['dropout'])
        # 分类层
        self.fc = nn.Linear(2 * hid_dim, config['num_classes'])

    def forward(self, x):
        # x: [B, L]
        emb = self.embedding(x)                    # [B, L, D]
        emb = emb.unsqueeze(1)                     # [B, 1, L, D]
        # CNN 提取多尺度特征
        cnn_feats = [torch.relu(conv(emb)).squeeze(3) for conv in self.convs]  # list of [B, F, L]
        cnn_feats = [feat.permute(0, 2, 1) for feat in cnn_feats]              # list of [B, L, F]
        feat = torch.cat(cnn_feats, dim=2)          # [B, L, F_total]
        # Transformer 编码上下文
        trans_out = self.transformer(feat)          # [B, L, F_total]
        # Bi-LSTM 聚合序列
        lstm_out, _ = self.lstm(trans_out)          # [B, L, 2*H]
        # 取序列最后位置
        out = lstm_out[:, -1, :]                   # [B, 2*H]
        out = self.dropout(out)
        return self.fc(out)