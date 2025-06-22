import torch
import torch.nn as nn

class TransformerTranslationModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, vocab_size=30522):
        super(TransformerTranslationModel, self).__init__()
        
        # 传统的 Transformer 模型架构
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.src_pad_idx = 0  # 默认的PAD token id，可以调整
        self.tgt_pad_idx = 0  # 默认的PAD token id，可以调整

    def forward(self, src, tgt):
        src_embedding = self.embedding(src)
        tgt_embedding = self.embedding(tgt)
        
        # Transformer 解码
        src_embedding = src_embedding.permute(1, 0, 2)  # [batch_size, seq_len, d_model] -> [seq_len, batch_size, d_model]
        tgt_embedding = tgt_embedding.permute(1, 0, 2)  # [batch_size, seq_len, d_model] -> [seq_len, batch_size, d_model]
        
        output = self.transformer(src_embedding, tgt_embedding)
        
        # 输出层
        output = self.fc_out(output)
        
        return output
