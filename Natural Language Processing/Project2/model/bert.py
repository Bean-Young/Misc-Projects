import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertTranslationModel(nn.Module):
    def __init__(self, model_name='/home/yyz/NLP-Class/Project2/bert', d_model=768, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048):
        super(BertTranslationModel, self).__init__()
        
        # 使用BERT作为编码器
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        
        # Transformer解码器
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward
        )
        
        # 输出层
        self.fc_out = nn.Linear(d_model, len(self.tokenizer))
        self.src_pad_idx = self.tokenizer.pad_token_id
        self.tgt_pad_idx = self.tokenizer.pad_token_id

    def forward(self, src, tgt):
        # 使用BERT作为编码器
        src_embedding = self.encoder(input_ids=src).last_hidden_state
        tgt_embedding = self.encoder(input_ids=tgt).last_hidden_state
        
        # Transformer 解码器
        src_embedding = src_embedding.permute(1, 0, 2)  # [batch_size, seq_len, d_model] -> [seq_len, batch_size, d_model]
        tgt_embedding = tgt_embedding.permute(1, 0, 2)  # [batch_size, seq_len, d_model] -> [seq_len, batch_size, d_model]
        
        # 通过Transformer解码
        output = self.transformer(src_embedding, tgt_embedding)
        
        # 输出层
        output = self.fc_out(output)
        
        return output
