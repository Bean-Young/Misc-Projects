import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

class HybridTranslationModel(nn.Module):
    def __init__(self, model_name='/home/yyz/NLP-Class/Project2/bert', d_model=512, 
                 nhead=8, num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048, cnn_out_channels=256, kernel_size=3):
        super(HybridTranslationModel, self).__init__()
        
        # 初始化参数
        self.d_model = d_model
        
        # BERT Encoder
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        
        # CNN Layer: 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=768, 
            out_channels=cnn_out_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size//2
        )
        
        # 源序列特征变换
        self.src_linear = nn.Linear(cnn_out_channels, d_model)  # 将CNN输出变换到d_model
        
        # 目标序列特征变换
        self.tgt_linear = nn.Linear(768, d_model)  # 将BERT输出变换到d_model
        
        # Transformer Decoder
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
        
        # 位置编码 - 注册为缓冲区，确保与模型在同一设备上
        self.register_buffer('positional_encoding', self._create_positional_encoding(d_model, max_len=512))

    def _create_positional_encoding(self, d_model, max_len=512):
        """创建位置编码"""
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe  # 不要移动设备，模型会处理

    def forward(self, src, tgt):
        # 确保位置编码与输入在同一设备上
        device = src.device
        self.positional_encoding = self.positional_encoding.to(device)
        
        # BERT编码
        src_embedding = self.encoder(input_ids=src).last_hidden_state  # [batch_size, seq_len, 768]
        tgt_embedding = self.encoder(input_ids=tgt).last_hidden_state  # [batch_size, seq_len, 768]
        
        # 源序列特征提取
        src_embedding = src_embedding.permute(0, 2, 1)  # [batch_size, 768, seq_len]
        src_embedding = self.conv1d(src_embedding)      # [batch_size, cnn_out_channels, seq_len]
        src_embedding = src_embedding.permute(0, 2, 1)  # [batch_size, seq_len, cnn_out_channels]
        src_embedding = self.src_linear(src_embedding)  # [batch_size, seq_len, d_model]
        
        # 目标序列特征变换
        tgt_embedding = self.tgt_linear(tgt_embedding)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        src_positions = self.positional_encoding[:src_embedding.size(1)].squeeze(1)
        tgt_positions = self.positional_encoding[:tgt_embedding.size(1)].squeeze(1)
        
        src_embedding = src_embedding + src_positions.unsqueeze(0)
        tgt_embedding = tgt_embedding + tgt_positions.unsqueeze(0)
        
        # 调整维度顺序: [seq_len, batch_size, d_model]
        src_embedding = src_embedding.permute(1, 0, 2)  
        tgt_embedding = tgt_embedding.permute(1, 0, 2)
        
        # 生成注意力掩码
        src_mask = self._create_src_mask(src)
        tgt_pad_mask, tgt_mask = self._create_tgt_mask(tgt)
        
        # 确保掩码与数据在同一设备上
        src_mask = src_mask.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        
        # Transformer解码
        output = self.transformer(
            src=src_embedding, 
            tgt=tgt_embedding,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask
        )
        
        # 输出层
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        output = self.fc_out(output)
        
        return output

    def _create_src_mask(self, src):
        """创建源序列的填充掩码"""
        return (src == self.src_pad_idx)
    
    def _create_tgt_mask(self, tgt):
        """创建目标序列的填充掩码和未来掩码"""
        # 填充掩码
        tgt_pad_mask = (tgt == self.tgt_pad_idx)
        
        # 未来掩码（防止解码器看到未来信息）
        seq_len = tgt.size(1)
        future_mask = torch.triu(
            torch.ones(seq_len, seq_len), 
            diagonal=1
        ).bool()
        
        return tgt_pad_mask, future_mask