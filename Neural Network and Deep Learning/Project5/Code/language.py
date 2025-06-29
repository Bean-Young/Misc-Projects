import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(DecoderOnlyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 创建解码器层（注意这里使用的是TransformerDecoderLayer）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 堆叠多个解码器层
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        
    def forward(self, x):
        # 创建因果掩码（确保模型只能看到当前位置之前的token）
        mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        # 嵌入和位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # 通过Transformer解码器
        # 注意：这里将memory参数设为None，因为我们不使用编码器的输出
        output = self.transformer_decoder(x, memory=None, tgt_mask=mask)
        
        # 预测下一个token
        return self.output_layer(output)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# 使用示例
def train_language_model():
    vocab_size = 10000
    model = DecoderOnlyTransformer(vocab_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for batch in data_loader:
            inputs = batch[:, :-1]  # 除了最后一个token的所有tokens
            targets = batch[:, 1:]  # 从第二个token开始的所有tokens
            
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
