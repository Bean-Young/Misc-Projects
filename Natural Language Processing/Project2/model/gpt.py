import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2TranslationModel(nn.Module):
    def __init__(self, model_name='/home/yyz/NLP-Class/Project2/gpt2', d_model=512):
        super(GPT2TranslationModel, self).__init__()
        
        # 使用本地路径加载GPT-2模型
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # 确保设置padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.encoder_decoder = GPT2LMHeadModel.from_pretrained(model_name)
        self.encoder_decoder.config.pad_token_id = self.tokenizer.pad_token_id
        
        # 不再需要额外的线性层，因为GPT2LMHeadModel已经有输出层
        # self.fc_out = nn.Linear(d_model, len(self.tokenizer))
        self.src_pad_idx = self.tokenizer.pad_token_id
        self.tgt_pad_idx = self.tokenizer.pad_token_id

    def forward(self, src, tgt):
        # 创建完整的输入序列: [源序列, 目标序列(去掉最后一个token)]
        input_ids = torch.cat((src, tgt[:, :-1]), dim=1)
        
        # 创建标签: [-100...源序列部分..., 目标序列(去掉第一个token)]
        labels = torch.cat((
            torch.full_like(src, -100),  # 忽略源序列部分的损失计算
            tgt[:, 1:]                  # 目标序列从第二个token开始
        ), dim=1)
        
        # 传入输入和标签
        outputs = self.encoder_decoder(
            input_ids=input_ids,
            labels=labels,
            attention_mask=(input_ids != self.src_pad_idx).float()
        )
        
        return outputs.loss  # 直接返回损失值