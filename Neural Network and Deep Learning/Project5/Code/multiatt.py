import math
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os
# 点积注意力类（可以理解为每个头的注意力机制）
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 多头注意力类
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # 将查询、键、值变换成多个头的形状
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # 如果valid_lens不为空，需要重复valid_lens，使其与头数匹配
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # 通过注意力机制计算结果
        output = self.attention(queries, keys, values, valid_lens)
        
        # 拼接并通过线性层进行变换
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

# 转置查询、键、值的形状
def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

# 转置输出的形状
def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

# Masked softmax操作
def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

def sequence_mask(X, valid_lens, value):
    # 遮掩多余的部分
    for i, length in enumerate(valid_lens):
        X[i, length:] = value
    return X



# 创建保存路径
save_path = '/home/yyz/NNDL-Class/Project5/Result/attention_weights/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def save_attention_weights(attention_weights, num_heads, num_queries, num_kvpairs):
    # 画出每个头的注意力权重并保存
    for i in range(num_heads):
        plt.figure(figsize=(8, 6))
        plt.imshow(attention_weights[i].detach().cpu().numpy(), cmap='Blues', aspect='auto')
        plt.colorbar()
        plt.title(f"Attention Head {i+1}")
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        # 保存图像到文件
        file_path = os.path.join(save_path, f"attention_head_{i+1}.png")
        plt.savefig(file_path)
        plt.close()  # 关闭当前图像，以释放内存

# 修改主函数，加入提取并保存多个头的注意力权重
def main():
    # 设置参数
    batch_size = 2
    num_queries = 4
    num_kvpairs = 6
    num_hiddens = 10
    num_heads = 5
    valid_lens = torch.tensor([3, 2])  # 有效长度
    queries = torch.ones((batch_size, num_queries, num_hiddens))
    keys = torch.ones((batch_size, num_kvpairs, num_hiddens))
    values = torch.ones((batch_size, num_kvpairs, num_hiddens))

    # 创建多头注意力模型
    attention = MultiHeadAttention(key_size=num_hiddens, query_size=num_hiddens, value_size=num_hiddens,
                                   num_hiddens=num_hiddens, num_heads=num_heads, dropout=0.5)
    
    # 执行前向传播
    output = attention(queries, keys, values, valid_lens)

    # 打印结果
    print("Attention output shape:", output.shape)  # 输出的形状应该是 (batch_size, num_queries, num_hiddens)
    print("Output:", output)

    # 提取并保存每个头的注意力权重
    attention_weights = attention.attention.attention_weights  # 获取注意力权重
    print("Attention weights:", attention_weights.shape)

    # 保存每个头的注意力权重
    save_attention_weights(attention_weights, num_heads, num_queries, num_kvpairs)

if __name__ == "__main__":
    main()
