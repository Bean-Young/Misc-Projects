import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attention_weights = self.masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(attention_weights), values)

    def masked_softmax(self, X, valid_lens):
        """在最后一个维度进行softmax，并处理mask"""
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = self.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

    def sequence_mask(self, X, valid_lens, value=0):
        """生成有效长度的mask"""
        max_len = X.shape[1]
        mask = torch.arange(max_len, device=X.device)[None, :] < valid_lens[:, None]
        X[~mask] = value
        return X


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=False)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=False)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

    def transpose_qkv(self, X):
        """为多头注意力计算变换"""
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """逆转transpose_qkv的操作"""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

# 测试位置编码
num_hiddens, num_steps = 32, 60
pos_encoding = PositionalEncoding(num_hiddens, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, num_hiddens)))
P = pos_encoding.P[:, :X.shape[1], :]

# 可视化位置编码
import matplotlib.pyplot as plt
plt.pcolormesh(P[0].cpu().detach().numpy(), cmap='Blues')
plt.xlabel('Encoding Dimension')
plt.ylabel('Position')
plt.show()

# 测试多头注意力
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
attention.eval()

# 输入数据
batch_size, num_queries = 2, 4
num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))

# 计算注意力
attention(X, Y, Y, valid_lens).shape  # 输出形状应为 (batch_size, num_queries, num_hiddens)
# 测试位置编码
num_hiddens, num_steps = 32, 60
pos_encoding = PositionalEncoding(num_hiddens, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, num_hiddens)))
P = pos_encoding.P[:, :X.shape[1], :]

# 保存位置编码的图像
import matplotlib.pyplot as plt
save_path = '/home/yyz/NNDL-Class/Project5/Result/positional_encoding.png'
plt.pcolormesh(P[0].cpu().detach().numpy(), cmap='Blues')
plt.xlabel('Encoding Dimension')
plt.ylabel('Position')
plt.savefig(save_path)
plt.close() 

print(f"The position encoded image has been saved to {save_path}")

# 测试多头注意力
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
attention.eval()

# 输入数据
batch_size, num_queries = 2, 4
num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))

# 计算注意力
output = attention(X, Y, Y, valid_lens)

# 打印输出的形状和输出的内容
print("Output shape:", output.shape)  # 输出形状应为 (batch_size, num_queries, num_hiddens)
print("Output content:", output)
