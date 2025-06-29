import math
import torch
from torch import nn
from d2l import torch as d2l
import os

# 数据和结果路径
DATA_PATH = "/home/yyz/NNDL-Class/Project5/Data"
RESULT_PATH = "/home/yyz/NNDL-Class/Project5/Result"
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

# 掩蔽softmax操作
def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# 加性注意力
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 缩放点积注意力
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 小例子：运行和可视化
def run_attention_demo():
    # 构造输入
    queries_add = torch.normal(0, 1, (2, 1, 20))
    queries_dot = torch.normal(0, 1, (2, 1, 2))  # 与 keys 维度一致

    # 改动的 keys（练习题1要求修改）
    keys = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    keys = keys.unsqueeze(0).repeat(2, 1, 1)

    # 更改key的数值
    keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0],
                        [2.0, 2.0], [3.0, 1.0], [1.0, 3.0], [4.0, 0.0], [0.0, 4.0]]] * 2)


    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])

    # 加性注意力
    add_attention = AdditiveAttention(2, 20, 8, 0.1)
    add_attention.eval()
    out_add = add_attention(queries_add, keys, values, valid_lens)

    # 缩放点积注意力
    dot_attention = DotProductAttention(0.1)
    dot_attention.eval()
    out_dot = dot_attention(queries_dot, keys, values, valid_lens)

    print("加性注意力输出:\n", out_add)
    print("缩放点积注意力输出:\n", out_dot)

    # 可视化
    d2l.show_heatmaps(add_attention.attention_weights.reshape((1, 1, 2, 10)),
                      xlabel='Keys', ylabel='Queries')
    d2l.plt.savefig(f"{RESULT_PATH}/additive_attention_heatmap_1.png")

    d2l.show_heatmaps(dot_attention.attention_weights.reshape((1, 1, 2, 10)),
                      xlabel='Keys', ylabel='Queries')
    d2l.plt.savefig(f"{RESULT_PATH}/dot_product_attention_heatmap_1.png")

if __name__ == "__main__":
    run_attention_demo()
