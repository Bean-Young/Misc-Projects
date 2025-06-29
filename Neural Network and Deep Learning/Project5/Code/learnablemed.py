import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, num_positions, embedding_dim):
        super(LearnablePositionalEncoding, self).__init__()
        # 初始化位置编码参数，每个位置有一个可学习的向量
        self.positional_embeddings = nn.Embedding(num_positions, embedding_dim)

    def forward(self, X):
        # 获取序列长度
        seq_len = X.size(1)
        
        # 创建位置索引 (0, 1, ..., seq_len-1)
        position_indices = torch.arange(seq_len, device=X.device).unsqueeze(0).repeat(X.size(0), 1)
        
        # 获取对应位置的编码
        position_encoding = self.positional_embeddings(position_indices)
        
        # 将位置编码加到输入张量
        return X + position_encoding

# 测试可学习的位置编码
batch_size, seq_len, embedding_dim = 2, 60, 32
learnable_pos_encoding = LearnablePositionalEncoding(seq_len, embedding_dim)
learnable_pos_encoding.eval()

# 创建一个形状为 (batch_size, seq_len, embedding_dim) 的输入
X = torch.zeros((batch_size, seq_len, embedding_dim))

# 获取加入了可学习位置编码后的结果
output = learnable_pos_encoding(X)

# 打印输出的形状和内容
print("Output shape:", output.shape)  # 输出形状应为 (batch_size, num_queries, num_hiddens)
print("Output content:", output)

