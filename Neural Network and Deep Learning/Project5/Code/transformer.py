import torch
from torch import nn
import math
import pandas as pd
import matplotlib.pyplot as plt
from d2l import torch as d2l
from bahdanau import load_data_from_file
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class EncoderDecoder(nn.Module):
   def __init__(self, encoder, decoder):
       super(EncoderDecoder, self).__init__()
       self.encoder = encoder
       self.decoder = decoder

   def forward(self, X, Y, enc_valid_lens):
       encoder_output = self.encoder(X, enc_valid_lens)
       state = self.decoder.init_state(encoder_output, enc_valid_lens)
       output, state = self.decoder(Y, state)
       return output, state

class PositionWiseFFN(nn.Module):
   """基于位置的前馈网络"""
   def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
       super(PositionWiseFFN, self).__init__(**kwargs)
       self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
       self.relu = nn.ReLU()
       self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

   def forward(self, X):
       return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
   """残差连接后进行层规范化"""
   def __init__(self, normalized_shape, dropout, **kwargs):
       super(AddNorm, self).__init__(**kwargs)
       self.dropout = nn.Dropout(dropout)
       self.ln = nn.LayerNorm(normalized_shape)

   def forward(self, X, Y):
       return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
   """Transformer编码器块"""
   def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
       super(EncoderBlock, self).__init__(**kwargs)
       self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
       self.addnorm1 = AddNorm(norm_shape, dropout)
       self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
       self.addnorm2 = AddNorm(norm_shape, dropout)

   def forward(self, X, valid_lens):
       Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
       return self.addnorm2(Y, self.ffn(Y))

def transpose_qkv(X, num_heads):
   """为了多头注意力头的并行计算而变换形状"""
   X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
   X = X.permute(0, 2, 1, 3)
   return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
   """逆转transpose_qkv函数的操作"""
   X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
   X = X.permute(0, 2, 1, 3)
   return X.reshape(X.shape[0], X.shape[1], -1)

class DotProductAttention(nn.Module):
   """缩放点积注意力"""
   def __init__(self, dropout=0.0, **kwargs):
       super(DotProductAttention, self).__init__(**kwargs)
       self.dropout = nn.Dropout(dropout)
       self.attention_weights = None  # 初始化属性

   def forward(self, queries, keys, values, valid_lens=None):
       """查询、键和值，valid_lens用于掩蔽"""
       d = queries.shape[-1]  # 查询的最后一维是d
       # 计算缩放点积
       scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)

       # 对scores进行softmax，得到注意力权重
       attention_weights = masked_softmax(scores, valid_lens)

       # 保存attention_weights到成员变量，方便外部访问
       self.attention_weights = attention_weights

       # 计算加权和
       return torch.bmm(self.dropout(attention_weights), values)

def masked_softmax(X, valid_lens):
   """计算softmax并对无效位置进行掩蔽"""
   if valid_lens is None:
       return nn.functional.softmax(X, dim=-1)
   else:
       shape = X.shape
       if valid_lens.dim() == 1:
           valid_lens = torch.repeat_interleave(valid_lens, shape[1])
       else:
           valid_lens = valid_lens.reshape(-1)
       
       # 对超出有效长度的位置赋予非常大的负值，使其softmax输出为0
       X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
       return nn.functional.softmax(X.reshape(shape), dim=-1)

def sequence_mask(X, valid_lens, value=0):
   """给定有效长度，掩蔽无效部分"""
   max_len = X.shape[1]
   mask = torch.arange(max_len, device=X.device).expand(len(valid_lens), max_len) < valid_lens.unsqueeze(1)
   X[~mask] = value
   return X

class MultiHeadAttention(nn.Module):
   """多头注意力"""
   def __init__(self, key_size, query_size, value_size, num_hiddens,
                num_heads, dropout, bias=False, **kwargs):
       super(MultiHeadAttention, self).__init__(**kwargs)
       self.num_heads = num_heads
       self.attention = DotProductAttention(dropout)
       
       # 定义查询、键和值的线性映射
       self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
       self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
       self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
       
       # 输出的线性变换
       self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

   def forward(self, queries, keys, values, valid_lens):
       # 将输入的查询、键和值进行线性变换
       queries = transpose_qkv(self.W_q(queries), self.num_heads)
       keys = transpose_qkv(self.W_k(keys), self.num_heads)
       values = transpose_qkv(self.W_v(values), self.num_heads)

       if valid_lens is not None:
           # 将 valid_lens 进行复制以适应 num_heads 的大小
           valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

       # 计算多头注意力
       output = self.attention(queries, keys, values, valid_lens)
       
       # 将输出拼接在一起并通过输出的线性变换
       output_concat = transpose_output(output, self.num_heads)
       return self.W_o(output_concat)

class DecoderBlock(nn.Module):
   """解码器中第i个块"""
   def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
       super(DecoderBlock, self).__init__(**kwargs)
       self.i = i
       self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
       self.addnorm1 = AddNorm(norm_shape, dropout)
       self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
       self.addnorm2 = AddNorm(norm_shape, dropout)
       self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
       self.addnorm3 = AddNorm(norm_shape, dropout)

   def forward(self, X, state):
       enc_outputs, enc_valid_lens = state[0], state[1]
       if state[2][self.i] is None:
           key_values = X
       else:
           key_values = torch.cat((state[2][self.i], X), axis=1)
       state[2][self.i] = key_values

       if self.training:
           batch_size, num_steps, _ = X.shape
           dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
       else:
           dec_valid_lens = None

       X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
       Y = self.addnorm1(X, X2)
       Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
       Z = self.addnorm2(Y, Y2)
       return self.addnorm3(Z, self.ffn(Z)), state

class TransformerEncoder(d2l.Encoder):
   """Transformer编码器"""
   def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, use_bias=False, **kwargs):
       super(TransformerEncoder, self).__init__(**kwargs)
       self.num_hiddens = num_hiddens
       self.embedding = nn.Embedding(vocab_size, num_hiddens)
       self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
       self.blks = nn.Sequential()
       for i in range(num_layers):
           self.blks.add_module("block" + str(i), EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias))

   def forward(self, X, valid_lens, *args):
       X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
       self.attention_weights = [None] * len(self.blks)
       for i, blk in enumerate(self.blks):
           X = blk(X, valid_lens)
           self.attention_weights[i] = blk.attention.attention.attention_weights
       return X

class TransformerDecoder(d2l.AttentionDecoder):
   """Transformer解码器"""
   def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
       super(TransformerDecoder, self).__init__(**kwargs)
       self.num_hiddens = num_hiddens
       self.num_layers = num_layers
       self.embedding = nn.Embedding(vocab_size, num_hiddens)
       self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
       self.blks = nn.Sequential()
       for i in range(num_layers):
           self.blks.add_module("block" + str(i), DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, i))
       self.dense = nn.Linear(num_hiddens, vocab_size)

   def init_state(self, enc_outputs, enc_valid_lens, *args):
       return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

   def forward(self, X, state):
       X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
       self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
       for i, blk in enumerate(self.blks):
           X, state = blk(X, state)
           self._attention_weights[0][i] = blk.attention1.attention.attention_weights
           self._attention_weights[1][i] = blk.attention2.attention.attention_weights
       return self.dense(X), state

   @property
   def attention_weights(self):
       return self._attention_weights

class PositionalEncoding(nn.Module):
   """位置编码"""
   def __init__(self, num_hiddens, dropout, max_len=1000):
       super(PositionalEncoding, self).__init__()
       self.dropout = nn.Dropout(dropout)
       # 创建一个足够长的P
       self.P = torch.zeros((1, max_len, num_hiddens))
       X = torch.arange(max_len, dtype=torch.float32).reshape(
           -1, 1) / torch.pow(10000, torch.arange(
               0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
       self.P[:, :, 0::2] = torch.sin(X)
       self.P[:, :, 1::2] = torch.cos(X)

   def forward(self, X):
       X = X + self.P[:, :X.shape[1], :].to(X.device)
       return self.dropout(X)

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
   def forward(self, pred, label, valid_len):
       weights = torch.ones_like(label)
       mask = torch.arange(label.shape[1], device=label.device)[None, :] < valid_len[:, None]
       weights = weights * mask
       self.reduction = 'none'
       unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
           pred.permute(0, 2, 1), label)
       weighted_loss = (unweighted_loss * weights).sum(dim=1) / valid_len
       return weighted_loss

from tqdm import tqdm

def train_seq2seq_custom(net, data_iter, lr, num_epochs, tgt_vocab, device):
   def xavier_init_weights(m):
       if isinstance(m, nn.Linear):
           nn.init.xavier_uniform_(m.weight)
       elif isinstance(m, nn.GRU):
           for param in m._flat_weights_names:
               if "weight" in param:
                   nn.init.xavier_uniform_(m._parameters[param])
   
   net.apply(xavier_init_weights)
   net.to(device)
   optimizer = torch.optim.Adam(net.parameters(), lr=lr)
   loss = MaskedSoftmaxCELoss()
   net.train()
   
   # 初始化存储训练损失的列表
   train_losses = []
   
   for epoch in range(num_epochs):
       timer = d2l.Timer()
       metric = d2l.Accumulator(2)  # 累加训练损失和词元数
       epoch_loss = 0
       num_batches = 0

       for X, Y in tqdm(data_iter, desc=f"Epoch {epoch + 1}/{num_epochs}"):
           X_valid_len = (X != tgt_vocab['<pad>']).sum(dim=1)
           Y_valid_len = (Y != tgt_vocab['<pad>']).sum(dim=1)

           bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
           dec_input = torch.cat([bos, Y[:, :-1]], dim=1)

           Y_hat, _ = net(X, dec_input, X_valid_len)
           l = loss(Y_hat, Y, Y_valid_len)

           optimizer.zero_grad()
           l.sum().backward()
           d2l.grad_clipping(net, 1)
           optimizer.step()

           num_tokens = Y_valid_len.sum()
           metric.add(l.sum(), num_tokens)
           
           epoch_loss += l.sum().item()
           num_batches += 1

       # 计算并存储每个epoch的平均损失
       avg_epoch_loss = epoch_loss / num_batches
       train_losses.append(avg_epoch_loss)
       
       print(f'epoch {epoch + 1}, loss {metric[0] / metric[1]:.3f}, '
             f'{metric[1] / timer.stop():.1f} tokens/sec')
   
   return train_losses

def plot_learning_curves(train_losses, val_losses=None, train_accs=None, val_accs=None):
    epochs = range(1, len(train_losses) + 1)

    # 检查是否有准确率数据
    has_acc_data = (train_accs is not None or val_accs is not None)
    
    # 根据我们要绘制的内容调整图形大小
    if has_acc_data:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax = [ax]  # 使ax可迭代，即使它是单个图

    # 绘制 Loss 曲线
    ax[0].plot(epochs, train_losses, label='Train Loss')
    if val_losses:
        ax[0].plot(epochs, val_losses, label='Validation Loss')
    ax[0].set_title('Loss Curve')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # 绘制 Accuracy 曲线（可选）
    if has_acc_data:
        if train_accs:
            ax[1].plot(epochs, train_accs, label='Train Acc')
        if val_accs:
            ax[1].plot(epochs, val_accs, label='Validation Acc')
        ax[1].set_title('Accuracy Curve')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()

    plt.tight_layout()
    return fig

def plot_attention_heatmap(attention_weights, src_sentence, tgt_sentence):
    """绘制注意力权重热图"""
    # 获取注意力权重
    attention = attention_weights.cpu().detach().numpy()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热图
    im = ax.imshow(attention, cmap='viridis')
    
    # 设置坐标轴标签
    ax.set_xticks(range(len(src_sentence)))
    ax.set_yticks(range(len(tgt_sentence)))
    
    # 设置标签内容
    ax.set_xticklabels(src_sentence, rotation=45)
    ax.set_yticklabels(tgt_sentence)
    
    # 添加颜色条
    plt.colorbar(im)
    
    # 设置标题
    ax.set_title("Attention Weights")
    
    plt.tight_layout()
    return fig



def main():
    # 超参数设置
    num_hiddens, num_layers, dropout = 32, 2, 0.1
    batch_size, num_steps = 1024, 10
    lr, num_epochs = 0.005, 50
    device = d2l.try_gpu()

    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]

    # 文件路径
    data_path = '/home/yyz/NNDL-Class/Project5/Data/fra-eng'
    file_path = os.path.join(data_path, 'fra.txt')

    # 加载数据
    src_vocab, tgt_vocab, train_iter = load_data_from_file(file_path, batch_size, num_steps)

    # 初始化 Transformer 模型
    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout
    )
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout
    )
    net = EncoderDecoder(encoder, decoder)

    # 训练模型并获取损失列表
    train_losses = train_seq2seq_custom(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 保存模型
    torch.save(net.state_dict(), '/home/yyz/NNDL-Class/Project5/Result/transformer_model.pth')

    # 绘制并保存损失曲线
    loss_fig = plot_learning_curves(train_losses)
    loss_fig.savefig('/home/yyz/NNDL-Class/Project5/Result/training_loss_curve_transformer.png')
    
    # 从训练集中获取一个样本
    for X, Y in train_iter:
        sample_X = X[0:1].to(device)  # 只取第一个样本
        sample_Y = Y[0:1].to(device)
        X_valid_len = (sample_X != tgt_vocab['<pad>']).sum(dim=1)
        
        # 准备解码器输入
        bos = torch.tensor([tgt_vocab['<bos>']], device=device).reshape(1, 1)
        dec_input = torch.cat([bos, sample_Y[:, :-1]], dim=1)
        
        # 前向传播获取注意力权重
        with torch.no_grad():
            encoder_output = encoder(sample_X, X_valid_len)
            state = decoder.init_state(encoder_output, X_valid_len)
            _, state = decoder(dec_input, state)
            
            # 获取注意力权重
            attention_weights = decoder.attention_weights[1][0]  # 编码器-解码器注意力
            
            # 获取句子文本（将索引转换为词）
            # 手动查找每个索引对应的词
            src_tokens = []
            for idx in sample_X[0]:
                idx_item = idx.item()
                if idx_item != src_vocab['<pad>']:
                    # 查找索引对应的词
                    for token, index in src_vocab.items():
                        if index == idx_item:
                            src_tokens.append(token)
                            break
            
            tgt_tokens = []
            for idx in sample_Y[0]:
                idx_item = idx.item()
                if idx_item != tgt_vocab['<pad>']:
                    # 查找索引对应的词
                    for token, index in tgt_vocab.items():
                        if index == idx_item:
                            tgt_tokens.append(token)
                            break
            
            # 绘制注意力热图
            attention_fig = plot_attention_heatmap(
                attention_weights[0], src_tokens, tgt_tokens
            )
            attention_fig.savefig('/home/yyz/NNDL-Class/Project5/Result/attention_heatmap_transformer.png')
        
        break  # 只处理一个样本


if __name__ == "__main__":
   main()
