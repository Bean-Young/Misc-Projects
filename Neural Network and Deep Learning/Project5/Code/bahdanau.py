import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import seaborn as sns
from collections import Counter

# 添加缺失的 masked_softmax 函数
def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        
        max_len = X.size(-1)
        mask = torch.arange(max_len, dtype=torch.float32, device=X.device)[None, :] < valid_lens[:, None]
        mask = mask.reshape(shape)
        
        X_masked = X.clone()
        X_masked[~mask] = -1e6
        return nn.functional.softmax(X_masked, dim=-1)

# 定义 AdditiveAttention 类
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

class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.1):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X):
        # 确保X具有正确的形状：(seq_len, batch_size)
        if X.dim() == 2:
            # 如果X是(batch_size, seq_len)，则进行转置
            X = X.transpose(0, 1)
        
        # 应用嵌入并通过RNN
        X = self.embedding(X)  # 现在X应该是(seq_len, batch_size, embed_size)
        outputs, hidden_state = self.rnn(X)
        return outputs, hidden_state

class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
        # 初始化 attention_weights 用于存储每个时间步的注意力权重
        self.attention_weights = []

    def init_state(self, enc_outputs, enc_valid_lens=None, *args):
        # 修正：正确处理编码器输出的元组
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        
        # 清空之前存储的注意力权重
        self.attention_weights = []  
        
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            
            # 存储注意力权重 - 修改这里
            self.attention_weights.append(self.attention.attention_weights)
            
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            
        outputs = self.dense(torch.cat(outputs, dim=0))
        
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X, Y, state=None):
        enc_outputs, hidden_state = self.encoder(X)
        if state is None:
            state = self.decoder.init_state((enc_outputs, hidden_state), enc_valid_lens=None)
        output, state = self.decoder(Y, state=state)
        return output, state

from tqdm import tqdm

def train_seq2seq(model, train_iter, lr, num_epochs, tgt_vocab, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    train_loss = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        # Create progress bar using tqdm
        progress_bar = tqdm(enumerate(train_iter), total=len(train_iter), 
                           desc=f"Epoch {epoch+1}/{num_epochs}", 
                           ncols=100)
        
        for batch_idx, (X, Y) in progress_bar:
            X, Y = X.to(device), Y.to(device)
            
            Y_input = Y[:, :-1]  # Use first n-1 tokens of Y as input
            Y_target = Y[:, 1:]  # Use last n-1 tokens of Y as target
            
            Y_hat, _ = model(X, Y_input)
            
            loss = loss_fn(Y_hat.reshape(-1, Y_hat.shape[-1]), Y_target.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            total_loss += loss.item()
            
            # Update the loss displayed in the progress bar
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_iter)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch + 1} completed, Average Loss: {avg_loss:.4f}")
    
    return train_loss

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载和预处理函数
# 2. 更新sentence_to_indices函数，正确处理max_len
def sentence_to_indices(sentence, vocab, max_len=None):
    indices = [vocab.get(word, vocab['<unk>']) for word in sentence.split()]
    if max_len is not None:
        if len(indices) > max_len:
            indices = indices[:max_len]  # 截断
        else:
            indices = indices + [vocab['<pad>']] * (max_len - len(indices))  # 填充
    return indices


def build_vocab(sentences, max_vocab_size=10000):
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence.split())
    
    # 特殊标记放在前面
    vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
    
    # 添加最常见的词
    for word, _ in counter.most_common(max_vocab_size - len(vocab)):
        if word not in vocab:  # 避免重复
            vocab[word] = len(vocab)
            
    return vocab

def pad_batch(batch):
    src_batch, tgt_batch = zip(*batch)
    
    # 添加<bos>和<eos>标记
    src_batch = [[2] + seq + [3] for seq in src_batch]
    tgt_batch = [[2] + seq + [3] for seq in tgt_batch]
    
    # 计算最大长度
    src_max_len = max(len(seq) for seq in src_batch)
    tgt_max_len = max(len(seq) for seq in tgt_batch)
    
    # 填充序列
    src_padded = [seq + [0] * (src_max_len - len(seq)) for seq in src_batch]
    tgt_padded = [seq + [0] * (tgt_max_len - len(seq)) for seq in tgt_batch]
    
    src_tensor = torch.tensor(src_padded, dtype=torch.long).to(device)
    tgt_tensor = torch.tensor(tgt_padded, dtype=torch.long).to(device)
    
    return src_tensor, tgt_tensor

def load_data_from_file(file_path, batch_size, num_steps, max_vocab_size=10000):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pairs = [line.strip().split('\t')[:2] for line in lines]  
    
    src_sentences = [pair[0] for pair in pairs]
    tgt_sentences = [pair[1] for pair in pairs]

    src_vocab = build_vocab(src_sentences, max_vocab_size)
    tgt_vocab = build_vocab(tgt_sentences, max_vocab_size)

    src_sentences_idx = [sentence_to_indices(sentence, src_vocab, num_steps) for sentence in src_sentences]
    tgt_sentences_idx = [sentence_to_indices(sentence, tgt_vocab, num_steps) for sentence in tgt_sentences]

    data = list(zip(src_sentences_idx, tgt_sentences_idx))
    data_iter = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)
    return src_vocab, tgt_vocab, data_iter

# 主程序
if __name__ == "__main__":
    data_path = '/home/yyz/NNDL-Class/Project5/Data/fra-eng'
    file_path = os.path.join(data_path, 'fra.txt')
    
    # 超参数
    batch_size = 1024
    num_steps = 20
    
    # 加载数据
    src_vocab, tgt_vocab, train_iter = load_data_from_file(file_path, batch_size, num_steps)
    
    # 模型参数
    embed_size, num_hiddens, num_layers, dropout = 64, 128, 2, 0.2
    
    # 模型实例
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    net = net.to(device)
    
    # 训练参数
    lr = 0.001
    num_epochs = 50
    
    # 训练模型
    train_loss = train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig('/home/yyz/NNDL-Class/Project5/Result/training_loss_curve.png')
    
    # 可视化注意力权重并保存
    sample_attention = decoder.attention_weights[0]
    attention_matrix = sample_attention.squeeze().detach().cpu().numpy()
            
    # 使用matplotlib绘制热图并保存
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_matrix, cmap="YlGnBu", annot=False, cbar=True)
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.title("Attention Heatmap")
    plt.savefig('/home/yyz/NNDL-Class/Project5/Result/attention_heatmap.png')
    plt.close()
    
    print("Training completed and results saved.")
