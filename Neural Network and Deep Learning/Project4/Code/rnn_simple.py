import os
import torch
import re
import collections
import math
import random
from torch import nn
from torch.nn import functional as F
from longdata import load_corpus_time_machine, Vocab, count_corpus,load_data_time_machine
data_dir = '/home/yyz/NNDL-Class/Project4/Data'
result_dir = '/home/yyz/NNDL-Class/Project4/Result'

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

# 定义模型
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size).float()
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

# 训练函数定义（来自教材第8.6节）
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    def grad_clipping(net, theta):
        if isinstance(net, nn.Module):
            params = [p for p in net.parameters() if p.requires_grad]
        else:
            params = net.params
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm

    loss = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(net.parameters(), lr)
    perplexities = []

    for epoch in range(num_epochs):
        state, metric = None, [0.0, 0.0]
        for X, Y in train_iter:
            if state is None or use_random_iter:
                state = net.begin_state(batch_size=X.shape[0], device=device)
            else:
                if isinstance(state, tuple):
                    state = tuple(s.detach() for s in state)
                else:
                    state = state.detach()
            X, Y = X.to(device), Y.T.reshape(-1).to(device)
            y_hat, state = net(X, state)
            l = loss(y_hat, Y.long())
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
            metric[0] += l.item() * Y.numel()
            metric[1] += Y.numel()

        ppl = torch.exp(torch.tensor(metric[0] / metric[1])).item()
        perplexities.append(ppl)
        print(f'epoch {epoch + 1}, perplexity {ppl:.1f}')

    print(predict_ch8('time traveller', 50, net, vocab, device))
    print(predict_ch8('traveller', 50, net, vocab, device))

    # 返回最后困惑度和所有困惑度序列
    return ppl, perplexities

def predict_ch8(prefix, num_preds, net, vocab, device):
    """在 prefix 后面生成 num_preds 个字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]  # 输入第一个字符

    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    # 预热期：先输入 prefix 中的其余字符，不生成输出，只更新 state
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

    # 生成 num_preds 个新字符
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    return ''.join([vocab.idx_to_token[i] for i in outputs])


# 训练模型
num_hiddens = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rnn_layer = nn.RNN(len(vocab), num_hiddens)
model = RNNModel(rnn_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 1
# 训练模型
ppl, perplexity_curve = train_ch8(model, train_iter, vocab, lr, num_epochs, device)

# 保存结果
with open(f"{result_dir}/rnn_concise_results.txt", "a") as f:
    f.write(f"Model: RNN with PyTorch API\n")
    f.write(f"Perplexity: {ppl:.1f}\n")
    f.write(f"Prediction for 'time traveller': {predict_ch8('time traveller', 50, model, vocab, device)}\n")
    f.write(f"Prediction for 'traveller': {predict_ch8('traveller', 50, model, vocab, device)}\n\n")

# 绘制并保存训练曲线图
import matplotlib.pyplot as plt
import os

plt.figure()
plt.plot(range(1, len(perplexity_curve) + 1), perplexity_curve)
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("Training Perplexity Over Epochs")
os.makedirs(result_dir, exist_ok=True)
plt.savefig(os.path.join(result_dir, "rnn_training_curve.png"))
