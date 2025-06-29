import os
import math
import torch
from torch import nn
from torch.nn import functional as F
from longdata import load_data_time_machine

# 设置数据路径
data_dir = '/home/yyz/NNDL-Class/Project4/Data'
result_dir = '/home/yyz/NNDL-Class/Project4/Result'

class RNNModelScratch:
    """从零开始实现的循环神经网络模型（支持LSTM）"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
        self.device = device

    def __call__(self, X, state):
        # 输入 shape: (batch_size, num_steps)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)  # 转成 (num_steps, batch_size, vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    """基于前缀生成文本序列"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([[outputs[-1]]], device=device)
    
    for y in prefix[1:]:  # 预热
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):
    """裁剪梯度，防止梯度爆炸"""
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in net.params))
    if norm > theta:
        for param in net.params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device):
    """训练一个迭代周期"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 累加训练损失和词元数量

    for X, Y in train_iter:
        if state is None:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(state, tuple):
                state = tuple(s.detach() for s in state)
            else:
                state = state.detach()

        y = Y.T.reshape(-1).to(device)
        X = X.to(device)

        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()

        for param in net.params:
            if param.grad is not None:
                param.grad.zero_()
        l.backward()
        grad_clipping(net, 1)
        updater(batch_size=1)

        metric.add(l * y.numel(), y.numel())

    return math.exp(metric[0] / metric[1])

def sgd(params, lr, batch_size):
    """随机梯度下降"""
    for param in params:
        param.data.sub_(lr * param.grad / batch_size)

import matplotlib.pyplot as plt

def train_ch8(net, train_iter, vocab, lr, num_epochs, device):
    loss = nn.CrossEntropyLoss()
    updater = lambda batch_size: sgd(net.params, lr, batch_size)

    perplexities = []

    for epoch in range(num_epochs):
        ppl = train_epoch_ch8(net, train_iter, loss, updater, device)
        perplexities.append(ppl)
        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            print(f'epoch {epoch + 1}, perplexity {ppl:.1f}')
            print('预测:', predict_ch8('time traveller', 50, net, vocab, device))

    # 绘制困惑度曲线并保存
    plt.figure()
    plt.plot(range(1, num_epochs + 1), perplexities, label='train perplexity')
    plt.xlabel('epoch')
    plt.ylabel('perplexity')
    plt.title('LSTM from Scratch: Train Perplexity')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{result_dir}/lstm_scratch_perplexity.png')
    plt.close()


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()

    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f,
              W_xo, W_ho, b_o, W_xc, W_hc, b_c,
              W_hq, b_q]

    for param in params:
        param.requires_grad_(True)
    return params

def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

def lstm(inputs, state, params):
    (W_xi, W_hi, b_i,
     W_xf, W_hf, b_f,
     W_xo, W_ho, b_o,
     W_xc, W_hc, b_c,
     W_hq, b_q) = params
    H, C = state
    outputs = []

    for X in inputs:
        I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)
        F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)
        O = torch.sigmoid(X @ W_xo + H @ W_ho + b_o)
        C_tilda = torch.tanh(X @ W_xc + H @ W_hc + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = H @ W_hq + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H, C)


def evaluate_perplexity(net, data_iter, vocab, device):
    """评估模型在数据集上的困惑度"""
    loss = nn.CrossEntropyLoss()
    total_loss, total_num = 0.0, 0

    state = None
    for X, Y in data_iter:
        if state is None:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(state, tuple):
                state = tuple(s.detach() for s in state)
            else:
                state = state.detach()

        X, y = X.to(device), Y.T.reshape(-1).to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long())
        total_loss += l.item() * y.numel()
        total_num += y.numel()

    return math.exp(total_loss / total_num)


from d2l import torch as d2l 


# 超参数设置
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens = len(vocab), 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs, lr = 500, 1

model = RNNModelScratch(vocab_size, num_hiddens, device,
                        get_lstm_params, init_lstm_state, lstm)

train_ch8(model, train_iter, vocab, lr, num_epochs, device)

# 评估困惑度与输出结果
ppl = evaluate_perplexity(model, train_iter, vocab, device)
pred1 = predict_ch8('time traveller', 50, model, vocab, device)
pred2 = predict_ch8('traveller', 50, model, vocab, device)

# 保存结果
os.makedirs(result_dir, exist_ok=True)
with open(f"{result_dir}/lstm_scratch_results.txt", "a") as f:
    f.write(f"Model: LSTM from scratch\n")
    f.write(f"Perplexity: {ppl:.1f}\n")
    f.write(f"Prediction for 'time traveller': {pred1}\n")
    f.write(f"Prediction for 'traveller': {pred2}\n\n")
