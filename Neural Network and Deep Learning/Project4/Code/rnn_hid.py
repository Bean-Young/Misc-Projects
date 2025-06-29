import os
import math
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from longdata import load_data_time_machine

# 路径配置
data_dir = '/home/yyz/NNDL-Class/Project4/Data'
result_dir = '/home/yyz/NNDL-Class/Project4/Result'
os.makedirs(result_dir, exist_ok=True)

# 数据加载
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

# 模型定义
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super().__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = rnn_layer.hidden_size
        self.num_directions = 2 if rnn_layer.bidirectional else 1
        self.linear = nn.Linear(self.num_hiddens * self.num_directions, vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size).float()
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        shape = (self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens)
        if isinstance(self.rnn, nn.LSTM):
            return (torch.zeros(shape, device=device), torch.zeros(shape, device=device))
        return torch.zeros(shape, device=device)

# 预测函数
def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# 训练函数
def train_ch8(net, train_iter, vocab, lr, num_epochs, device):
    def grad_clipping(net, theta):
        params = [p for p in net.parameters() if p.requires_grad]
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
            if state is None:
                state = net.begin_state(batch_size=X.shape[0], device=device)
            else:
                state = tuple(s.detach() for s in state) if isinstance(state, tuple) else state.detach()
            X, Y = X.to(device), Y.T.reshape(-1).to(device)
            y_hat, state = net(X, state)
            l = loss(y_hat, Y.long())
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
            metric[0] += l.item() * Y.numel()
            metric[1] += Y.numel()
        ppl = math.exp(metric[0] / metric[1])
        perplexities.append(ppl)
        print(f'epoch {epoch + 1}, perplexity {ppl:.1f}')
    return ppl, perplexities

# 配置与训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_hiddens, num_layers, num_epochs, lr = 256, 2, 500, 1
rnn_layer = nn.RNN(len(vocab), num_hiddens, num_layers=num_layers)
model = RNNModel(rnn_layer, len(vocab)).to(device)

ppl, curve = train_ch8(model, train_iter, vocab, lr, num_epochs, device)

# 保存结果
with open(f"{result_dir}/rnn_multilayer_results.txt", "a") as f:
    f.write(f"Model: 2-layer RNN\n")
    f.write(f"Perplexity: {ppl:.1f}\n")
    f.write(f"Prediction for 'time traveller': {predict_ch8('time traveller', 50, model, vocab, device)}\n")
    f.write(f"Prediction for 'traveller': {predict_ch8('traveller', 50, model, vocab, device)}\n\n")

# 绘图保存
plt.figure()
plt.plot(range(1, len(curve) + 1), curve)
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("2-layer RNN Training Curve")
plt.savefig(os.path.join(result_dir, "rnn_multilayer_curve.png"))
