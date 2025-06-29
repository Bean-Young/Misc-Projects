import time
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
from longdata import load_data_time_machine
from rnn_simple import RNNModel,predict_ch8
import time
import math
#   设置参数  
batch_size, num_steps = 32, 35
num_hiddens = 256
num_layers = 2
num_epochs, lr = 500, 2
device = d2l.try_gpu()
save_path = '/home/yyz/NNDL-Class/Project4/Result/'

#   读取数据  
train_iter, vocab = load_data_time_machine(batch_size, num_steps)
vocab_size = len(vocab)

#   训练帮助函数  

def train_and_record(model, name):
    print(f"Training {name}...")
    ppl_list = []
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(model.parameters(), lr)
    start_time = time.time()
    
    for epoch in range(num_epochs):
        state, timer = None, d2l.Timer()
        metric = d2l.Accumulator(2)
        for X, Y in train_iter:
            if state is None:
                state = model.begin_state(batch_size=X.shape[0], device=device)
            elif isinstance(state, tuple):
                state = tuple(s.detach() for s in state)
            else:
                state = state.detach()

            y = Y.T.reshape(-1).to(device)
            X = X.to(device)
            y_hat, state = model(X, state)
            l = loss(y_hat, y.long()).mean()
            updater.zero_grad()
            l.backward()
            d2l.grad_clipping(model, 1)
            updater.step()
            metric.add(l * y.numel(), y.numel())

        ppl = math.exp(metric[0] / metric[1])
        ppl_list.append(ppl)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: perplexity {ppl:.2f}")

    training_time = time.time() - start_time
    torch.save(model.state_dict(), f"{save_path}{name}_model.pt")

    # 预测文本
    pred = predict_ch8('time traveller', 50, model, vocab, device)

    # 保存对比结果
    with open(f"{save_path}deep_rnn_comparison.txt", "a") as f:
        f.write(f"Comparison result for {name.upper()} model:\n")
        f.write(f"{name.upper()} Training Time: {training_time:.2f} seconds\n")
        f.write(f"{name.upper()} Final Perplexity: {ppl:.2f}\n")
        f.write(f"{name.upper()} Prediction for 'time traveller': {pred}\n\n")

    return ppl_list

#   训练 LSTM  
lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens, num_layers=num_layers)
lstm_model = RNNModel(lstm_layer, vocab_size).to(device)
lstm_ppl = train_and_record(lstm_model, 'lstm')

#   训练 GRU  
gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens, num_layers=num_layers)
gru_model = RNNModel(gru_layer, vocab_size).to(device)
gru_ppl = train_and_record(gru_model, 'gru')

#   图像保存  
plt.figure()
plt.plot(lstm_ppl, label='LSTM')
plt.plot(gru_ppl, label='GRU')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.legend()
plt.title('GRU vs LSTM Perplexity')
plt.savefig(f'{save_path}perplexity_comparison.png')
