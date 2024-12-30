import torch
from torch import nn
# BP神经网络模型
class BPNetwork(nn.Module):
    def __init__(self, input_dim):
        super(BPNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)