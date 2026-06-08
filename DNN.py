
# Deep Neural Network architecute based on the base paper
import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn. ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)
