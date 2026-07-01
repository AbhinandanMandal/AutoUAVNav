

"""
Deep Q Network implementation
For more, visit https://arxiv.org/abs/1312.5602
https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae 
"""

import torch.nn as nn
# Deep Q Network
# We're utilizizng 128, 156, 128 structure with ReLU() activation
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.net(x)
