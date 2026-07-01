
"""
It stores state, action, reward, next_state, done (if the exploration is completed or not) transitions
"""

import torch
import random
import numpy as np 
from collections import deque
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)

    def __len__(self):
        return len(self.data)

    def push(self, state, action, reward, next_state, done):
        self.data.append((state, action, reward, next_state, done))

    # It returns the tensor for training RL agent
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE),
            torch.tensor(actions, dtype=torch.long,
                         device=DEVICE).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32,
                         device=DEVICE).unsqueeze(1),
            torch.tensor(np.array(next_states),
                         dtype=torch.float32, device=DEVICE),
            torch.tensor(dones, dtype=torch.float32,
                         device=DEVICE).unsqueeze(1),
        )
