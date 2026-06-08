
from hyperparameters import MEMORY_SIZE, LR, EPS_DECAY, EPS_END, EPS_START, BATCH_SIZE, GAMMA
from collections import deque
from DNN import DNN
import torch.optim as optim
import torch.nn as nn
import math
import random
import torch
import numpy as np

# For CPU device
device = "cuda" if torch.cuda.is_available() else "cpu"


class IDDQNAgent:
    def __init__(self, state_size, action_size):
        self.device = torch.device(device)

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)  # Maximum length of deque

        self.current_net = DNN(state_size, action_size).to(self.device)
        self.target_net = DNN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(
            self.current_net.state_dict())  # Loading parameters
        self.target_net.eval()  # Target net only for testing. Not for training

        self.optimizer = optim.Adam(self.current_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Adaptive Epsilon-Greedy Learning Strategy
    def act(self, state, episode):
        eps_k = EPS_END + ((EPS_START - EPS_END) /
                           (1 + math.exp(episode/EPS_DECAY)))
        if random.random() < eps_k:
            return random.randrange(self.action_size)  # Exploration

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.current_net(state_tensor)  # Q value Q(s,a)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(
            *batch)  # Zipping all the associated values
        states = torch.FloatTensor(np.array(states)).to(self.device)  # State of the UAV
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # Actions of the UAV
        rewards = torch.FloatTensor(rewards).unsqueeze(
            1).to(self.device)  # Rewards associated with it
        # Next state into state, action space
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(
            1).to(self.device)  # Completion of the task ?

        # For Double DQN
        next_actions = self.current_net(
            next_states).argmax(1).unsqueeze(1)  # a
        next_q_values = self.target_net(next_states).gather(
            1, next_actions)  # q_estimate
        target_q = rewards + (GAMMA*next_q_values*(1-dones))  # target q
        current_q = self.current_net(states).gather(1, actions)

        # Finding loss and Optimize
        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()  # backprop
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.current_net.state_dict())
