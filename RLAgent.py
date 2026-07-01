
"""
# Our RL agent will be trained on 4 speific algorithm
# DQN, Double DQN, Dueling DQN, IDDQN (https://ieeexplore.ieee.org/document/10475692)
"""

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from DeepQNetwork import QNetwork
from DuelingQNetwork import DuelingQNetwork
from ReplayBufferDualSensorFusion import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RLAgent:
    def __init__(self, state_size, action_size, config, algorithm):
        self.cfg = config
        self.algorithm = algorithm
        self.action_size = action_size
        use_dueling = algorithm in {"Dueling DQN", "IDDQN"}
        net_cls = DuelingQNetwork if use_dueling else QNetwork
        self.current_net = net_cls(state_size, action_size).to(DEVICE)
        self.target_net = net_cls(state_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.current_net.state_dict())
        self.target_net.eval()
        self.memory = ReplayBuffer(config.memory_size)
        self.optimizer = optim.Adam(
            self.current_net.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()

    # Specific exploration strategy for IDDQN algorithm.
    # Taken from paper Path Planning of Autonomous Mobile Robot in Comprehensive Unknown Environment Using Deep Reinforcement Learning
    def epsilon(self, episode):
        if self.algorithm == "IDDQN":
            return self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) / (1.0 + math.exp(episode / self.cfg.eps_decay))
        decay = max(0.05, self.cfg.eps_start * (0.995 ** episode))
        return max(self.cfg.eps_end, decay)

    def act(self, state, episode=10_000, greedy=False):
        eps = 0.0 if greedy else self.epsilon(episode)
        if random.random() < eps:
            return random.randrange(self.action_size)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32,
                             device=DEVICE).unsqueeze(0)
            return int(self.current_net(s).argmax(dim=1).item())

    # Learns from replay buffer and updates current network
    def learn(self):
        if len(self.memory) < self.cfg.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.cfg.batch_size)
        q_values = self.current_net(states).gather(1, actions)
        with torch.no_grad():
            if self.algorithm in {"Double DQN", "IDDQN"}:
                next_actions = self.current_net(
                    next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states).max(
                    dim=1, keepdim=True).values
            target = rewards + self.cfg.gamma * next_q * (1.0 - dones)
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.current_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    # Hard copy or soft target updates the target network
    def update_target(self, soft=False):
        if soft:
            tau = self.cfg.soft_tau
            for target_param, current_param in zip(self.target_net.parameters(), self.current_net.parameters()):
                target_param.data.copy_(
                    tau * current_param.data + (1.0 - tau) * target_param.data)
        else:
            self.target_net.load_state_dict(self.current_net.state_dict())
