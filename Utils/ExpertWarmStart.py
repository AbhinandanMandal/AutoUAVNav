

"""
We're using A* path planning algorithm as an expert warm start for UAV navigation with RL algorithm
To learn more about A* visit https://www.geeksforgeeks.org/dsa/a-search-algorithm/, 
https://en.wikipedia.org/wiki/A*_search_algorithm 

"""

import math
import heapq
import torch
import torch.nn as nn
import numpy as np
from Utils.HyperparametersConfig import Config
from Utils.ObstacleGrid2D import build_obstacle_map
from Utils.SensorFusionEnv import DualSensorFusionUAVGridEnv

cfg = Config()
OBSTACLES = build_obstacle_map()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A special thing
# A* expert warm start, start_path() computes an A* reference path from start to target through the grid


def astar_path(start=cfg.start, target=cfg.target):
    obstacles = set(OBSTACLES)
    actions = [tuple(map(int, a)) for a in DualSensorFusionUAVGridEnv.ACTIONS]

    def h(cell):
        return math.hypot(cell[0] - target[0], cell[1] - target[1])

    queue = [(h(start), 0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}
    while queue:
        _, cost, current = heapq.heappop(queue)
        if current == target:
            break
        for dx, dy in actions:
            nxt = (current[0] + dx, current[1] + dy)
            if nxt[0] < 0 or nxt[0] > cfg.grid_size or nxt[1] < 0 or nxt[1] > cfg.grid_size or nxt in obstacles:
                continue
            new_cost = cost + math.hypot(dx, dy)
            if new_cost < cost_so_far.get(nxt, float("inf")):
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(queue, (new_cost + h(nxt), new_cost, nxt))
    if target not in came_from:
        return []
    path = []
    cell = target
    while cell is not None:
        path.append(cell)
        cell = came_from[cell]
    return path[::-1]


# Simulates the finding path in the 2D grid environment
def expert_transitions(env, path):
    action_lookup = {tuple(map(int, action)): idx for idx,
                     action in enumerate(env.ACTIONS)}
    transitions = []
    state = env.reset()
    for current, nxt in zip(path[:-1], path[1:]):
        action = action_lookup[(nxt[0] - current[0], nxt[1] - current[1])]
        next_state, reward, done, _ = env.step(action)
        transitions.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            break
    return transitions


# Fill agent memory with expert transition
def prefill_replay(env, agent, path, repeats=50):
    transitions = expert_transitions(env, path)
    for _ in range(repeats):
        for transition in transitions:
            agent.memory.push(*transition)


# Trains the network to imitate the expert behaviour and path via supervised cross-entropy
def behavior_clone(env, agent, path, epochs=200):
    transitions = expert_transitions(env, path)
    if not transitions:
        return None
    states = torch.tensor(
        np.array([t[0] for t in transitions]), dtype=torch.float32, device=DEVICE)
    actions = torch.tensor([t[1] for t in transitions],
                           dtype=torch.long, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        logits = agent.current_net(states)
        loss = loss_fn(logits, actions)
        agent.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            agent.current_net.parameters(), agent.cfg.grad_clip)
        agent.optimizer.step()
    agent.target_net.load_state_dict(agent.current_net.state_dict())
    return float(loss.item())



