

""" 
This file trains the agent for exploration and exploitation
with respect to different algorithms.

Specially it considers Improved DQN algorithm, which acts as a base of this UAV navigation.
For more details regarding IDDQN, refer https://ieeexplore.ieee.org/document/10475692  

This train_agent() function gives total_reward of the agent in his exploration and exploitation
It's total number of steps, It's success i.e. whether it reaches goal or not
Its total number of loss and final energy (if agent successfully able to explore in the 2D grid environment)
"""
import math
import time  # we're also considering execution time
import numpy as np


# Function for training an agent in exploration and exploitation
def train_agent(env, agent, episodes=None, verbose_every=100):
    episodes = episodes or env.cfg.episodes
    history = {"reward": [], "steps": [],
               "success": [], "loss": [], "final_energy": []}
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        losses = []
        final_info = {}
        for _ in range(env.cfg.max_steps):
            action = agent.act(state, episode)
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            if agent.algorithm == "IDDQN":
                agent.update_target(soft=True)
            state = next_state
            total_reward += reward
            final_info = info
            if done:
                break
        if agent.algorithm != "IDDQN" and episode % env.cfg.target_update == 0:
            agent.update_target(soft=False)
        history["reward"].append(total_reward)
        history["steps"].append(env.steps)
        history["success"].append(bool(final_info.get("reached", False)))
        history["loss"].append(float(np.mean(losses)) if losses else np.nan)
        history["final_energy"].append(
            float(final_info.get("energy", env.energy)))
        if verbose_every and (episode + 1) % verbose_every == 0:
            recent_success = np.mean(history["success"][-verbose_every:])
            print(f"{agent.algorithm:12s} | episode {episode + 1:4d}/{episodes} | reward {total_reward:8.2f} | energy {env.energy:6.2f} | recent success {recent_success:.2f}")
    return history


# Evaluates the trained policy greedily and returns path, reward, info time.
def greedy_rollout(env, agent):
    start_time = time.perf_counter()
    state = env.reset()
    total_reward = 0.0
    final_info = {}
    for _ in range(env.cfg.max_steps):
        action = agent.act(state, greedy=True)
        state, reward, done, info = env.step(action)
        total_reward += reward
        final_info = info
        if done:
            break
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return list(env.path), list(env.energy_history), total_reward, final_info, elapsed_ms


# Total path length covered by agent in exploration of the UAV 2D grid
def path_length(path):
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:])))


# Total number of corners taken by agent in exploration
def count_corners(path):
    if len(path) < 3:
        return 0
    corners = 0
    prev = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for a, b in zip(path[1:-1], path[2:]):
        step = (b[0] - a[0], b[1] - a[1])
        if step != prev:
            corners += 1
        prev = step
    return corners
