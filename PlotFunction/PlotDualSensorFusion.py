

"""
This file plots the behaviour of algorithms
Behaviour in sense it plots 
1. Path of the agent to explore 2D grid
2. The total reward curv during exploration and exploitation
3. And various metrics to represent generalization of the agent's behaviour

For more visit: https://matplotlib.org/ 
"""

import numpy as np
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
from Utils.HyperparametersConfig import Config
from Utils.ObstacleGrid2D import build_obstacle_map

# Hyperparameter configuration and obstalces of 2D grid environment
cfg=Config()
OBSTACLES=build_obstacle_map()

BAR_PATH = Path("dual_sensor_metric_bars_DQN.png")
REWARD_PLOT_PATH = Path("dual_sensor_reward_curves_DQN.png")
PLOT_PATH = Path("dual_sensor_paths_DQN.png")

STYLE = {
    "LiDAR + IMU": {"color": "tab:blue", "linestyle": "-", "linewidth": 2.7},
    "Camera + IMU": {"color": "tab:green", "linestyle": "--", "linewidth": 2.7},
    "LiDAR + Camera": {"color": "tab:orange", "linestyle": ":", "linewidth": 3.0},
    "LiDAR + Camera + IMU": {"color": "crimson", "linestyle": "-.", "linewidth": 3.0},
}


def plot_paths(rollouts, save_path=PLOT_PATH):
    fig, ax = plt.subplots(figsize=(8.8, 8.8))
    obstacle_grid = np.zeros(
        (cfg.grid_size + 1, cfg.grid_size + 1), dtype=float)
    for x, y in OBSTACLES:
        if 0 <= x <= cfg.grid_size and 0 <= y <= cfg.grid_size:
            obstacle_grid[y, x] = 1.0
    masked_obstacles = np.ma.masked_where(obstacle_grid == 0, obstacle_grid)
    ax.imshow(masked_obstacles, cmap=plt.cm.gray_r, origin="lower", extent=(-0.5,
              cfg.grid_size + 0.5, -0.5, cfg.grid_size + 0.5), interpolation="nearest", vmin=0, vmax=1)
    for key, result in rollouts.items():
        path = np.array(result["path"])
        mode_name = result["sensor_mode"]
        ax.plot(path[:, 0], path[:, 1], label=key, **STYLE[mode_name])
    ax.scatter(*cfg.start, c="lightpink", s=360, marker="o",
               edgecolors="dimgray", linewidths=1.0, zorder=5)
    ax.scatter(*cfg.target, c="red", s=420, marker="*",
               edgecolors="black", linewidths=0.8, zorder=6)
    ax.text(cfg.start[0] - 1.55, cfg.start[1] -
            0.15, "SP", color="navy", fontsize=11)
    ax.text(cfg.target[0] + 0.70, cfg.target[1] -
            0.35, "TP", color="black", fontsize=11)
    ax.set_xlim(0, cfg.grid_size)
    ax.set_ylim(0, cfg.grid_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0, cfg.grid_size + 1, 2))
    ax.set_yticks(np.arange(0, cfg.grid_size + 1, 2))
    ax.set_xticks(np.arange(0, cfg.grid_size + 1, 1), minor=True)
    ax.set_yticks(np.arange(0, cfg.grid_size + 1, 1), minor=True)
    ax.grid(which="minor", color="#4f4f4f", linewidth=0.85, alpha=0.75)
    ax.grid(which="major", color="#4f4f4f", linewidth=1.05, alpha=0.75)
    ax.set_title("Dual Sensor Fusion Path Comparison", fontsize=14, pad=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower right", frameon=True, fancybox=False,
              framealpha=1.0, edgecolor="black", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    return fig, ax


def plot_reward_curves(histories, window=25, save_path=REWARD_PLOT_PATH):
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, history in histories.items():
        rewards = pd.Series(history["reward"], dtype="float64")
        if rewards.empty:
            continue
        mode_name = key.split(" | ", 1)[0]
        smoothed = rewards.rolling(window=window, min_periods=1).mean()
        ax.plot(smoothed.index + 1, smoothed.values,
                label=key, **STYLE[mode_name])
    ax.set_title(f"Dual Sensor Fusion Reward Moving Average (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    return fig, ax


def plot_metric_bars(df, save_path=BAR_PATH):
    metrics = [("reward", "Reward"), ("steps", "Steps"), ("path_length", "Path Length"), ("corners", "Corners"),
               ("final_energy", "Final Energy"), ("energy_used", "Energy Used"), ("inference_time_ms", "Inference Time (ms)")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    labels = df["sensor_mode"] if df["algorithm"].nunique(
    ) == 1 else df["sensor_mode"] + "\n" + df["algorithm"]
    colors = [STYLE[m]["color"] for m in df["sensor_mode"]]
    for ax, (metric, label) in zip(axes, metrics):
        ax.bar(labels, df[metric], color=colors)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].axis("off")
    fig.suptitle("Dual Sensor Fusion Ablation Metrics", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig, axes


