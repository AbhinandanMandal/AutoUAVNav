
""" 
Hyperparameters are intentionally close to SensorfusionUAVNavigation.py
This notebook changes the active sensor set to compare dual-sensor fusion pairs against full fusion.

Most values are taken directly from https://ieeexplore.ieee.org/document/10475692 
"""

from dataclasses import dataclass

@dataclass
class Config:
    grid_size: int = 30
    start: tuple = (5, 1)
    target: tuple = (14, 29)
    max_steps: int = 160
    episodes: int = 1000
    gamma: float = 0.90
    lr: float = 0.0025
    batch_size: int = 64
    memory_size: int = 20_000
    target_update: int = 10
    soft_tau: float = 0.02
    eps_start: float = 0.90
    eps_end: float = 0.02
    eps_decay: float = 450.0
    safe_radius: float = 0.60
    target_reward: float = 500.0
    collision_penalty: float = -300.0
    boundary_penalty: float = -300.0
    distance_scale: float = -0.02
    progress_scale: float = 3.0
    step_penalty: float = -0.10
    grad_clip: float = 10.0

    # Energy based reward for agent.
    initial_energy: float = 100.0
    base_energy_cost: float = 0.70
    diagonal_energy_cost: float = 0.30
    corner_energy_cost: float = 0.45
    obstacle_risk_energy_cost: float = 0.25
    camera_warning_energy_cost: float = 0.35
    goal_energy_bonus: float = 25.0
    depleted_energy_penalty: float = -250.0
    energy_reward_scale: float = 0.60
