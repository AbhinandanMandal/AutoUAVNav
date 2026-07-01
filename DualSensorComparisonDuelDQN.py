

"""
Implementation of Duel DQN algorithm in Dual sensor fusion case
The idea of initial taken from base paper https://ieeexplore.ieee.org/document/10475692 

We consider 3 sensors for this simulation approach
1. LiDAR
2. IMU
3. Camera

In this dual sensor comparison approach, we will consider fusion of
1. LiDAR + Camera
2. Camera + IMU
3. LiDAR + Camera
4. LiDAR + Camera + IMU (combined)
"""

# Importing libraries and files for DQN algorithm running
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from RLAgent import RLAgent
import matplotlib.pyplot as plt
from HyperparametersConfig import Config
from ObstacleGrid2D import build_obstacle_map
from TrainAgentDualSensorFusion import train_agent
from SensorFusionEnv import SensorFusionUAVGridEnv, DualSensorFusionUAVGridEnv
from ExpertWarmStart import astar_path, prefill_replay, behavior_clone
from TrainAgentDualSensorFusion import greedy_rollout, path_length, count_corners
from PlotDualSensorFusion import plot_paths, plot_reward_curves, plot_metric_bars

# reproducibility and device setting
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print(DEVICE)

# Hyperparameter configuration
cfg = Config()
print(cfg)

# Obstacle mapping for 2D grid environment
OBSTACLES = build_obstacle_map()
print(len(OBSTACLES))

# Path files for respective fields saving
CSV_PATH = Path(
    "DualSensorFusionResults/dual_sensor_comparison_metrics_DQN.csv")
EPISODE_REWARD_CSV_PATH = Path(
    "DualSensorFusionResults/dual_sensor_episode_rewards_DQN.csv")
PLOT_PATH = Path("DualSensorFusionResults/dual_sensor_paths_DQN.png")
REWARD_PLOT_PATH = Path(
    "DualSensorFusionResults/dual_sensor_reward_curves_DQN.png")
BAR_PATH = Path("DualSensorFusionResults/dual_sensor_metric_bars_DQN.png")
CHECKPOINT_DIR = Path("DualSensorFusionResults/dual_sensor_rl_checkpoints_DQN")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# A comprehensive environment for UAV navigation
env = SensorFusionUAVGridEnv(cfg, OBSTACLES)
env.reset()
print(env.state_size)
print(env.action_size)


# Sensor modes for sensor fusion
SENSOR_MODES = {
    "LiDAR + IMU": ("lidar", "imu"),
    "Camera + IMU": ("camera", "imu"),
    "LiDAR + Camera": ("lidar", "camera"),
    "LiDAR + Camera + IMU": ("lidar", "camera", "imu"),
}

for mode_name, sensors in SENSOR_MODES.items():
    env = DualSensorFusionUAVGridEnv(cfg, OBSTACLES, sensors, mode_name)
    print(f"{mode_name:22s} | state_size={env.state_size:2d} | sensors={sensors}")


ALGORITHMS = ["Dueling DQN"]
RUN_TRAINING = True
USE_EXPERT_WARMSTART = True
histories = {}
agents = {}
rollouts = {}
rows = []
expert_path = astar_path()  # A* is being used as a expert warm path
print(len(expert_path)-1)

for mode_name, sensors in SENSOR_MODES.items():
    env = DualSensorFusionUAVGridEnv(cfg, OBSTACLES, sensors, mode_name)
    for algorithm in ALGORITHMS:
        print(f"Training {algorithm} with {mode_name}...")
        agent = RLAgent(env.state_size, env.action_size, cfg, algorithm)
        if USE_EXPERT_WARMSTART and expert_path:
            prefill_replay(env, agent, expert_path, repeats=50)
            behavior_clone(env, agent, expert_path,
                           epochs=120 if algorithm == "IDDQN" else 60)
        checkpoint_name = f"{mode_name.lower().replace(' + ', '_').replace(' ', '_')}_{algorithm.lower().replace(' ', '_')}.pth"
        checkpoint_path = CHECKPOINT_DIR / checkpoint_name
        key = f"{mode_name} | {algorithm}"
        if RUN_TRAINING:
            histories[key] = train_agent(env, agent, verbose_every=100)
            torch.save(agent.current_net.state_dict(), checkpoint_path)
        elif checkpoint_path.exists():
            agent.current_net.load_state_dict(
                torch.load(checkpoint_path, map_location=DEVICE))
            agent.target_net.load_state_dict(agent.current_net.state_dict())
            histories[key] = {"reward": [], "steps": [],
                              "success": [], "loss": [], "final_energy": []}
        else:
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        path, energy_trace, total_reward, info, elapsed_ms = greedy_rollout(
            env, agent)
        agents[key] = agent
        rollouts[key] = {"sensor_mode": mode_name, "algorithm": algorithm, "path": path,
                         "energy": energy_trace, "reward": total_reward, "info": info, "elapsed_ms": elapsed_ms}
        rows.append({
            "sensor_mode": mode_name,
            "active_sensors": "+".join(sensors),
            "algorithm": algorithm,
            "state_size": env.state_size,
            "reward": total_reward,
            "steps": len(path) - 1,
            "path_length": path_length(path),
            "corners": count_corners(path),
            "success": bool(info.get("reached", False)),
            "final_energy": float(info.get("energy", env.energy)),
            "energy_used": float(cfg.initial_energy - info.get("energy", env.energy)),
            "inference_time_ms": elapsed_ms,
            "collision": bool(info.get("collision", False)),
            "out_of_bounds": bool(info.get("out_of_bounds", False)),
            "energy_depleted": bool(info.get("energy_depleted", False)),
        })

comparison_df = pd.DataFrame(rows)
comparison_df.to_csv(CSV_PATH, index=False)

reward_rows = []
for key, history in histories.items():
    mode_name, algorithm = key.split(" | ", 1)
    for episode, reward in enumerate(history["reward"], start=1):
        reward_rows.append(
            {"sensor_mode": mode_name, "algorithm": algorithm, "episode": episode, "reward": reward})
pd.DataFrame(reward_rows).to_csv(EPISODE_REWARD_CSV_PATH, index=False)
print(comparison_df)

# Plotting results and metrics
plot_paths(rollouts)
plot_reward_curves(histories)
plot_metric_bars(comparison_df)
plt.show()
