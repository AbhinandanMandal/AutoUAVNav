

"""
Implementation of IDDQN algorithm in Individual sensor fusion case
The idea of initial taken from base paper https://ieeexplore.ieee.org/document/10475692 

We consider 3 sensors for this simulation approach
1. LiDAR
2. IMU
3. Camera

In this sensor comparison approach, we will consider fusion of
1. LiDAR 
2. Camera
3. IMU
4. LiDAR + IMU
5. Camera + IMU
6. LiDAR + Camera
4. LiDAR + Camera + IMU
"""

# Importing libraries and files for DQN algorithm running
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from Utils.RLAgent import RLAgent
import matplotlib.pyplot as plt
from Utils.HyperparametersConfig import Config
from Utils.ObstacleGrid2D import build_obstacle_map
from TrainAgent.TrainAgentDualSensorFusion import train_agent
from Utils.SensorFusionEnv import SensorFusionUAVGridEnv, IndividualSensorUAVGridEnv
from Utils.ExpertWarmStart import astar_path, prefill_replay, behavior_clone
from TrainAgent.TrainAgentIndividualSensorFusion import greedy_rollout, path_length, count_corners
from PlotFunction.PlotIndividualSensorFusion import plot_paths, plot_reward_curves, plot_metric_bars

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
expert_path = astar_path()
len(expert_path) - 1

# Path files for respective fields saving
CSV_PATH = Path(
    "AttentionIndividualSensorFusionResults/dual_sensor_comparison_metrics_A-IDDQN.csv")
EPISODE_REWARD_CSV_PATH = Path(
    "AttentionIndividualSensorFusionResults/dual_sensor_episode_rewards_A-IDDQN.csv")
PLOT_PATH = Path("AttentionIndividualSensorFusionResults/dual_sensor_paths_A-IDDQN.png")
REWARD_PLOT_PATH = Path(
    "AttentionIndividualSensorFusionResults/dual_sensor_reward_curves_A-IDDQN.png")
BAR_PATH = Path(
    "AttentionIndividualSensorFusionResults/dual_sensor_metric_bars_A-IDDQN.png")
CHECKPOINT_DIR = Path(
    "AttentionIndividualSensorFusionResults/dual_sensor_rl_checkpoints_A-IDDQN")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# A comprehensive environment for UAV navigation
env = SensorFusionUAVGridEnv(cfg, OBSTACLES)
env.reset()
print(env.state_size)
print(env.action_size)


# Sensor modes for sensor fusion
SENSOR_MODES = {
    "LiDAR": ("lidar"),
    "Camera": ("camera"),
    "IMU": ("imu"),
    "LiDAR + Camera + IMU": ("lidar", "camera", "imu"),
}

for mode_name, sensors in SENSOR_MODES.items():
    env = IndividualSensorUAVGridEnv(cfg, OBSTACLES, sensors, mode_name)
    print(f"{mode_name:22s} | state_size={env.state_size:2d} | sensors={sensors}")


ALGORITHMS = ["A-IDDQN"]
RUN_TRAINING = True
USE_EXPERT_WARMSTART = True

# Per-algorithm behavior-cloning epoch budget. A-IDDQN gets a slightly
# larger budget (matching IDDQN's previous "extra" allocation) since it has
# more parameters to warm-start, but each algorithm's BC budget is now
# defined exactly once (the original notebook accidentally ran the
# prefill/behavior-clone block twice for every algorithm).
BC_EPOCHS = {
    "DQN": 60,
    "Double DQN": 60,
    "Dueling DQN": 60,
    "IDDQN": 120,
    "A-IDDQN": 120,
}

histories = {}
agents = {}
rollouts = {}
rows = []

for mode_name, sensors in SENSOR_MODES.items():
    env = IndividualSensorUAVGridEnv(cfg, OBSTACLES, sensors, mode_name)
    for algorithm in ALGORITHMS:
        print(f"Training {algorithm} with {mode_name}...")
        agent = RLAgent(env.state_size, env.action_size, cfg, algorithm)

        if USE_EXPERT_WARMSTART and expert_path:
            prefill_replay(env, agent, expert_path, repeats=50)
            behavior_clone(env, agent, expert_path,
                           epochs=BC_EPOCHS.get(algorithm, 60))

        mode_slug = mode_name.lower().replace(
            " + ", "_").replace(" ", "_").replace("-", "_")
        algorithm_slug = algorithm.lower().replace(" ", "_").replace("-", "_")
        checkpoint_path = CHECKPOINT_DIR / f"{mode_slug}_{algorithm_slug}.pth"
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

        path, energy_trace, total_reward, info, elapsed_ms, attention_trace = greedy_rollout(
            env, agent)
        agents[key] = agent
        rollouts[key] = {
            "sensor_mode": mode_name,
            "algorithm": algorithm,
            "path": path,
            "energy": energy_trace,
            "reward": total_reward,
            "info": info,
            "elapsed_ms": elapsed_ms,
            "attention": attention_trace,
        }
        rows.append({
            "sensor_mode": mode_name,
            "active_sensors": "+".join(sensors),
            "algorithm": algorithm,
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
            {"sensor_mode": mode_name, "algorithm": algorithm,
                "episode": episode, "reward": reward}
        )
pd.DataFrame(reward_rows).to_csv(EPISODE_REWARD_CSV_PATH, index=False)
comparison_df


# Plotting results and metrics
plot_paths(rollouts)
plot_reward_curves(histories)
plot_metric_bars(comparison_df)
plt.show()
