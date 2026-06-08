
# Hyperparamters for the project
GRID_SIZE = 30
MAX_STEPS = 200
EPISODES = 1000
LR = 0.0025
GAMMA = 0.9
BATCH_SIZE = 64
MEMORY_SIZE =  10000
TARGET_UPDATE = 4 # Update target network every 4 steps
SAFE_RADIUS = 0.6 # Minimum safe distance from the obstacle

# Parameters for Adaptive Epsilon-Greedy Policy
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 500


