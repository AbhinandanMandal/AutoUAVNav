

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math
from hyperparameters import GRID_SIZE, SAFE_RADIUS, MAX_STEPS


class UAVGridEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.state_size = 4 # [x, y, dist_to_target, dist_to_obstacle]
        self.action_size = 8 # [up, down, left, right, upper_left, upper_right, down_left, down_right]
        self.target_pos = np.array([28.0, 28.0]) # Target Position

        self.obstacles = [np.array([15.0, 15.0])]
        self.reset()

    def reset(self):
        self.uav_pos = np.array([2.0, 2.0]) # Source Position (SP)
        self.steps = 0
        return self._get_state()
    
    def _get_state(self):
        td = np.linalg.norm(self.uav_pos - self.target_pos) # Distance form target
        od = min([np.linalg.norm(self.uav_pos - obs) for obs in self.obstacles]) # Distance from obstacles
        return np.array([self.uav_pos[0], self.uav_pos[1], td, od], dtype=np.float32) # Return [x, y, dist_to_target, dist_to_obstacle]
    
    def step(self, action):
        self.step+=1
        
        # Possible moves for uav
        # [up, down, left, right, upper left, upper right, down left, down rigt]
        moves={
            0:[0, 1], 1:[0, -1], 2:[-1, 0], 3: [1, 0], 4: [-1, 1], 5: [1, 1], 6: [-1, 1], 7:[1, -1]
        }

        # Moving UAV
        move = np.array(moves[action])
        self.uav_pos = self.uav_pos+move 

        state  =self._get_state()
        td, od = state[2], state[3] # TD, OD

        # Composite Reward Function
        # R = r1 + r2 + r3 + r4
        reward = 0
        done = False

        # r1 : Target Reward 
        if td<=math.sqrt(2)/2:
            reward+=1 # lamda_1
            done = True

        # r2 : continuous heuristic
        reward+=0.1*td # lamda_2

        # r3 : Boundary reward 
        if (self.uav_pos[0]<0 or self.uav_pos[0]>self.grid_size or self.uav_pos[1]<0 or self.uav_pos[1]>self.grid_size):
            reward+=-50 # lamda_3
            done = True
        
        # r4: Obstacle Reward
        if od<SAFE_RADIUS:
            reward+=-50 # lamda_4
            done = True # collision scenario
        
        if self.steps>=MAX_STEPS:
            done = True
        
        return state, reward, done 
    
    
        



    

