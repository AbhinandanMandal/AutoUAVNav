

"""  
A comprehensive class of UAV grid environment
It returns the info of UAV and obstacle and how effective is the journey of UAV
It returns target_distance, obstacle_distance,
Whether UAV will reach target or collied before that
It also gave information regarding whether UAV goes out of bounds or not
It also gave information regarding UAV's current energy, energy cost for simulating source to target, energy depleted in simulation
It also represents it's motion behaviour, whether is smooth or criss cross
Also it gaves two important information, whether any obstacle with camera input and lidar reading
"""

import math
import math
import numpy as np


class SensorFusionUAVGridEnv:
    ACTIONS = np.array([
        [0, 1], [0, -1], [-1, 0], [1, 0],
        [1, 1], [1, -1], [-1, 1], [-1, -1],
    ], dtype=np.float32)
    ACTION_NAMES = ["up", "down", "left", "right", "upper_right",
                    "lower_right", "upper_left", "lower_left"]  # Action space for the agent
    # LiDAR detection direction is up, down, left, right
    LIDAR_DIRECTIONS = np.array(
        [[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=np.int32)

    def __init__(self, config, obstacles):
        self.cfg = config
        self.obstacle_cells = set(obstacles)
        self.obstacles = np.array(sorted(obstacles), dtype=np.float32)
        self.start_pos = np.array(config.start, dtype=np.float32)
        self.target_pos = np.array(config.target, dtype=np.float32)
        self.action_size = len(self.ACTIONS)

        """ 
        Environment state vector has 4 parts
        5 for base state features [x, y, dist from target, dist from obstacle, normalized remaining energy]

        12 LiDAR readings
        LiDAR has 4 directions for readings [up, down, left, right]
        And we consider that upto 3 blocks it can sense obstacle presence
        So 4 directions x 3 grid cells each = 12

        self.action_size = 8 directions action for agent

        4 for IMU styled features
        IMU helps drone to stabilize and effectively navigate
        here 4 IMU features are [last motion dx, last motion dy, straight motion indicator, corner motion indicator]
        """
        self.state_size = 5 + 12 + self.action_size + 4
        self.reset()  # Places UAV at start position with resettign energy and history of navigation

    def reset(self):
        self.pos = self.start_pos.copy()
        self.steps = 0
        self.done = False
        self.energy = float(self.cfg.initial_energy)
        self.last_action = None
        self.last_motion = np.zeros(2, dtype=np.float32)
        self.last_motion_kind = "start"
        self.path = [tuple(map(int, self.pos))]
        self.energy_history = [self.energy]
        return self._state()

    # Bounding condition with the boundary
    def _in_bounds(self, cell):
        return 0 <= cell[0] <= self.cfg.grid_size and 0 <= cell[1] <= self.cfg.grid_size

    # Blocking condition with blocks (obstacles)
    def _is_blocked_cell(self, cell):
        cell = tuple(map(int, cell))
        return (not self._in_bounds(cell)) or cell in self.obstacle_cells

    # For calculation of target distance
    def _target_distance(self, pos=None):
        p = self.pos if pos is None else np.array(pos, dtype=np.float32)
        return float(np.linalg.norm(p - self.target_pos))

    # For calculation of obstacle distance
    def _obstacle_distance(self, pos=None):
        p = self.pos if pos is None else np.array(pos, dtype=np.float32)
        if len(self.obstacles) == 0:
            return math.sqrt(2) * self.cfg.grid_size
        return float(np.min(np.linalg.norm(self.obstacles - p, axis=1)))

    # For LiDAR reading
    def _lidar_scan(self):
        readings = []
        current = tuple(map(int, self.pos))
        for direction in self.LIDAR_DIRECTIONS:
            for distance in range(1, 4):
                cell = (current[0] + int(direction[0]) * distance,
                        current[1] + int(direction[1]) * distance)
                readings.append(1.0 if self._is_blocked_cell(cell) else 0.0)
        return np.array(readings, dtype=np.float32)

    # Camera's reading for detecting obstacle and efficient navigation
    def _camera_flags(self):
        flags = []
        current = tuple(map(int, self.pos))
        for action in self.ACTIONS.astype(int):
            dx, dy = int(action[0]), int(action[1])
            proposed = (current[0] + dx, current[1] + dy)
            blocked = self._is_blocked_cell(proposed)
            if dx != 0 and dy != 0:
                blocked = blocked or self._is_blocked_cell(
                    (current[0] + dx, current[1])) or self._is_blocked_cell((current[0], current[1] + dy))
            flags.append(1.0 if blocked else 0.0)
        return np.array(flags, dtype=np.float32)

    # For IMU reading
    def _imu_features(self):
        dx, dy = self.last_motion
        straight = 1.0 if self.last_motion_kind == "straight" else 0.0
        corner = 1.0 if self.last_motion_kind == "corner" else 0.0
        return np.array([dx, dy, straight, corner], dtype=np.float32)

    # Calculating current stage of the UAV at present condition
    def _state(self):
        max_dist = math.sqrt(2) * self.cfg.grid_size
        od = min(self._obstacle_distance(), max_dist)
        base = np.array([
            self.pos[0] / self.cfg.grid_size,
            self.pos[1] / self.cfg.grid_size,
            self._target_distance() / max_dist,
            od / max_dist,
            self.energy / self.cfg.initial_energy,
        ], dtype=np.float32)
        return np.concatenate([base, self._lidar_scan(), self._camera_flags(), self._imu_features()]).astype(np.float32)

    # This functio controlling motion of drone
    def _motion_kind(self, action):
        if self.last_action is None:
            return "straight"
        previous = self.ACTIONS[self.last_action]
        current = self.ACTIONS[action]
        return "straight" if np.array_equal(previous, current) else "corner"

    # Energy cost for UAV navigation
    # It associated base energy cost, diagonal energy cost, corner energy cost, obstacle risk energy cost, camera warning energy cost
    def _energy_cost(self, action, camera_blocked):
        move = self.ACTIONS[action]
        cost = self.cfg.base_energy_cost
        if abs(move[0]) + abs(move[1]) == 2:
            cost += self.cfg.diagonal_energy_cost
        if self._motion_kind(action) == "corner":
            cost += self.cfg.corner_energy_cost
        nearby_obstacles = self._lidar_scan().reshape(4, 3)[:, 0].sum()
        cost += self.cfg.obstacle_risk_energy_cost * nearby_obstacles
        if camera_blocked:
            cost += self.cfg.camera_warning_energy_cost
        return float(cost)

    # step function computes
    # new position, collision / out-of-bounds / reached target
    # energy cost from motion type, nearby obstalces and camera warning
    # Reward combining progress, distance, energy changes and penalties
    # It also looked at condition when goal reached, collision, boundary, energy depletion and max steps
    def step(self, action):
        if self.done:
            return self._state(), 0.0, True, {}

        action = int(action)
        previous_td = self._target_distance()
        previous_energy = self.energy
        proposed = self.pos + self.ACTIONS[action]
        camera_flags = self._camera_flags()
        camera_blocked = bool(camera_flags[action] > 0.5)
        energy_cost = self._energy_cost(action, camera_blocked)
        motion_kind = self._motion_kind(action)
        self.steps += 1

        out = proposed[0] < 0 or proposed[0] > self.cfg.grid_size or proposed[1] < 0 or proposed[1] > self.cfg.grid_size
        self.pos = proposed
        self.path.append(tuple(map(int, self.pos)))

        td = self._target_distance()
        od = self._obstacle_distance()
        reached = td <= math.sqrt(2) / 2
        collision = tuple(map(int, self.pos)
                          ) in self.obstacle_cells or od < self.cfg.safe_radius

        self.energy = max(0.0, self.energy - energy_cost)
        if reached:
            self.energy = min(self.cfg.initial_energy,
                              self.energy + self.cfg.goal_energy_bonus)
        energy_delta = self.energy - previous_energy
        energy_depleted = self.energy <= 0.0 and not reached

        reward = self.cfg.step_penalty + self.cfg.distance_scale * \
            td + self.cfg.progress_scale * (previous_td - td)
        reward += self.cfg.energy_reward_scale * energy_delta
        if reached:
            reward += self.cfg.target_reward
        if collision:
            reward += self.cfg.collision_penalty
        if out:
            reward += self.cfg.boundary_penalty
        if energy_depleted:
            reward += self.cfg.depleted_energy_penalty

        self.last_action = action
        self.last_motion = self.ACTIONS[action].copy()
        self.last_motion_kind = motion_kind
        self.energy_history.append(self.energy)
        self.done = bool(
            reached or collision or out or energy_depleted or self.steps >= self.cfg.max_steps)
        info = {
            "target_distance": td,
            "obstacle_distance": od,
            "reached": reached,
            "collision": collision,
            "out_of_bounds": out,
            "energy": self.energy,
            "energy_cost": energy_cost,
            "energy_depleted": energy_depleted,
            "motion_kind": motion_kind,
            "camera_blocked": camera_blocked,
            "lidar_near_obstacles": int(self._lidar_scan().reshape(4, 3)[:, 0].sum()),
        }
        return self._state(), float(reward), self.done, info


class DualSensorFusionUAVGridEnv(SensorFusionUAVGridEnv):
    VALID_SENSORS = {"lidar", "camera", "imu"}

    def __init__(self, config, obstacles, active_sensors, mode_name="sensor mode"):
        self.active_sensors = tuple(active_sensors)
        self.mode_name = mode_name
        invalid = set(self.active_sensors) - self.VALID_SENSORS
        if invalid:
            raise ValueError(f"Unknown sensors: {sorted(invalid)}")
        super().__init__(config, obstacles)
        self.state_size = self._state().shape[0]

    def has_sensor(self, name):
        return name in self.active_sensors

    def _state(self):
        max_dist = math.sqrt(2) * self.cfg.grid_size
        base = np.array([
            self.pos[0] / self.cfg.grid_size,
            self.pos[1] / self.cfg.grid_size,
            self._target_distance() / max_dist,
            self.energy / self.cfg.initial_energy,
        ], dtype=np.float32)
        features = [base]
        if self.has_sensor("lidar"):
            features.append(self._lidar_scan())
        if self.has_sensor("camera"):
            features.append(self._camera_flags())
        if self.has_sensor("imu"):
            features.append(self._imu_features())
        return np.concatenate(features).astype(np.float32)

    def _energy_cost(self, action, camera_blocked):
        move = self.ACTIONS[action]
        cost = self.cfg.base_energy_cost
        if abs(move[0]) + abs(move[1]) == 2:
            cost += self.cfg.diagonal_energy_cost
        if self.has_sensor("imu") and self._motion_kind(action) == "corner":
            cost += self.cfg.corner_energy_cost
        if self.has_sensor("lidar"):
            nearby_obstacles = self._lidar_scan().reshape(4, 3)[:, 0].sum()
            cost += self.cfg.obstacle_risk_energy_cost * nearby_obstacles
        if self.has_sensor("camera") and camera_blocked:
            cost += self.cfg.camera_warning_energy_cost
        return float(cost)


class IndividualSensorUAVGridEnv(SensorFusionUAVGridEnv):
    VALID_SENSORS = {"lidar", "camera", "imu"}

    def __init__(self, config, obstacles, active_sensors, mode_name="sensor mode"):
        self.active_sensors = tuple(active_sensors)
        self.mode_name = mode_name
        invalid = set(self.active_sensors) - self.VALID_SENSORS
        if invalid:
            raise ValueError(f"Unknown sensors: {sorted(invalid)}")
        super().__init__(config, obstacles)
        self.state_size = self._state().shape[0]

    def has_sensor(self, name):
        return name in self.active_sensors

    def _state(self):
        max_dist = math.sqrt(2) * self.cfg.grid_size
        base = np.array([
            self.pos[0] / self.cfg.grid_size,
            self.pos[1] / self.cfg.grid_size,
            self._target_distance() / max_dist,
            self.energy / self.cfg.initial_energy,
        ], dtype=np.float32)
        features = [base]
        if self.has_sensor("lidar"):
            features.append(self._lidar_scan())
        if self.has_sensor("camera"):
            features.append(self._camera_flags())
        if self.has_sensor("imu"):
            features.append(self._imu_features())
        return np.concatenate(features).astype(np.float32)

    def _energy_cost(self, action, camera_blocked):
        move = self.ACTIONS[action]
        cost = self.cfg.base_energy_cost
        if abs(move[0]) + abs(move[1]) == 2:
            cost += self.cfg.diagonal_energy_cost
        if self.has_sensor("imu") and self._motion_kind(action) == "corner":
            cost += self.cfg.corner_energy_cost
        if self.has_sensor("lidar"):
            nearby_obstacles = self._lidar_scan().reshape(4, 3)[:, 0].sum()
            cost += self.cfg.obstacle_risk_energy_cost * nearby_obstacles
        if self.has_sensor("camera") and camera_blocked:
            cost += self.cfg.camera_warning_energy_cost
        return float(cost)
