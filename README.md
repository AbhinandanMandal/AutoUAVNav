# Autonomous UAV Navigation
This is the implementation of autonomous UAV navigation using reinforcement learning in 2D grid with expert warm start path planning algorithm in GNSS denied environment.
The initial idea was taken from paper [Path Planning of Autonomous Mobile Robot in Comprehensive Unknown Environment Using Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/10475692/).

<img width="960" height="718" alt="Spying_quadcopter_(cropped)" src="https://github.com/user-attachments/assets/fff83187-131e-4a3a-b7d9-220f78e0c295" />

## Sensors
For this project, three fundamental sensors are taking into consideration,
1. LiDAR
2. Camera
3. IMU
   
Using the above three sensors, the following analysis has been done extensively on multiple scenarios and conditions. 

## Project Structure
The following project has been divided into some set of `individual` and `dual` sensor analysis on fours reinforcement learning algorithm,
1. DQN
2. Double DQN
3. Dueling DQN
4. IDDQN *(propsoed in the following research paper)*

```text
AutoUAVNav/
│
├── PlotFunction/
│   ├── PlotDualSensorFusion.py
│   └── PlotIndividualSensorFusion.py
│
├── QNetwork/
│   ├── DeepQNetwork.py
│   └── DuelingQNetwork.py
│
├── TrainAgent/
│   ├── TrainAgentDualSensorFusion.py
│   └── TrainAgentIndividualSensorFusion.py
│
├── Utils/
│   ├── ExpertWarmStart.py
│   ├── HyperparametersConfig.py
│   ├── ObstacleGrid2D.py
│   ├── ReplayBuffer.py
│   ├── RLAgent.py
│   └── SensorFusionEnv.py
│
├── DualSensorComparisonDQN.py
├── DualSensorComparisonDuelDQN.py
├── DualSensorComparisonDDQN.py
├── DualSensorComparisonIDDQN.py
│
├── IndividualSensorComparisonDQN.py
├── IndividualSensorComparisonDuelDQN.py
├── IndividualSensorComparisonDDQN.py
└── IndividualSensorComparisonIDDQN.py
```


In the following `individual` sensor represents only `LiDAR`, `Camera`, `IMU`. And `dual` sensors represents `LiDAR + IMU`, `Camera + IMU`, `LiDAR + Camera`, `LiDAR + Camera + IMU`.

## Analytical Results
Episodic result of `LiDAR + Camera + IMU` with varying reinforcement learning algorithm.
<img width="1800" height="900" alt="sensorfusion_rl_episode_reward_moving_average" src="https://github.com/user-attachments/assets/e14a98a5-807c-46ec-9ea5-3baf2101f2dc" />
Episodic result of individual sensors with `IDDQN` algorithm
<img width="1800" height="900" alt="individual_sensor_reward_curves" src="https://github.com/user-attachments/assets/604bd5ba-3b06-4c3f-8f1a-e18640714119" />
Episodic result of dual sensors with `IDDQN` algorithm
<img width="1800" height="900" alt="dual_sensor_reward_curves_IDDQN" src="https://github.com/user-attachments/assets/9bf0bbf6-f3ec-4da9-a7ed-2e101375be71" />

**Made with ☕️ by Abhinandan**

