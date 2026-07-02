# Attention-Based Individual Sensor Fusion for Energy-Aware UAV Navigation using A-IDDQN

## 1. Overview

This report formalizes the implemented attention-based individual sensor-fusion UAV navigation framework in research-paper style. The method models UAV navigation as a finite-horizon Markov decision process (MDP) on a two-dimensional grid map with obstacle avoidance, energy-aware movement, and modality-dependent sensing. The proposed attention-based improved double deep Q-network, denoted A-IDDQN, extends IDDQN with a goal-conditioned cross-attention module that learns how much to rely on LiDAR, camera, and IMU modality tokens at each decision step.

The implementation evaluates four active sensor settings:

1. LiDAR only: `("lidar",)`
2. Camera only: `("camera",)`
3. IMU only: `("imu",)`
4. Full fusion: `("lidar", "camera", "imu")`

The UAV starts from `(5, 1)` and navigates toward target `(14, 29)` in a grid of size `30 x 30`, with a maximum episode length of `160` steps.

## 2. MDP Formulation

The navigation task is formulated as an episodic MDP:

```text
M = (S, A, P, R, gamma)
```

where `S` is the state space, `A` is the discrete action space, `P` is the transition function induced by UAV movement and collision rules, `R` is the shaped reward function, and `gamma = 0.90` is the discount factor.

At time step `t`, the UAV observes state `s_t`, selects action `a_t`, moves to `s_{t+1}`, receives reward `r_t`, and stores transition:

```text
(s_t, a_t, r_t, s_{t+1}, done_t)
```

in a replay buffer.

## 3. Environment Geometry

Let the grid be:

```text
G = {(x, y) | 0 <= x <= N, 0 <= y <= N},    N = 30.
```

The UAV position at time `t` is:

```text
p_t = (x_t, y_t).
```

The start and target positions are:

```text
p_start = (5, 1),
p_goal  = (14, 29).
```

Let `O` denote the set of obstacle cells. The Euclidean distance from the UAV to the target is:

```text
d_g(p_t) = ||p_t - p_goal||_2.
```

The distance to the nearest obstacle is:

```text
d_o(p_t) = min_{o in O} ||p_t - o||_2.
```

If no obstacle exists, the maximum possible grid distance is used:

```text
d_max = sqrt(2) N.
```

The UAV is considered to have reached the target when:

```text
d_g(p_t) <= sqrt(2) / 2.
```

A collision is detected when either the integer-valued UAV cell belongs to the obstacle set or the UAV is inside the safety radius:

```text
collision_t = 1[(floor_cell(p_t) in O) or d_o(p_t) < r_safe],
```

where:

```text
r_safe = 0.60.
```

## 4. Action Space

The UAV uses an eight-direction discrete action space:

```text
A = {
  up, down, left, right,
  upper_right, lower_right, upper_left, lower_left
}.
```

The corresponding motion vectors are:

```text
a_0 = ( 0,  1)   up
a_1 = ( 0, -1)   down
a_2 = (-1,  0)   left
a_3 = ( 1,  0)   right
a_4 = ( 1,  1)   upper_right
a_5 = ( 1, -1)   lower_right
a_6 = (-1,  1)   upper_left
a_7 = (-1, -1)   lower_left
```

The proposed next position is:

```text
p'_t = p_t + a_t.
```

The environment then evaluates boundary violation, obstacle collision, target reaching, energy consumption, and terminal conditions.

## 5. Sensor Modeling

The implementation models three sensing modalities: LiDAR, camera, and IMU. The individual sensor-fusion setting keeps a fixed state width for fair comparison. Inactive sensors produce zero-valued feature blocks, and a sensor mask explicitly informs the network which modalities are available.

### 5.1 LiDAR Model

LiDAR senses obstacles in four cardinal directions:

```text
D_L = {(0,1), (0,-1), (-1,0), (1,0)}.
```

For each direction, the LiDAR checks cells at distances `k = 1, 2, 3`. Therefore, the LiDAR feature vector has:

```text
4 directions x 3 ranges = 12 features.
```

For direction `d_j` and range `k`, the LiDAR binary reading is:

```text
l_{j,k}(p_t) =
  1, if p_t + k d_j is blocked,
  0, otherwise.
```

A cell is blocked if it is outside the grid or belongs to the obstacle set:

```text
blocked(c) = 1[(c notin G) or (c in O)].
```

Thus:

```text
l_t = [l_{1,1}, l_{1,2}, l_{1,3}, ..., l_{4,3}] in R^12.
```

The LiDAR is used in two places:

1. State perception: local obstacle readings.
2. Energy cost: additional cost when nearby obstacles are detected in the first LiDAR range.

The number of immediate nearby obstacles is:

```text
n_obs(t) = sum_{j=1}^{4} l_{j,1}(p_t).
```

### 5.2 Camera Model

The camera produces an action-conditioned collision-warning vector of length `8`, one flag for each possible action. For action `a_i = (dx_i, dy_i)`, the proposed next cell is:

```text
c_i = p_t + a_i.
```

The camera flag is:

```text
c_i^{cam}(p_t) = 1[blocked(c_i)].
```

For diagonal motion, the camera also checks whether either adjacent axial cell is blocked:

```text
if dx_i != 0 and dy_i != 0:
  c_i^{cam}(p_t) =
    1[blocked(p_t + (dx_i, dy_i))
      or blocked(p_t + (dx_i, 0))
      or blocked(p_t + (0, dy_i))].
```

The complete camera vector is:

```text
c_t^{cam} = [c_0^{cam}, c_1^{cam}, ..., c_7^{cam}] in R^8.
```

The selected action's camera flag is:

```text
b_t^{cam} = c_{a_t}^{cam}.
```

If the camera is active and `b_t^{cam} = 1`, an additional energy cost is imposed.

### 5.3 IMU Model

The IMU describes recent motion dynamics. It contains:

```text
m_t = [dx_{t-1}, dy_{t-1}, straight_t, corner_t] in R^4.
```

where `(dx_{t-1}, dy_{t-1})` is the previous motion vector. The motion type is:

```text
straight_t = 1, if a_t = a_{t-1},
corner_t   = 1, if a_t != a_{t-1}.
```

At the first action of an episode, the motion is treated as straight. If IMU is active and the UAV changes direction, a cornering energy cost is added.

## 6. State Representation

The base navigation state has five normalized features:

```text
b_t = [
  x_t / N,
  y_t / N,
  d_g(p_t) / d_max,
  min(d_o(p_t), d_max) / d_max,
  E_t / E_0
] in R^5.
```

where:

```text
d_max = sqrt(2) N,
E_0 = 100.
```

For individual sensor comparison, the state is:

```text
s_t = [b_t, l_t, c_t^{cam}, m_t, z_t],
```

where:

```text
b_t        in R^5
l_t        in R^12
c_t^{cam}  in R^8
m_t        in R^4
z_t        in R^3
```

and `z_t` is the active sensor mask:

```text
z_t = [z_L, z_C, z_I],
```

with:

```text
z_L = 1 if LiDAR is active, else 0,
z_C = 1 if camera is active, else 0,
z_I = 1 if IMU is active, else 0.
```

The total state dimension is:

```text
dim(s_t) = 5 + 12 + 8 + 4 + 3 = 32.
```

Inactive modalities are represented by zero vectors:

```text
l_t = 0 if LiDAR is inactive,
c_t^{cam} = 0 if camera is inactive,
m_t = 0 if IMU is inactive.
```

This design enables fair comparison across individual sensor modes while preserving a fixed input size for A-IDDQN.

## 7. Energy Utilization Model

The UAV starts each episode with:

```text
E_0 = 100.
```

At each step, the energy cost is a sum of base motion cost, diagonal motion cost, turning cost, obstacle-risk cost, and camera-warning cost.

For individual sensor mode, the energy cost is:

```text
C_t =
  c_base
  + 1[diagonal(a_t)] c_diag
  + z_I 1[corner(a_t)] c_corner
  + z_L c_obs n_obs(t)
  + z_C 1[b_t^{cam}=1] c_cam.
```

The constants used in the implementation are:

```text
c_base   = 0.70
c_diag   = 0.30
c_corner = 0.45
c_obs    = 0.25
c_cam    = 0.35
```

A diagonal action is any action where both coordinate changes are nonzero:

```text
diagonal(a_t) = 1[|dx_t| + |dy_t| = 2].
```

A corner action occurs when the current action differs from the previous action:

```text
corner(a_t) = 1[a_t != a_{t-1}].
```

The energy update before target bonus is:

```text
E'_t = max(0, E_{t-1} - C_t).
```

If the UAV reaches the target, it receives a goal-energy bonus:

```text
E_t = min(E_0, E'_t + E_bonus), if reached_t = 1,
E_t = E'_t, otherwise.
```

where:

```text
E_bonus = 25.
```

The energy change used in the reward is:

```text
Delta E_t = E_t - E_{t-1}.
```

Since energy normally decreases, `Delta E_t` is usually negative. When the UAV reaches the goal, the energy bonus can make `Delta E_t` less negative or positive, encouraging efficient goal-reaching.

Energy depletion occurs when:

```text
E_t <= 0 and reached_t = 0.
```

## 8. Reward Function

The reward function combines step penalty, target-distance shaping, progress reward, energy utilization, and terminal events.

Let:

```text
d_{t-1} = d_g(p_{t-1}),
d_t     = d_g(p_t).
```

The progress term is:

```text
Delta d_t = d_{t-1} - d_t.
```

If the UAV moves closer to the target, `Delta d_t > 0`; if it moves away, `Delta d_t < 0`.

The complete reward is:

```text
r_t =
  r_step
  + alpha_d d_t
  + alpha_p (d_{t-1} - d_t)
  + alpha_E Delta E_t
  + 1[reached_t] R_goal
  + 1[collision_t] R_collision
  + 1[out_t] R_boundary
  + 1[energy_depleted_t] R_depleted.
```

The implemented constants are:

```text
r_step      = -0.10
alpha_d     = -0.02
alpha_p     =  3.00
alpha_E     =  0.60
R_goal      =  500.00
R_collision = -300.00
R_boundary  = -300.00
R_depleted  = -250.00
```

Interpretation:

1. `r_step` discourages unnecessarily long trajectories.
2. `alpha_d d_t` penalizes being far from the target.
3. `alpha_p(d_{t-1} - d_t)` rewards progress toward the target.
4. `alpha_E Delta E_t` penalizes energy consumption and rewards energy recovery at the goal.
5. `R_goal` strongly rewards successful target reaching.
6. `R_collision`, `R_boundary`, and `R_depleted` impose terminal safety penalties.

The episode terminates if any of the following conditions is true:

```text
done_t =
  reached_t
  or collision_t
  or out_t
  or energy_depleted_t
  or steps_t >= T_max.
```

where:

```text
T_max = 160.
```

## 9. Evaluation Metrics

After training, a greedy rollout is performed with exploration disabled. The following metrics are computed.

### 9.1 Total Return

```text
G = sum_{t=1}^{T} r_t.
```

### 9.2 Number of Steps

```text
steps = T.
```

### 9.3 Path Length

For a path `P = [p_0, p_1, ..., p_T]`, the path length is:

```text
L(P) = sum_{t=1}^{T} ||p_t - p_{t-1}||_2.
```

### 9.4 Number of Corners

Let:

```text
u_t = p_t - p_{t-1}.
```

The number of corners is:

```text
K(P) = sum_{t=2}^{T} 1[u_t != u_{t-1}].
```

### 9.5 Final Energy and Energy Used

```text
E_final = E_T,
E_used = E_0 - E_T.
```

### 9.6 Success, Collision, Boundary, and Energy-Depletion Flags

The evaluation also records:

```text
success = reached_T,
collision = collision_T,
out_of_bounds = out_T,
energy_depleted = energy_depleted_T.
```

## 10. IDDQN Learning Formulation

The baseline IDDQN uses a Q-network and a target Q-network:

```text
Q(s,a; theta),
Q^-(s,a; theta^-).
```

For each sampled replay transition `(s_t, a_t, r_t, s_{t+1}, done_t)`, the online network selects the next action:

```text
a^* = argmax_{a'} Q(s_{t+1}, a'; theta).
```

The target network evaluates this selected action:

```text
y_t = r_t + gamma (1 - done_t) Q^-(s_{t+1}, a^*; theta^-).
```

The temporal-difference loss is:

```text
L_TD(theta) =
  mean[(Q(s_t, a_t; theta) - y_t)^2].
```

For IDDQN and A-IDDQN, the target network is updated softly:

```text
theta^- <- tau theta + (1 - tau) theta^-,
```

with:

```text
tau = 0.02.
```

The exploration policy is epsilon-greedy:

```text
pi(a|s) =
  random action, with probability epsilon(e),
  argmax_a Q(s,a;theta), otherwise.
```

For IDDQN/A-IDDQN, the episode-dependent epsilon is:

```text
epsilon(e) =
  epsilon_end
  + (epsilon_start - epsilon_end) / (1 + exp(e / epsilon_decay)).
```

where:

```text
epsilon_start = 0.90,
epsilon_end   = 0.02,
epsilon_decay = 450.
```

## 11. Dueling Q-Network Decomposition

The dueling architecture decomposes Q-values into a state-value stream and an action-advantage stream:

```text
V(s; theta_V),
A(s,a; theta_A).
```

The final Q-value is:

```text
Q(s,a;theta) =
  V(s;theta_V)
  + A(s,a;theta_A)
  - (1 / |A|) sum_{a'} A(s,a';theta_A).
```

This decomposition allows the network to estimate the value of a state separately from action-specific advantages, which is useful in navigation because many states may be valuable even when several actions have similar quality.

## 12. Proposed A-IDDQN: Goal-Conditioned Sensor Attention

A-IDDQN extends IDDQN by replacing the standard dueling network input processing with a goal-conditioned cross-attention module over modality tokens.

The state is partitioned as:

```text
s_t = [b_t, l_t, c_t^{cam}, m_t, z_t],
```

where:

```text
b_t        in R^5
l_t        in R^12
c_t^{cam}  in R^8
m_t        in R^4
z_t        in R^3.
```

Each component is embedded into a common dimension `d = 64`:

```text
h_b = ReLU(W_b b_t + q_b)              in R^d,
h_L = ReLU(W_L l_t + q_L)              in R^d,
h_C = ReLU(W_C c_t^{cam} + q_C)        in R^d,
h_I = ReLU(W_I m_t + q_I)              in R^d.
```

The sensor tokens are:

```text
H_S = [h_L + e_L, h_C + e_C, h_I + e_I] in R^{3 x d},
```

where `e_L`, `e_C`, and `e_I` are learnable token-type embeddings.

Layer normalization is applied:

```text
q = LayerNorm(h_b)      in R^d,
K = LayerNorm(H_S)      in R^{3 x d},
V = LayerNorm(H_S)      in R^{3 x d}.
```

The base navigation embedding acts as the query. The sensor modality embeddings act as keys and values. Therefore, attention is conditioned on the UAV's current navigation context, target distance, obstacle distance, and remaining energy.

For one attention head, the scaled dot-product attention is:

```text
Attention(q,K,V) =
  softmax((q K^T) / sqrt(d_h) + M) V,
```

where `M` is a mask derived from the active-sensor vector `z_t`. If a sensor is inactive, its corresponding key is masked and cannot receive attention probability.

For multi-head attention with `H = 4` heads:

```text
head_i = Attention(q W_i^Q, K W_i^K, V W_i^V),
MultiHead(q,K,V) = Concat(head_1, ..., head_H) W^O.
```

The attention weights are:

```text
alpha_t = [alpha_L, alpha_C, alpha_I],
```

where:

```text
alpha_L + alpha_C + alpha_I = 1
```

over active sensors. Inactive sensors are masked. These weights provide an interpretable estimate of the relative contribution of LiDAR, camera, and IMU for the current navigation decision.

The attended representation is processed using residual and feed-forward layers:

```text
u_t = LayerNorm(q + MultiHead(q,K,V)),
v_t = LayerNorm(u_t + FFN(u_t)).
```

The final fused representation concatenates the original base embedding and attended sensor context:

```text
f_t = Fusion([h_b, v_t]).
```

The dueling head then computes:

```text
z_t^{shared} = ReLU(W_s f_t + b_s),
V(s_t) = f_V(z_t^{shared}),
A(s_t,a) = f_A(z_t^{shared}),
```

and:

```text
Q(s_t,a) =
  V(s_t)
  + A(s_t,a)
  - mean_{a'} A(s_t,a').
```

## 13. Attention Entropy Regularization

The implementation includes an attention entropy term for A-IDDQN. The purpose is to encourage the network to make sharper sensor-selection decisions instead of assigning nearly uniform attention to every modality.

For attention vector:

```text
alpha_t = [alpha_L, alpha_C, alpha_I],
```

the entropy is:

```text
H(alpha_t) = - sum_{j in {L,C,I}} alpha_j log(alpha_j).
```

For a minibatch of size `B`:

```text
H_batch = (1/B) sum_{i=1}^{B} H(alpha_i).
```

The A-IDDQN objective is:

```text
L_A-IDDQN(theta) =
  L_TD(theta) + lambda_H H_batch,
```

when the entropy target is set to `"low"`. The implementation uses:

```text
lambda_H = 0.01.
```

Minimizing this term encourages low-entropy, sharper modality attention. If configured for high-entropy attention, the sign can be reversed:

```text
L_A-IDDQN(theta) =
  L_TD(theta) - lambda_H H_batch.
```

In the current notebook, the low-entropy option is used.

## 14. A* Expert Warm Start

The training procedure optionally uses an A* expert trajectory before reinforcement learning. A* computes a reference path from start to target using Euclidean heuristic:

```text
h(c) = ||c - p_goal||_2.
```

The path search cost for a move `(dx,dy)` is:

```text
cost(dx,dy) = sqrt(dx^2 + dy^2).
```

The A* priority is:

```text
f(c) = g(c) + h(c),
```

where `g(c)` is the accumulated path cost from the start.

The expert path is used in two ways:

1. Replay prefill: expert transitions are inserted into the replay buffer repeatedly.
2. Behavior cloning: the network is trained with cross-entropy to imitate the expert action sequence.

For behavior cloning, if expert state-action pairs are `{(s_i, a_i^*)}`, then:

```text
L_BC(theta) =
  - sum_i log softmax(Q(s_i, .; theta))_{a_i^*}.
```

This warm start helps stabilize early learning by giving the network an initial feasible navigation policy.

## 15. A-IDDQN Training Algorithm

The following pseudocode summarizes the implemented method.

```text
Algorithm: A-IDDQN for Attention-Based Individual Sensor Fusion

Input:
  Grid G, obstacle set O, start p_start, target p_goal
  active sensor set Z in {LiDAR, Camera, IMU}
  replay buffer D, online parameters theta, target parameters theta^-

Initialize:
  theta randomly
  theta^- <- theta
  E_0 <- 100
  optionally compute A* expert path and prefill replay buffer
  optionally behavior-clone expert actions

For episode e = 1 to E:
  Reset UAV position p_0 <- p_start
  Reset energy E_0 <- 100
  Construct initial state s_0

  For t = 0 to T_max - 1:
    With probability epsilon(e), choose random action a_t
    Otherwise:
      Split s_t into base, LiDAR, camera, IMU, mask
      Encode base as query
      Encode sensors as key/value modality tokens
      Mask inactive sensors
      Compute goal-conditioned cross-attention weights alpha_t
      Fuse attended sensor context with base navigation embedding
      Compute dueling Q-values
      Select a_t = argmax_a Q(s_t,a;theta)

    Execute a_t:
      p'_t <- p_t + a_t
      compute LiDAR obstacle flags
      compute camera blocked flag
      compute IMU motion type
      compute energy cost C_t
      update energy E_t
      compute reward r_t
      determine done_t

    Store (s_t, a_t, r_t, s_{t+1}, done_t) in D

    If |D| >= batch_size:
      Sample minibatch from D
      a^* <- argmax_a Q(s_{t+1},a;theta)
      y <- r + gamma (1-done) Q^-(s_{t+1},a^*;theta^-)
      L_TD <- mean squared TD error
      H <- mean entropy of attention weights
      L <- L_TD + lambda_H H
      Update theta using Adam
      Clip gradients
      Soft update theta^- <- tau theta + (1-tau) theta^-

    If done_t:
      break

Output:
  Trained A-IDDQN policy and evaluation metrics.
```

## 16. Research Contribution Framing

The contribution can be stated as follows:

1. An energy-aware UAV grid-navigation MDP is designed with distance shaping, progress reward, terminal safety penalties, and sensor-dependent energy consumption.
2. A fixed-width individual sensor-fusion state representation is introduced, enabling direct comparison between LiDAR-only, camera-only, IMU-only, and full-fusion settings.
3. A goal-conditioned cross-attention dueling Q-network is proposed for A-IDDQN, where the base navigation state queries sensor modality tokens.
4. Active-sensor masking allows the same architecture to operate across individual and fused sensor settings.
5. Attention entropy regularization encourages sharper and more interpretable modality reliance.
6. A* warm-start and behavior cloning are used to improve sample efficiency and stabilize early policy learning.

## 17. Suggested Paper Wording for Method Section

The proposed A-IDDQN framework formulates UAV navigation as an energy-aware MDP in which the agent must reach a target while avoiding static obstacles and preserving battery energy. At each time step, the state consists of normalized position, target distance, nearest-obstacle distance, remaining energy, LiDAR obstacle scans, camera action-risk flags, IMU motion descriptors, and an active-sensor mask. The reward function jointly optimizes target progress, path efficiency, energy usage, and safety. Unlike a conventional DQN that directly maps the concatenated state vector to Q-values, the proposed A-IDDQN decomposes the state into semantic modality groups and applies goal-conditioned cross-attention. The base navigation features act as a query, while LiDAR, camera, and IMU embeddings act as key-value tokens. This allows the agent to adaptively emphasize the most relevant sensor modality according to the current navigation context. The attended sensor context is fused with the base navigation embedding and passed through dueling value and advantage streams. The network is trained using a Double-DQN target with soft target updates, replay memory, epsilon-greedy exploration, and an entropy regularizer over attention weights to encourage discriminative modality selection.

## 18. Important Implementation Notes for Publication

1. The state dimension in the individual-sensor experiment is fixed at `32`, even when only one sensor is active.
2. Inactive sensor features are zero-filled and also indicated through a three-dimensional mask.
3. LiDAR contributes obstacle-range perception and obstacle-risk energy cost.
4. Camera contributes action-wise blocked-move warnings and camera-warning energy cost.
5. IMU contributes motion-history features and cornering energy cost.
6. The target network is updated softly for IDDQN and A-IDDQN using `tau = 0.02`.
7. A-IDDQN uses `embed_dim = 64`, `num_heads = 4`, and attention dropout `0.0`.
8. Attention weights are logged during greedy rollout and can be plotted to support interpretability claims.
9. The reward is not only goal-based; it is dense and shaped by progress, distance, and energy.
10. The final reported performance should include reward, steps, path length, corners, success, final energy, energy used, inference time, collision, out-of-bounds, and energy depletion.

## 19. Compact Equation Set for Paper

State:

```text
s_t = [b_t, l_t, c_t^{cam}, m_t, z_t]
```

Energy cost:

```text
C_t =
  c_base
  + 1[diagonal(a_t)] c_diag
  + z_I 1[corner(a_t)] c_corner
  + z_L c_obs n_obs(t)
  + z_C 1[b_t^{cam}=1] c_cam
```

Energy update:

```text
E'_t = max(0, E_{t-1} - C_t)
```

```text
E_t =
  min(E_0, E'_t + E_bonus), if reached_t = 1
  E'_t, otherwise
```

Reward:

```text
r_t =
  r_step
  + alpha_d d_t
  + alpha_p (d_{t-1} - d_t)
  + alpha_E (E_t - E_{t-1})
  + 1[reached_t] R_goal
  + 1[collision_t] R_collision
  + 1[out_t] R_boundary
  + 1[energy_depleted_t] R_depleted
```

Double-DQN target:

```text
a^* = argmax_a Q(s_{t+1},a;theta)
```

```text
y_t = r_t + gamma(1-done_t)Q^-(s_{t+1},a^*;theta^-)
```

Dueling Q-value:

```text
Q(s,a) = V(s) + A(s,a) - mean_{a'} A(s,a')
```

Cross-attention:

```text
alpha_t = softmax((q_t K_t^T) / sqrt(d_h) + M_t)
```

```text
o_t = alpha_t V_t
```

A-IDDQN loss:

```text
L(theta) =
  mean[(Q(s_t,a_t;theta)-y_t)^2]
  + lambda_H H(alpha_t)
```

Attention entropy:

```text
H(alpha_t) = - sum_j alpha_{t,j} log(alpha_{t,j})
```

