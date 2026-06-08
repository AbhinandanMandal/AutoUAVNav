
from UAV_grid_env import UAVGridEnv
from IDDQNAgent import IDDQNAgent
from hyperparameters import EPISODES, TARGET_UPDATE, MAX_STEPS
import torch


if __name__ == "__main__":
    env = UAVGridEnv()
    agent = IDDQNAgent(env.state_size, env.action_size)

    for e in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            action = agent.act(state, e)
            next_state, reward, done = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

            if done:
                break

        if e % TARGET_UPDATE == 0:
            agent.update_target_network()

        print(
            f"Episode: {e+1}/{EPISODES} | Score: {total_reward:.2f} | Steps: {step}")

    MODEL_PATH = "uav_idqn_model.pth" # Trained Model
    torch.save(agent.current_net.state_dict(), MODEL_PATH)
    print(f"Training Complete. Model saved as '{MODEL_PATH}'")


