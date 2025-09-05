# Commented which is for CartPole Project
import gymnasium as gym
import torch
from dqn_agent import DQNAgent
import numpy as np

# for CartPole
# STATE_SIZE = 4
# ACTION_SIZE = 2 

# for LunarLander
STATE_SIZE = 8
ACTION_SIZE = 4

BUFFER_CAPACITY = 100000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.0005
TAU = 0.005
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995


if __name__ == "__main__":
#   env = gym.make("CartPole-v1")
    env = gym.make("LunarLander-v3")
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE, BUFFER_CAPACITY, BATCH_SIZE, GAMMA, LR, TAU)

    epsilon = EPSILON_START
#   num_episodes = 500
    num_episodes = 2000

#   print("Starting DQN Training...")
    print("Starting DQN Training on LunarLander-v3...")

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = agent.select_action(state, epsilon)

            next_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, done)
            
            agent.learn()
            
            state = next_state
            episode_reward += reward

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % 50 == 0:
            print(f"Episode: {episode}, Epsilon: {epsilon:.4f}, Reward: {episode_reward}")

    print("DQN Training finished.")
#   agent.save_model("models/dqn_model.pth")
    agent.save_model("models/dqn_lunar_lander.pth")
    env.close()