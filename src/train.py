# src/train.py

import gymnasium as gym
from src.agents.q_learning_agent import QLearningAgent
import numpy as np

# --- 1. Setup ---
env = gym.make('CartPole-v1')

# For Q-learning with CartPole, we need to discretize the state space.
# We'll create bins for the four continuous state variables.
# This is a key part of adapting Q-learning to a continuous environment.
pos_bins = np.linspace(-2.4, 2.4, 10)
vel_bins = np.linspace(-4, 4, 10)
angle_bins = np.linspace(-0.2095, 0.2095, 10)
angular_vel_bins = np.linspace(-4, 4, 10)
# A helper function to convert continuous state to a discrete tuple
def get_state_index(state):
    pos_idx = np.digitize(state[0], pos_bins)
    vel_idx = np.digitize(state[1], vel_bins)
    angle_idx = np.digitize(state[2], angle_bins)
    angular_vel_idx = np.digitize(state[3], angular_vel_bins)
    return (pos_idx, vel_idx, angle_idx, angular_vel_idx)

# Total discrete states = 10*10*10*10 = 10,000
# The Q-table will have dimensions (10, 10, 10, 10, 2)
# The `2` is for the two possible actions (left/right)
state_space_size = (len(pos_bins) + 1, len(vel_bins) + 1, len(angle_bins) + 1, len(angular_vel_bins) + 1)
action_space_size = env.action_space.n

# --- 2. Instantiate Agent ---
agent = QLearningAgent(
    observation_space_size=state_space_size, 
    action_space_size=action_space_size,
    learning_rate=0.1, 
    gamma=0.99, 
    epsilon=1.0  # Start with high exploration
)

# --- 3. Training Loop ---
num_episodes = 2000
for episode in range(num_episodes):
    state, info = env.reset()
    state_idx = get_state_index(state)
    terminated = False
    truncated = False
    total_reward = 0
    
    while not terminated and not truncated:
        action = agent.choose_action(state_idx)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state_idx = get_state_index(next_state)
        
        agent.learn(state_idx, action, reward, next_state_idx)
        
        state_idx = next_state_idx
        total_reward += reward

    print(f"Episode {episode}: Total Reward = {total_reward}")

    # Gradually decrease epsilon to reduce exploration over time
    agent.epsilon = max(0.01, agent.epsilon * 0.995)

env.close()