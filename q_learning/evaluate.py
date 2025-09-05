import gymnasium as gym
import numpy as np


NUM_BINS = (10, 10, 10, 10)
STATE_BOUNDS = list(zip(
    [-2.4, -3.0, -0.2095, -3.0], 
    [2.4, 3.0, 0.2095, 3.0]
))

def discretize_state(state):
    """
    Converts a continuous state from the environment into a discrete bin index.
    This function must be identical to the one used in main.py.
    """
    state_bin = []
    for i in range(len(state)):
        bin_size = (STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]) / NUM_BINS[i]
        bin_index = int((state[i] - STATE_BOUNDS[i][0]) / bin_size)
        state_bin.append(max(0, min(bin_index, NUM_BINS[i] - 1)))
    return tuple(state_bin)

try:
    q_table = np.load("models/q_table.npy")
    print("Q-Table loaded successfully. Shape:", q_table.shape)
except FileNotFoundError:
    print("Error: The 'q_table.npy' file was not found. Please run main.py first to train the agent.")
    exit()


env = gym.make("CartPole-v1", render_mode="human")

num_eval_episodes = 10
total_rewards = []

for episode in range(num_eval_episodes):
    state, info = env.reset()
    state_bin = discretize_state(state)
    terminated = False
    truncated = False
    episode_reward = 0

    while not terminated and not truncated:
        
        action = np.argmax(q_table[state_bin])

        next_state, reward, terminated, truncated, info = env.step(action)
        state_bin = discretize_state(next_state)
        episode_reward += reward

    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Final Reward = {episode_reward}")

print("\n--- Evaluation Complete ---")
print(f"Average Reward over {num_eval_episodes} episodes: {np.mean(total_rewards):.2f}")
env.close()