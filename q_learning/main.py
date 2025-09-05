import gymnasium as gym
import numpy as np
import math


LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995


NUM_BINS = (10, 10, 10, 10)
STATE_BOUNDS = list(zip(
    [-2.4, -3.0, -0.2095, -3.0], 
    [2.4, 3.0, 0.2095, 3.0]      
))


def discretize_state(state):
    """
    Converts a continuous state from the environment into a discrete bin index.
    """
    state_bin = []
    for i in range(len(state)):
        
        bin_size = (STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]) / NUM_BINS[i]
        
        bin_index = int(
            (state[i] - STATE_BOUNDS[i][0]) / bin_size
        )
        
        state_bin.append(
            max(0, min(bin_index, NUM_BINS[i] - 1))
        )
    return tuple(state_bin)


env = gym.make("CartPole-v1")


q_table = np.zeros(NUM_BINS + (env.action_space.n,))

epsilon = EPSILON_START
for episode in range(1000): 
    state, info = env.reset()
    state_bin = discretize_state(state)
    terminated = False
    truncated = False

    while not terminated and not truncated:
        # Epsilon-greedy policy for action selection
        if np.random.random() < epsilon:
            # Exploration: choose a random action
            action = env.action_space.sample()
        else:
            # Exploitation: choose the action with the highest Q-value
            action = np.argmax(q_table[state_bin])

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state_bin = discretize_state(next_state)

        current_q = q_table[state_bin + (action,)]
        best_future_q = np.max(q_table[next_state_bin])
        
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * best_future_q - current_q)
        q_table[state_bin + (action,)] = new_q

        state_bin = next_state_bin

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % 50 == 0:
        print(f"Episode: {episode}, Epsilon: {epsilon:.2f}")

print("Training finished.")

np.save("models/q_table.npy", q_table)