import gymnasium as gym
import torch
import numpy as np
from ppo_agent import PPOAgent

hyperparameters = {
    'LR_ACTOR': [3e-4, 5e-4],
    'LR_CRITIC': [1e-3, 5e-3],
    'CLIP_EPSILON': [0.1, 0.2],
    'PPO_EPOCHS': [5, 10]
}

def run_training_session(params):
    STATE_SIZE = 24
    ACTION_SIZE = 4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    BATCH_SIZE = 64
    ROLLOUT_SIZE = 2048

    env = gym.make("BipedalWalker-v3")
    agent = PPOAgent(
        STATE_SIZE, ACTION_SIZE, params['LR_ACTOR'], params['LR_CRITIC'], GAMMA,
        GAE_LAMBDA, params['CLIP_EPSILON'], params['PPO_EPOCHS'], BATCH_SIZE
    )

    total_reward = 0
    num_episodes = 0
    max_episodes = 200
    
    print(f"\n--- Starting Training with Hyperparameters: {params} ---")

    while num_episodes < max_episodes:
        # (Your existing training loop code goes here)
        state, _ = env.reset()
        done = False
        step = 0
        episode_reward = 0

        while not done and step < ROLLOUT_SIZE:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.rewards.append(reward)
            agent.dones.append(done)
            
            state = next_state
            episode_reward += reward
            step += 1
            
        agent.learn(state)

        total_reward += episode_reward
        num_episodes += 1

        if num_episodes % 20 == 0:
            avg_reward = total_reward / 20
            print(f"Episode: {num_episodes}, Avg Reward: {avg_reward:.2f}")
            total_reward = 0
            
    env.close()
    return avg_reward # Return the final average reward for comparison
        
if __name__ == "__main__":
    from itertools import product
    
    keys = list(hyperparameters.keys())
    values = list(hyperparameters.values())
    
    best_avg_reward = -np.inf
    best_params = {}

    for param_values in product(*values):
        params = dict(zip(keys, param_values))
        avg_reward = run_training_session(params)
        
        # Track the best performing set of parameters
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_params = params
    
    print("\n--- Grid Search Complete ---")
    print(f"Best Parameters: {best_params}")
    print(f"Best Average Reward: {best_avg_reward:.2f}")      