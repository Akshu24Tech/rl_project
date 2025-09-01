import gymnasium as gym
import torch
import numpy as np
from networks import Actor, Critic

STATE_SIZE = 24
ACTION_SIZE = 4

if __name__ == "__main__":
    actor = Actor(STATE_SIZE, ACTION_SIZE)
    critic = Critic(STATE_SIZE)
    try:
        actor.load_state_dict(torch.load("models/ppo_bipedal_walker_actor.pth"))
        critic.load_state_dict(torch.load("models/ppo_bipedal_walker_critic.pth"))
        actor.eval()
        critic.eval()
        print("PPO models loaded successfully.")
    except FileNotFoundError:
        print("Error: Models not found. Please run main.py first.")
        exit()
        
    env = gym.make("BipedalWalker-v3", render_mode="human")
    
    num_eval_episodes = 5
    total_rewards = []
    
    print("Starting PPO Agent Evaluation on BipedalWalker-v3...")
    
    for episode in range(num_eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
    
        while not terminated and not truncated:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                dist = actor(state_tensor)
                action = dist.mean   
                
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy().flatten())
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Final Reward = {episode_reward:.2f}")
        
    print("\n--- Evaluation Complete ---")
    print(f"Average Reward over {num_eval_episodes} episodes: {np.mean(total_rewards):.2f}")
    
    env.close()    