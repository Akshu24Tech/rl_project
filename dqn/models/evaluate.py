import gymnasium as gym
import torch
import numpy as np
from dqn_model import DQN

# STATE_SIZE = 4
# ACTION_SIZE = 2

STATE_SIZE = 8
ACTION_SIZE = 4

if __name__ == "__main__":
    agent_model = DQN(STATE_SIZE, ACTION_SIZE)
    try:
        # agent_model.load_state_dict(torch.load("models/dqn_model.pth"))
        agent_model.load_state_dict(torch.load("models/dqn_lunar_lander.pth"))
        agent_model.eval()  # Set the model to evaluation mode
        # print("DQN model loaded successfully.")
        print("LunarLander DQN model loaded successfully.")
        
    except FileNotFoundError:
        # print("Error: The 'dqn_model.pth' file was not found. Please run main.py first to train the agent.")
        print("Error: The 'dqn_lunar_lander.pth' file was not found. Run main.py first.")
        exit()
    
    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("LunarLander-v3", render_mode="human")
    num_eval_episodes = 10
    total_rewards = []
    
    # print("Starting DQN Agent Evaluation...")
    print("Starting DQN Agent Evaluation on LunarLander-v3...")
    
    for episode in range(num_eval_episodes):
        state, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
    
        while not terminated and not truncated:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_values = agent_model(state_tensor)
            
            action = np.argmax(action_values.cpu().data.numpy())
            
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Final Reward = {episode_reward}")    
            
    print("\n--- Evaluation Complete ---")
    print(f"Average Reward over {num_eval_episodes} episodes: {np.mean(total_rewards):.2f}")
    
    env.close()
    