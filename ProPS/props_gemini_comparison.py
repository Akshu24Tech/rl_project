"""
Prompted Policy Search (ProPS) - Gemini 3.0 Comparison
Simplified version for laptop execution with reduced episodes
"""

import os
import time
import numpy as np
import gymnasium as gym
from collections import deque
from jinja2 import Template
from google import genai
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION - Optimized for laptop performance
# ============================================================================

CONFIG = {
    "NUM_EPISODES": 50,  # Reduced from 200 for faster execution
    "WARMUP_EPISODES": 10,  # Reduced from 20
    "NUM_EVALUATION_EPISODES": 5,  # Reduced from 20
    "MAX_TRAJ_LENGTH": 500,  # Reduced from 1000
    "MAX_TRAJ_COUNT": 100,  # Reduced from 1000
    "SEARCH_STD": 1.0,
    "RENDER_MODE": None,
    "ENV_NAME": "MountainCarContinuous-v0",
    "GEMINI_MODELS": [
        "gemini-3-deep-think-preview",  # Deep thinking model (latest)
        "gemini-2.5-flash",             # For comparison
        "gemini-2.0-flash"              # For comparison
    ]
}

# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

LLM_PROMPT_TEMPLATE = """
You are a global optimizer, helping me find the global maximum of a mathematical function f(params).
I will give you the function evaluation and the current iteration number at each step.
Your goal is to propose input values that efficiently lead us to the global maximum within a limited number of iterations ({{ max_episodes }}).

# Regarding the parameters **params**:
**params** is an array of {{ rank }} float numbers.
**params** values are in the range of [-6.0, 6.0] with 1 decimal place.

# Here's how we'll interact:
1. I will first provide MAX_STEPS ({{ max_episodes }}) along with a few training examples.
2. You will provide your response in the following exact format:
    * Line 1: a new input 'params[0]: , params[1]: , params[2]: ,..., params[{{ rank - 1 }}]: ', aiming to maximize the function's value f(params).
    Please propose params values in the range of [-6.0, 6.0], with 1 decimal place.
    * Line 2: detailed explanation of why you chose that input.
3. I will then provide the function's value f(params) at that point, and the current iteration.
4. We will repeat steps 2-3 until we reach the maximum number of iterations.

# Remember:
1. **Do not propose previously seen params.**
2. **The global optimum should be around {{ optimum }}.** If you are below that, this is just a local optimum. You should explore instead of exploiting.
3. Search both positive and negative values. **During exploration, use search step size of {{ step_size }}**.

Next, you will see examples of params and f(params) pairs.
{{ episode_reward_buffer_string }}

Now you are at iteration {{step_number}} out of {{ max_episodes }}. Please provide the results in the indicated format. Do not provide any additional texts.
"""

# ============================================================================
# ENVIRONMENT WRAPPER
# ============================================================================

class MountainCarWorld:
    def __init__(self, render_mode=None, max_traj_length=1000):
        self.env = gym.make("MountainCarContinuous-v0", render_mode=render_mode)
        self.max_traj_length = max_traj_length
        self.steps = 0
        self.accu_reward = 0
    
    def reset(self):
        state, _ = self.env.reset()
        self.steps = 0
        self.accu_reward = 0
        return state
    
    def step(self, action):
        self.steps += 1
        action = action[0] if isinstance(action, (list, np.ndarray)) else action
        state, reward, terminated, truncated, _ = self.env.step([action])
        self.accu_reward += reward
        
        done = self.steps >= self.max_traj_length or terminated or truncated
        return state, reward, done
    
    def get_accu_reward(self):
        return self.accu_reward

# ============================================================================
# POLICY AND BUFFER
# ============================================================================

class LinearPolicy:
    def __init__(self, dim_states, dim_actions):
        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.weight = np.random.randn(dim_states, dim_actions)
    
    def initialize_policy(self):
        self.weight = np.round(np.random.normal(0., 3., size=(self.dim_states, self.dim_actions)), 1)
    
    def get_action(self, state):
        return np.matmul(state.T, self.weight)
    
    def set_weights(self, weights):
        self.weight = weights.reshape(self.dim_states, self.dim_actions)
    
    def get_weights(self):
        return self.weight.flatten()

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, weights, reward):
        self.buffer.append((weights, reward))
    
    def __str__(self):
        buffer_str = "Parameters | Reward\n"
        for weights, reward in self.buffer:
            buffer_str += f"{weights.reshape(1, -1)} | {reward:.2f}\n"
        return buffer_str

# ============================================================================
# LLM BRAIN
# ============================================================================

class LLMBrain:
    def __init__(self, api_key, model_name, template):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.template = Template(template)
    
    def get_new_parameters(self, replay_buffer, step_number, rank, max_episodes, optimum=100, step_size=1.0):
        prompt = self.template.render(
            rank=rank,
            optimum=optimum,
            step_size=step_size,
            episode_reward_buffer_string=str(replay_buffer),
            step_number=step_number,
            max_episodes=max_episodes
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # Parse response
            text = response.text.strip()
            lines = text.split('\n')
            
            # Extract parameters from first line
            param_line = lines[0]
            params = []
            for part in param_line.split(','):
                if ':' in part:
                    value = float(part.split(':')[1].strip())
                    params.append(value)
            
            return np.array(params), text
        
        except Exception as e:
            print(f"Error calling LLM: {e}")
            # Return random parameters as fallback
            return np.round(np.random.normal(0., 3., size=rank), 1), f"Error: {e}"

# ============================================================================
# AGENT
# ============================================================================

class ProPSAgent:
    def __init__(self, world, api_key, model_name, config):
        self.world = world
        self.config = config
        self.policy = LinearPolicy(dim_states=2, dim_actions=1)
        self.replay_buffer = ReplayBuffer(max_size=config["MAX_TRAJ_COUNT"])
        self.llm_brain = LLMBrain(api_key, model_name, LLM_PROMPT_TEMPLATE)
    
    def evaluate_policy(self, num_episodes):
        rewards = []
        for _ in range(num_episodes):
            state = self.world.reset()
            done = False
            while not done:
                action = self.policy.get_action(state)
                state, reward, done = self.world.step(action)
            rewards.append(self.world.get_accu_reward())
        return np.mean(rewards)
    
    def warmup(self):
        print(f"Running {self.config['WARMUP_EPISODES']} warmup episodes...")
        for i in range(self.config['WARMUP_EPISODES']):
            self.policy.initialize_policy()
            reward = self.evaluate_policy(self.config['NUM_EVALUATION_EPISODES'])
            self.replay_buffer.add(self.policy.get_weights(), reward)
            print(f"  Warmup {i+1}/{self.config['WARMUP_EPISODES']}: Reward = {reward:.2f}")
    
    def train(self):
        rewards_history = []
        
        for episode in range(self.config['NUM_EPISODES']):
            # Get new parameters from LLM
            new_params, reasoning = self.llm_brain.get_new_parameters(
                self.replay_buffer,
                step_number=episode + 1,
                rank=2,  # 2 parameters for MountainCar
                max_episodes=self.config['NUM_EPISODES'],
                optimum=100,
                step_size=self.config['SEARCH_STD']
            )
            
            # Update policy
            self.policy.set_weights(new_params)
            
            # Evaluate
            reward = self.evaluate_policy(self.config['NUM_EVALUATION_EPISODES'])
            self.replay_buffer.add(new_params, reward)
            rewards_history.append(reward)
            
            print(f"Episode {episode+1}/{self.config['NUM_EPISODES']}: Reward = {reward:.2f}, Params = {new_params}")
        
        return rewards_history

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(api_key, model_name, config):
    print(f"\n{'='*60}")
    print(f"Running experiment with {model_name}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Initialize
    world = MountainCarWorld(
        render_mode=config["RENDER_MODE"],
        max_traj_length=config["MAX_TRAJ_LENGTH"]
    )
    agent = ProPSAgent(world, api_key, model_name, config)
    
    # Warmup
    agent.warmup()
    
    # Train
    print(f"\nStarting training for {config['NUM_EPISODES']} episodes...")
    rewards_history = agent.train()
    
    elapsed_time = time.time() - start_time
    
    results = {
        "model": model_name,
        "rewards": rewards_history,
        "final_reward": rewards_history[-1] if rewards_history else 0,
        "max_reward": max(rewards_history) if rewards_history else 0,
        "avg_reward": np.mean(rewards_history) if rewards_history else 0,
        "time_seconds": elapsed_time
    }
    
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Final Reward: {results['final_reward']:.2f}")
    print(f"Max Reward: {results['max_reward']:.2f}")
    print(f"Avg Reward: {results['avg_reward']:.2f}")
    
    return results

def plot_comparison(all_results, save_path="PrPoS/results"):
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    for result in all_results:
        plt.plot(result["rewards"], label=result["model"], marker='o', markersize=3)
    
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("ProPS Performance Comparison: Gemini Models")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(save_path, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filename, dpi=150)
    print(f"\nPlot saved to: {filename}")
    plt.show()
    
    # Save results to JSON
    json_filename = os.path.join(save_path, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {json_filename}")

def main():
    print("ProPS - Gemini Model Comparison")
    print("=" * 60)
    
    # Get API key
    api_key = input("Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("Error: API key is required!")
        return
    
    # Run experiments for each model
    all_results = []
    
    for model_name in CONFIG["GEMINI_MODELS"]:
        try:
            results = run_experiment(api_key, model_name, CONFIG)
            all_results.append(results)
        except Exception as e:
            print(f"\nError with {model_name}: {e}")
            continue
    
    # Plot comparison
    if all_results:
        plot_comparison(all_results)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for result in all_results:
            print(f"\n{result['model']}:")
            print(f"  Final Reward: {result['final_reward']:.2f}")
            print(f"  Max Reward: {result['max_reward']:.2f}")
            print(f"  Avg Reward: {result['avg_reward']:.2f}")
            print(f"  Time: {result['time_seconds']:.2f}s")

if __name__ == "__main__":
    main()
