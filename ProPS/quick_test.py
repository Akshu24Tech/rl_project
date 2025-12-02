"""
Quick test with minimal episodes - perfect for testing setup
Runs only 10 episodes with 1 model to verify everything works
"""

import os
import time
import numpy as np
import gymnasium as gym
from collections import deque
from jinja2 import Template
from google import genai

# Minimal configuration for quick testing
CONFIG = {
    "NUM_EPISODES": 10,
    "WARMUP_EPISODES": 3,
    "NUM_EVALUATION_EPISODES": 2,
    "MAX_TRAJ_LENGTH": 200,
    "MAX_TRAJ_COUNT": 50,
    "SEARCH_STD": 1.0,
    "RENDER_MODE": None,
}

PROMPT_TEMPLATE = """
You are a global optimizer helping me find the maximum of function f(params).
params is an array of {{ rank }} float numbers in range [-6.0, 6.0] with 1 decimal place.

Provide response in this format:
Line 1: params[0]: X.X, params[1]: X.X
Line 2: Brief explanation

History:
{{ history }}

Iteration {{step}} of {{ max_steps }}. Propose new params to maximize f(params).
"""

class SimpleWorld:
    def __init__(self):
        self.env = gym.make("MountainCarContinuous-v0")
        self.steps = 0
        self.reward = 0
    
    def reset(self):
        state, _ = self.env.reset()
        self.steps = 0
        self.reward = 0
        return state
    
    def step(self, action):
        self.steps += 1
        state, r, term, trunc, _ = self.env.step([action[0]])
        self.reward += r
        done = self.steps >= CONFIG["MAX_TRAJ_LENGTH"] or term or trunc
        return state, r, done
    
    def get_reward(self):
        return self.reward

class SimplePolicy:
    def __init__(self):
        self.w = np.random.randn(2, 1)
    
    def act(self, state):
        return np.dot(state, self.w)
    
    def set_params(self, params):
        self.w = params.reshape(2, 1)
    
    def get_params(self):
        return self.w.flatten()

def evaluate(world, policy, n_episodes):
    rewards = []
    for _ in range(n_episodes):
        state = world.reset()
        done = False
        while not done:
            action = policy.act(state)
            state, _, done = world.step(action)
        rewards.append(world.get_reward())
    return np.mean(rewards)

def main():
    print("Quick ProPS Test - Minimal Configuration")
    print("="*50)
    
    # Get API key
    api_key = input("Enter Gemini API key: ").strip()
    if not api_key:
        print("Error: API key required")
        return
    
    # Choose model
    print("\nAvailable models:")
    print("1. gemini-3-deep-think-preview (deep thinking)")
    print("2. gemini-2.5-flash")
    print("3. gemini-2.0-flash")
    choice = input("Choose (1-3, default=1): ").strip() or "1"
    
    models = {
        "1": "gemini-3-deep-think-preview",
        "2": "gemini-2.5-flash",
        "3": "gemini-2.0-flash"
    }
    model_name = models.get(choice, "gemini-3-deep-think-preview")
    
    print(f"\nUsing: {model_name}")
    print(f"Episodes: {CONFIG['NUM_EPISODES']}")
    print(f"This should take ~2-5 minutes\n")
    
    # Initialize
    client = genai.Client(api_key=api_key)
    template = Template(PROMPT_TEMPLATE)
    world = SimpleWorld()
    policy = SimplePolicy()
    history = deque(maxlen=CONFIG["MAX_TRAJ_COUNT"])
    
    # Warmup
    print("Warmup...")
    for i in range(CONFIG["WARMUP_EPISODES"]):
        policy.set_params(np.round(np.random.normal(0, 3, 2), 1))
        reward = evaluate(world, policy, CONFIG["NUM_EVALUATION_EPISODES"])
        history.append((policy.get_params(), reward))
        print(f"  {i+1}: reward={reward:.2f}")
    
    # Training
    print("\nTraining...")
    rewards = []
    
    for ep in range(CONFIG["NUM_EPISODES"]):
        # Build history string
        hist_str = "\n".join([f"{p} -> {r:.2f}" for p, r in history])
        
        # Get LLM suggestion
        prompt = template.render(
            rank=2,
            history=hist_str,
            step=ep+1,
            max_steps=CONFIG["NUM_EPISODES"]
        )
        
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            
            # Parse params
            text = response.text.strip().split('\n')[0]
            params = []
            for part in text.split(','):
                if ':' in part:
                    params.append(float(part.split(':')[1].strip()))
            params = np.array(params)
            
        except Exception as e:
            print(f"  LLM error: {e}, using random")
            params = np.round(np.random.normal(0, 3, 2), 1)
        
        # Evaluate
        policy.set_params(params)
        reward = evaluate(world, policy, CONFIG["NUM_EVALUATION_EPISODES"])
        history.append((params, reward))
        rewards.append(reward)
        
        print(f"  {ep+1}: params={params}, reward={reward:.2f}")
    
    # Results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Final reward: {rewards[-1]:.2f}")
    print(f"Max reward: {max(rewards):.2f}")
    print(f"Avg reward: {np.mean(rewards):.2f}")
    print(f"\nReward progression: {[f'{r:.1f}' for r in rewards]}")
    
    # Simple plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, marker='o')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Quick Test - {model_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs("PrPoS/results", exist_ok=True)
        filename = f"PrPoS/results/quick_test_{model_name.replace('-', '_')}.png"
        plt.savefig(filename)
        print(f"\nPlot saved: {filename}")
        plt.show()
    except:
        print("\nCouldn't create plot (matplotlib issue)")
    
    print("\n✅ Test complete!")
    print("If this worked, you can run the full comparison:")
    print("  python props_gemini_comparison.py")

if __name__ == "__main__":
    main()
