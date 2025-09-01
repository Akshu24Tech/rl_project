# File: ppo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks import Actor, Critic

class PPOAgent:
    def __init__(self, state_size, action_size, lr_actor, lr_critic, gamma, gae_lambda, clip_epsilon, ppo_epochs, batch_size):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor and Critic networks
        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Buffers for collecting rollouts
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist = self.actor(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        value = self.critic(state)

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        return action.cpu().numpy().flatten()
    
    def learn(self, next_state):
        # ... (The complete learn method code goes here)
        states = torch.cat(self.states).detach()
        actions = torch.cat(self.actions).detach()
        log_probs = torch.cat(self.log_probs).detach() 
        values = torch.cat(self.values).detach()
        
        returns, advantages = self.calculate_returns_and_advantages(next_state)

        for _ in range(self.ppo_epochs):
            for batch_start in range(0, len(states), self.batch_size):
                batch_end = batch_start + self.batch_size
                s = states[batch_start:batch_end]
                a = actions[batch_start:batch_end]
                lp = log_probs[batch_start:batch_end]
                v = values[batch_start:batch_end]
                r = returns[batch_start:batch_end]
                adv = advantages[batch_start:batch_end].unsqueeze(1)

                new_dist = self.actor(s)
                new_log_probs = new_dist.log_prob(a).sum(dim=-1, keepdim=True)
                ratio = (new_log_probs - lp).exp()
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                actor_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()

                new_values = self.critic(s).squeeze()
                critic_loss = nn.functional.mse_loss(new_values, r)
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.clear_buffers()

    def calculate_returns_and_advantages(self, next_state):
        rewards = self.rewards
        dones = self.dones
        values = torch.cat(self.values).squeeze().detach().cpu().numpy()
        
        next_value = self.critic(torch.FloatTensor(next_state).unsqueeze(0).to(self.device)).item()
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t+1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + values
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        return returns, advantages

    def clear_buffers(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def save_models(self, path):
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")