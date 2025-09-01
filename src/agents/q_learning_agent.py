import numpy as np

class QLearningAgent:
    def __init__(self, observation_space_size, action_space_size, learning_rate, gamma, epsilon):
        self.q_table = np.zeros((observation_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon # For exploration-exploitation
    
    def choose_action(self, state):
        pass
    
    def learn(self, state, action, reward, next_state):
        pass