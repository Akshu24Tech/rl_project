### 🤖 Reinforcement Learning Project: From Q-Learning to PPO 🧠

This project documents a journey through three fundamental reinforcement learning algorithms, demonstrating how to build, train, and evaluate intelligent agents for a variety of environments. The project progresses from simple, discrete-state problems to complex, continuous-control tasks.

-----

### 📂 Project Structure

The project is organized into modular components, making it easy to understand and extend.

```
rl_project/
├── q_learning/               # Phase 1: Q-Learning for a discrete environment
│   ├── main.py
│   ├── evaluate.py
│   └── models/
│       └── q_table.npy
├── dqn/                      # Phase 2: DQN for a continuous-state environment
│   ├── main.py
│   ├── evaluate.py
│   ├── dqn_agent.py
│   ├── dqn_model.py
│   └── replay_buffer.py
│   └── models/
│       └── dqn_model.pth
├── ppo/                      # Phase 3: PPO for a continuous-action environment
│   ├── main.py
│   ├── evaluate.py
│   ├── ppo_agent.py
│   └── networks.py
│   └── models/
│       ├── ppo_actor.pth
│       └── ppo_critic.pth
├── .gitignore
├── README.md
└── requirements.txt
```

-----

### ✨ Project Phases

#### Phase 1: Q-Learning on CartPole-v1

  * **Objective:** Train an agent to balance a pole on a cart using a Q-Table.
  * **Concepts:** This phase introduces the basics of **Q-Learning**, a value-based, off-policy algorithm. We explore the **exploration-exploitation trade-off** and the challenge of discretizing a continuous state space.
  * **Environment:** `CartPole-v1` (discrete action space, continuous state space).
  * **Files:** `q_learning/main.py` (training with discretization), `q_learning/evaluate.py` (evaluation).

#### Phase 2: Deep Q-Network (DQN) on LunarLander-v2

  * **Objective:** Train an agent to safely land a lunar module, handling a more complex state space.
  * **Concepts:** We transition to **Deep Reinforcement Learning** by using a neural network to approximate the Q-function. This phase introduces key techniques for stable training: **Experience Replay** and a **Target Network**.
  * **Environment:** `LunarLander-v2` (discrete action space, continuous state space).
  * **Files:** `dqn/main.py`, `dqn/evaluate.py`, `dqn/dqn_model.py`, `dqn/replay_buffer.py`, and `dqn/dqn_agent.py`.

#### Phase 3: Proximal Policy Optimization (PPO) on BipedalWalker-v3

  * **Objective:** Train a bipedal robot to walk, a task with a continuous action space.
  * **Concepts:** We move to **Policy-Based Algorithms** with **PPO**, an on-policy, actor-critic method. This phase demonstrates how to handle **continuous action spaces** and uses techniques like **Generalized Advantage Estimation (GAE)** for stable learning.
  * **Environment:** `BipedalWalker-v3` (continuous action space, continuous state space).
  * **Files:** `ppo/main.py`, `ppo/evaluate.py`, `ppo/networks.py`, and `ppo/ppo_agent.py`.

-----

### 🔧 Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/rl_project.git
    cd rl_project
    ```

2.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: You'll need to create a `requirements.txt` file by running `pip freeze > requirements.txt` after installing all the libraries we used: `gymnasium`, `numpy`, `torch`.)*

-----

### 🚀 How to Run the Project

Navigate to the directory for the phase you want to run (e.g., `cd ppo/`).

1.  **Train the Agent:** Run the main training script. This will train the model and save the learned weights to the `models/` directory.

    ```bash
    python main.py
    ```

2.  **Evaluate the Agent:** Run the evaluation script to see the trained agent in action.

    ```bash
    python evaluate.py
    ```

-----

### 🙏 Acknowledgments

  * **OpenAI Gymnasium:** For providing a standardized and powerful set of reinforcement learning environments.
  * **PyTorch:** For a flexible and efficient framework for building deep learning models.
  * **Akshu Grewal:** For the hard work and dedication in completing this project.