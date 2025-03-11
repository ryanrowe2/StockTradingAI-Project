import numpy as np
import random

class QLearningAgent:
    """
    A simple Q-learning agent for stock trading.
    """
    def __init__(self, actions=['buy', 'sell', 'hold'], alpha=0.1, gamma=0.95, epsilon=0.1):
        self.q_table = {}  # Maps state (tuple) to {action: Q-value}
        self.actions = actions
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def get_state(self, state_features):
        """
        Convert state features into a hashable tuple representation.
        """
        return tuple(state_features)

    def choose_action(self, state_features):
        """
        Choose an action based on an epsilon-greedy policy.
        """
        state = self.get_state(state_features)
        if random.random() < self.epsilon or state not in self.q_table:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_value(self, state_features, action, reward, next_state_features):
        """
        Update Q-value for a given state-action pair using the Q-learning update rule.
        """
        state = self.get_state(state_features)
        next_state = self.get_state(next_state_features)
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}
        max_future_q = max(self.q_table[next_state].values())
        current_q = self.q_table[state][action]
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q

    def simulate_episode(self, env, max_steps=100):
        """
        Simulate one episode in the trading environment.
        
        Parameters:
            env: An environment with reset() and step(action) methods.
            max_steps (int): Maximum steps in the episode.
        
        Returns:
            float: Total reward accumulated during the episode.
        """
        state = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            action = self.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            self.update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if done:
                break
        return total_reward

# A simple trading environment for testing the RL agent
class SimpleTradingEnv:
    """
    A simple simulation environment for stock trading.
    """
    def __init__(self, data):
        """
        Parameters:
            data (pd.DataFrame): Processed DataFrame containing at least 'Close', 
                                 'Open_binned', 'High_binned', 'Low_binned', and 'Market_Regime' columns.
        """
        self.data = data.reset_index(drop=True)
        self.index = 0

    def reset(self):
        self.index = 0
        return self.get_state()

    def step(self, action):
        """
        Simulate a single trading step.
        For simplicity, reward is defined as the difference in 'Close' price when buying.
        """
        current_price = self.data.loc[self.index, 'Close']
        self.index += 1
        done = self.index >= len(self.data) - 1
        next_price = self.data.loc[self.index, 'Close'] if not done else current_price
        reward = (next_price - current_price) if action == 'buy' else 0
        return self.get_state(), reward, done, {}

    def get_state(self):
        """
        Return the current state as a tuple of (Open_binned, High_binned, Low_binned, Market_Regime).
        """
        row = self.data.loc[self.index]
        open_bin = row.get('Open_binned', 0)
        high_bin = row.get('High_binned', 0)
        low_bin = row.get('Low_binned', 0)
        market_regime = row.get('Market_Regime', 0)
        return (open_bin, high_bin, low_bin, market_regime)

# Enhanced testing code for the RL agent.
if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a dummy DataFrame for demonstration purposes.
    # In practice, replace this with your actual processed data.
    data = {
        'Close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110, 108, 111, 113, 112],
        'Open_binned': [2] * 15,
        'High_binned': [2] * 15,
        'Low_binned': [2] * 15,
        'Market_Regime': [1] * 15
    }
    df_dummy = pd.DataFrame(data)
    
    # Initialize the simple trading environment and Q-learning agent.
    env = SimpleTradingEnv(df_dummy)
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.1)
    
    # Define the number of episodes and maximum steps per episode.
    episodes = 50
    max_steps = len(df_dummy)  # Use the full length of the data for each episode.
    
    # Lists to store episode rewards.
    episode_rewards = []
    
    # Run episodes and collect rewards.
    for ep in range(episodes):
        total_reward = agent.simulate_episode(env, max_steps=max_steps)
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}/{episodes}: Total Reward = {total_reward:.2f}")
    
    # Calculate summary statistics.
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")
    
    # Plot the learning curve.
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, episodes + 1), episode_rewards, marker='o', linestyle='-', color='b')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning Agent Performance')
    plt.grid(True)
    plt.show()
