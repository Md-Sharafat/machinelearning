#using dataset in q-learning
import numpy as np
import pandas as pd

data = {
    'current_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'action': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
    'next_state': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'reward': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 1 for reaching the goal state
}

df = pd.DataFrame(data)

# Define environment parameters
n_states = 16  # Number of states in the grid world
n_actions = 4  # Number of possible actions (up, down, left, right)
goal_state = 15  # Goal state

# Initialize Q-table with zeros
Q_table = np.zeros((n_states, n_actions))

# Define parameters
learning_rate = 0.8
discount_factor = 0.95
epochs = 1000

# Q-learning algorithm with dataset
for epoch in range(epochs):
    for index, row in df.iterrows():
        current_state = row['current_state']
        action = row['action']
        next_state = row['next_state']
        reward = row['reward']

        # Update Q-value using the Q-learning update rule
        Q_table[current_state, action] += learning_rate * \
            (reward + discount_factor * np.max(Q_table[next_state]) - Q_table[current_state, action])

# After training, the Q-table represents the learned Q-values
print("Learned Q-table:")
print(Q_table)

