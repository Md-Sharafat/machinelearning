import numpy as np

# Define the environment
n_states = 16  
n_actions = 6  
goal_state = 15  

# Initialize Q-table with zeros
Q_table = np.zeros((n_states, n_actions))

# Define parameters
learning_rate = 0.8
discount_factor = 0.85
exploration_prob = 0.1
epochs = 100

# Q-learning algorithm
for epoch in range(epochs):
    current_state = np.random.randint(0, n_states)  # Start from a random state

    while current_state != goal_state:

        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, n_actions)  # Explore
        else:
            action = np.argmax(Q_table[current_state])  # Exploit


        next_state = (current_state + 1) % n_states

        reward = 1 if next_state == goal_state else 0


        Q_table[current_state, action] += learning_rate * \
            (reward + discount_factor *
             np.max(Q_table[next_state]) - Q_table[current_state, action])

        current_state = next_state

print("Learned Q-table:")
print(Q_table)

