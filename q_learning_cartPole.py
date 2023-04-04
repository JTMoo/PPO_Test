import gym
import numpy as np

# Initialize the environment
env = gym.make('CartPole-v0')

# Set the hyperparameters
num_episodes = 10000
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.99

# Initialize the Q-table
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# Loop over the episodes
for i in range(num_episodes):
    # Reset the environment
    state = env.reset()
    done = False
    
    # Loop over the steps in the episode
    while not done:
        # Choose an action using epsilon-greedy policy
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        # Take the chosen action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)
        
        # Update the Q-value using Q-learning
        Q[state][action] = (1 - learning_rate) * Q[state][action] + \
                           learning_rate * (reward + discount_factor * np.max(Q[next_state]))
        
        # Update the state
        state = next_state
    
    # Decay epsilon
    epsilon *= epsilon_decay

# Evaluate the agent
state = env.reset()
done = False
total_reward = 0
while not done:
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    total_reward += reward
env.close()

print("Total reward:", total_reward)
