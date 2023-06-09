# This is a repository for testing purposes only
# Table of Contents
1. [Deep reinforcement learning](#deep-reinforcement-learning)
2. [Policies](#policies)
3. [Agent](#agent)
4. [Algorithms](#algorithms)
5. [The PPO](#ppo)

## Deep reinforcement learning
Reinforcement learning is a type of machine learning that is used to train an artificial intelligence (AI) agent to make decisions based on trial and error.
In reinforcement learning, the AI agent interacts with an environment and learns from the feedback it receives in the form of rewards or punishments.

## Policies
In the context of reinforcement learning, a policy is a function that maps states to actions. The policy specifies the behavior of an agent in an environment by telling the agent what action to take in each state.
### Types of Policies
1. LuT
2. NN (Deep RL)
3. Rule-set
### On- and off-policy
Off-policy learning means that the learning algorithm can learn from experience generated by any policy, not just the one it is currently following. This is useful because it allows the algorithm to learn from a broader range of experience and to reuse past experience collected from other policies. In contrast, on-policy learning algorithms can only learn from experience generated by the current policy.


## Agent
In reinforcement learning, an agent is an entity that interacts with an environment to learn a policy that maximizes the expected cumulative reward. The agent can be thought of as an intelligent decision-making system that observes the current state of the environment and chooses an action to take based on that observation.
### Model-free and model-based
Model-free learning means that the algorithm does not rely on a model of the environment's dynamics to learn the optimal policy. Instead, it learns directly from experience by observing the rewards and transitions that result from its actions. In model-free learning, the goal is to learn a policy or a value function that maximizes the expected cumulative reward, without explicitly modeling the transition probabilities and rewards of the environment.

On the other hand, model-based learning algorithms explicitly learn a model of the environment's dynamics, including the transition probabilities and the rewards. This model can then be used to simulate different trajectories and estimate the expected cumulative reward under different policies.


## Algorithms
### Q-Learning
Q-Learning is a model-free, off-policy algorithm that learns the optimal Q-value function by iteratively updating the Q-values based on the observed rewards and transitions.
### SARSA
SARSA is another model-free, on-policy algorithm that learns the Q-value function by iteratively updating the Q-values based on the observed rewards and transitions.
### Deep Q-Networks (DQN)
DQN is a deep reinforcement learning algorithm that uses a neural network to represent the Q-value function. The neural network is trained using a variant of Q-Learning that uses experience replay and a target network to improve stability during training.
### Policy Gradient Methods
Policy gradient methods learn a policy directly by optimizing a surrogate objective function that maximizes the expected cumulative reward. Examples of policy gradient methods include REINFORCE and Proximal Policy Optimization (PPO).
### Actor-Critic Methods
Actor-Critic methods combine the benefits of both policy gradient methods and value-based methods by using two networks - an actor network that learns the policy and a critic network that learns the value function. Examples of actor-critic methods include Advantage Actor-Critic (A2C) and Deep Deterministic Policy Gradient (DDPG).
### Monte Carlo Methods
Monte Carlo methods learn the value function by sampling complete trajectories from the environment and using them to estimate the expected cumulative reward. Examples of Monte Carlo methods include Monte Carlo policy evaluation and Monte Carlo Tree Search.

## PPO
PPO stands for Proximal Policy Optimization, which is a deep reinforcement learning algorithm used for training artificial intelligence (AI) agents. 
It is a model-free, on-policy algorithm that is designed to optimize the policy of the agent in a way that maximizes the expected reward.
