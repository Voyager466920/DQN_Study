import random
import torch
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt

from Optimize import optimize
from OneHotEncoding import one_hot_encoding
from Model import DQN


def train_step(env, episodes, memory, learning_rate, mini_batch_size, discount_factor, network_sync, loss_fn):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    epsilon = 1
    epsilon_history = []
    step_count = 0
    rewards_per_episode = np.zeros(episodes)

    policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
    target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

    target_dqn.load_state_dict(policy_dqn.state_dict())  # 복사

    optimizer = Adam(policy_dqn.parameters(), lr=learning_rate)

    for i in range(episodes):
        state = env.reset()[0]  # initialize
        terminated = False
        truncated = False

        while (not terminated and not truncated): # epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_dqn(one_hot_encoding(state, num_states)).argmax().item()

            new_state, reward, terminated, truncated, _ = env.step(action)
            memory.append((state, action, new_state, reward, terminated))
            state = new_state
            step_count += 1

        if reward == 1:
            rewards_per_episode[i] = 1

        if len(memory) > mini_batch_size and np.sum(rewards_per_episode) > 0:
            mini_batch = memory.sample(mini_batch_size)
            optimize(mini_batch, policy_dqn, target_dqn, discount_factor, loss_fn, optimizer) # 역전파

            epsilon = max(epsilon - 1/episodes, 0)
            epsilon_history.append(epsilon)

            if step_count > network_sync:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_count = 0

    env.close()
    torch.save(policy_dqn.state_dict(), "frozen_lake_dqn.pt")

    plt.figure(1)

    sum_rewards = np.zeros(episodes)
    for x in range(episodes):
        sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
    plt.subplot(121)
    plt.plot(sum_rewards)

    plt.subplot(122)
    plt.plot(epsilon_history)

    plt.savefig("frozen_lake_dqn.png")

