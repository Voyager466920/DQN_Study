import torch.nn as nn

import gymnasium as gym

from Test_Step import test_step
from Train_Step import train_step
from ReplayMemory import ReplayMemory

LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.9
NETWORK_SYNC = 10               # number of steps the agent takes before syncing the policy and target network
REPLAY_MEMORY_SIZE = 1000
MINI_BATCH_SIZE = 32

IS_SLIPPERY = False
RENDER = False
EPISODES = 1000

ACTIONS = ['L', 'D', 'R', 'U']

loss_fn = nn.MSELoss()

env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=IS_SLIPPERY, render_mode="human" if RENDER else None)
memory = ReplayMemory(REPLAY_MEMORY_SIZE)

train_step(env, EPISODES, memory, LEARNING_RATE, MINI_BATCH_SIZE, DISCOUNT_FACTOR, NETWORK_SYNC, loss_fn )
test_step(env, EPISODES)

