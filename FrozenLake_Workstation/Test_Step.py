import torch

from StateToDQNInput import state_to_dqn_input
from Model import DQN


def test_step(env, episodes):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    policy_dqn = DQN(num_states, num_states, num_actions)
    policy_dqn.load_state_dict(torch.load('frozen_lake_dqn.pt'))
    policy_dqn.eval()

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while (not terminated and not truncated):
            with torch.no_grad():
                action = policy_dqn(state_to_dqn_input(state, num_states)).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)

    env.close()
