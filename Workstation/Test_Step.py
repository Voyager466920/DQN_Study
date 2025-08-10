import torch
import random

def test_step(env, policy_dqn, device, epsilon=0.05, episodes=10, render=False):
    total_reward = 0.0
    for _ in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device)
        terminated, truncated = False, False
        episode_reward = 0.0

        while not (terminated or truncated):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_dqn(state.unsqueeze(0)).argmax().item()

            new_state, reward, terminated, truncated, _ = env.step(action)
            state = torch.tensor(new_state, dtype=torch.float, device=device)
            episode_reward += reward

        total_reward += episode_reward

    return total_reward / episodes
