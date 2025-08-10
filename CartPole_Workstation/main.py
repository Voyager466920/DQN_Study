import itertools
import random
import torch
from Agent import Agent
from Train_Step import train_step
from Test_Step import test_step

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    agent = Agent(
        gymnasium_id="CartPole-v1",
        replay_memory_size=10000,
        mini_batch_size=64,
        epsilon_init=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        network_sync_rate=1000,
        learning_rate_a=0.001,
        discount_factor_g=0.99,
        enable_double_dqn=True
    )

    env, policy_dqn, target_dqn, optimizer, memory = agent.setup(render=False)

    epsilon = agent.epsilon_init
    step_count = 0

    for episode in itertools.count():
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
            new_state_t = torch.tensor(new_state, dtype=torch.float, device=device)
            reward_t = torch.tensor(reward, dtype=torch.float, device=device)

            memory.append((state, torch.tensor(action, dtype=torch.int64, device=device),
                           new_state_t, reward_t, terminated or truncated))
            step_count += 1
            episode_reward += reward
            state = new_state_t

            if len(memory) >= agent.mini_batch_size:
                mini_batch = memory.sample(agent.mini_batch_size)
                train_step(mini_batch, policy_dqn, target_dqn, optimizer,
                           torch.nn.MSELoss(), agent.discount_factor_g,
                           device, agent.double_dqn)

                if step_count >= agent.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        epsilon = max(epsilon * agent.epsilon_decay, agent.epsilon_min)

        if episode % 10 == 0:
            avg_reward = test_step(env, policy_dqn, device, epsilon=0.0)
            print(f"[Test] Episode {episode} - Avg Reward: {avg_reward}")
