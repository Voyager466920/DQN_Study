import torch


def train_step(mini_batch, policy_dqn, target_dqn, optimizer, loss_fn, gamma, device, double_dqn=False):

    states, actions, new_states, rewards, terminations = zip(*mini_batch)

    states = torch.stack(states)
    actions = torch.stack(actions)
    new_states = torch.stack(new_states)
    rewards = torch.stack(rewards)
    terminations = torch.tensor(terminations).float().to(device)

    with torch.no_grad():
        if double_dqn:
            best_actions = policy_dqn(new_states).argmax(dim=1)
            target_q = rewards + (1 - terminations) * gamma * \
                       target_dqn(new_states).gather(1, best_actions.unsqueeze(1)).squeeze()
        else:
            target_q = rewards + (1 - terminations) * gamma * target_dqn(new_states).max(dim=1)[0]

    current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
    loss = loss_fn(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
