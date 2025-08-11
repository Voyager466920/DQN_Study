import torch

from FrozenLake_Workstation.OneHotEncoding import one_hot_encoding


def optimize(mini_batch, policy_dqn, target_dqn, discount_factor, loss_fn, optimizer):
    num_states = policy_dqn.fc1.in_features

    current_q_list = []
    target_q_list = []

    for state, action, new_state, reward, terminiated in mini_batch:
        if terminiated:
            target = torch.FloatTensor([reward])
        else:
            with torch.no_grad():
                target = torch.FloatTensor(
                    reward + discount_factor * target_dqn(one_hot_encoding(new_state, num_states)).max()
                )

        current_q = policy_dqn(one_hot_encoding(state, num_states))
        current_q_list.append(current_q)

        target_q = target_dqn(one_hot_encoding(state, num_states))

        target_q[action] = target
        target_q_list.append(target_q)

    loss = loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

